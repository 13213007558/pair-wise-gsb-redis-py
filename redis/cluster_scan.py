from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Literal

from redis.exceptions import RedisClusterException
from redis.typing import PatternT
from redis.utils import str_if_bytes

CLUSTER_SCAN_SESSION_VERSION = 1
ClusterScanTargetScope = Literal["primaries", "replicas", "all", "explicit"]


# A session records enough state to continue each node's SCAN after a pause. It
# is neither a point-in-time snapshot nor a dedup layer while keys or slots are
# moving: Redis SCAN itself can return keys whose topology changed between
# batches. The resume path preserves each reachable node's cursor so it does
# not introduce the additional duplicate of restarting that node from zero.


class ClusterScanTopologyError(RedisClusterException):
    """Raised when resuming a cluster scan requires a topology decision."""

    def __init__(
        self,
        message: str,
        changes: "ClusterScanTopologyChanges",
        unfinished_ranges: list[tuple[str, list[tuple[int, int]]]],
    ) -> None:
        super().__init__(message)
        self.changes = changes
        self.unfinished_ranges = unfinished_ranges


@dataclass(slots=True)
class ClusterScanNode:
    node_id: str
    address: str
    cursor: int = 0
    completed: bool = False
    slots: list[tuple[int, int]] = field(default_factory=list)
    state: Literal["active", "missing"] = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "cursor": self.cursor,
            "completed": self.completed,
            "slots": [[start, end] for start, end in self.slots],
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClusterScanNode":
        try:
            slots = [
                (int(slot_range[0]), int(slot_range[1]))
                for slot_range in value["slots"]
            ]
            node = cls(
                node_id=str(value["node_id"]),
                address=str(value["address"]),
                cursor=int(value["cursor"]),
                completed=bool(value["completed"]),
                slots=slots,
                state=value.get("state", "active"),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Invalid cluster scan node state") from exc
        if node.state not in ("active", "missing"):
            raise ValueError("Invalid cluster scan node state")
        if node.cursor < 0 or any(start > end for start, end in node.slots):
            raise ValueError("Invalid cluster scan node state")
        if node.completed and node.cursor != 0:
            raise ValueError("Completed cluster scan node cannot retain a cursor")
        return node


@dataclass(slots=True)
class ClusterScanSession:
    match: PatternT | None = None
    count: int | None = None
    type: str | None = None
    target_scope: ClusterScanTargetScope = "primaries"
    nodes: dict[str, ClusterScanNode] = field(default_factory=dict)
    version: int = CLUSTER_SCAN_SESSION_VERSION

    @property
    def complete(self) -> bool:
        return bool(self.nodes) and all(
            node.completed and node.state == "active" for node in self.nodes.values()
        )

    def to_dict(self) -> dict[str, Any]:
        match: Any = None
        if isinstance(self.match, (bytes, bytearray, memoryview)):
            match = {
                "encoding": "base64",
                "value": base64.b64encode(bytes(self.match)).decode("ascii"),
            }
        else:
            match = self.match

        return {
            "version": self.version,
            "match": match,
            "count": self.count,
            "type": self.type,
            "target_scope": self.target_scope,
            "nodes": [node.to_dict() for node in self.nodes.values()],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClusterScanSession":
        try:
            version = int(value["version"])
            match = value["match"]
            if isinstance(match, dict) and match.get("encoding") == "base64":
                match = base64.b64decode(match["value"])
            count = value["count"]
            scan_type = value["type"]
            target_scope = value["target_scope"]
            nodes = {
                node.node_id: node
                for node in (ClusterScanNode.from_dict(item) for item in value["nodes"])
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid cluster scan session") from exc

        if version != CLUSTER_SCAN_SESSION_VERSION:
            raise ValueError("Unsupported cluster scan session version")
        if match is not None and not isinstance(
            match, (str, bytes, bytearray, memoryview)
        ):
            raise ValueError("Invalid cluster scan MATCH option")
        if count is not None and (not isinstance(count, int) or count < 0):
            raise ValueError("Invalid cluster scan COUNT option")
        if scan_type is not None and not isinstance(scan_type, str):
            raise ValueError("Invalid cluster scan TYPE option")
        if target_scope not in ("primaries", "replicas", "all", "explicit"):
            raise ValueError("Invalid cluster scan target scope")
        if not nodes:
            raise ValueError("Cluster scan session must contain nodes")

        return cls(
            match=match,
            count=count,
            type=scan_type,
            target_scope=target_scope,
            nodes=nodes,
            version=version,
        )


@dataclass(slots=True)
class ClusterScanTopologyChanges:
    address_changes: dict[str, tuple[str, str]] = field(default_factory=dict)
    slot_owner_changes: list[tuple[int, int, str | None, str | None]] = field(
        default_factory=list
    )
    missing_node_ids: list[str] = field(default_factory=list)
    new_node_ids: list[str] = field(default_factory=list)


def cluster_node_id(node: Any) -> str:
    node_id = getattr(node, "node_id", None)
    if node_id is not None:
        return str_if_bytes(node_id)
    return str(node.name)


def validate_cluster_scan_options(
    session: ClusterScanSession,
    match: PatternT | None = None,
    count: int | None = None,
    _type: str | None = None,
) -> None:
    if match != session.match or count != session.count or _type != session.type:
        raise ValueError(
            "MATCH, COUNT and TYPE are bound to a cluster scan session and "
            "cannot be changed when resuming it"
        )


def create_cluster_scan_session(
    client: Any,
    match: PatternT | None = None,
    count: int | None = None,
    _type: str | None = None,
    target_nodes: Any = None,
) -> ClusterScanSession:
    if match is not None and not isinstance(match, (str, bytes, bytearray, memoryview)):
        raise ValueError("Invalid cluster scan MATCH option")
    if count is not None and (not isinstance(count, int) or count < 0):
        raise ValueError("Invalid cluster scan COUNT option")
    if _type is not None and not isinstance(_type, str):
        raise ValueError("Invalid cluster scan TYPE option")

    scope, nodes = _scope_nodes(client, target_nodes)
    session = ClusterScanSession(
        match=match,
        count=count,
        type=_type,
        target_scope=scope,
    )
    slot_ranges = _slot_ranges_for_nodes(
        client, {cluster_node_id(node): node for node in nodes}, scope == "primaries"
    )
    for node in nodes:
        node_id = cluster_node_id(node)
        session.nodes[node_id] = ClusterScanNode(
            node_id=node_id,
            address=node.name,
            slots=slot_ranges[node_id],
        )
    if not session.nodes:
        raise ValueError("A cluster scan session must contain at least one node")
    return session


def refresh_cluster_scan_session(
    client: Any,
    session: ClusterScanSession,
    *,
    strict: bool = True,
    include_new_nodes: bool = False,
) -> ClusterScanTopologyChanges:
    _, current_nodes = _scope_nodes(
        client,
        session.target_scope,
        session.nodes if session.target_scope == "explicit" else None,
    )
    current_by_id = {cluster_node_id(node): node for node in current_nodes}
    changes = ClusterScanTopologyChanges()

    for node_id, node_state in session.nodes.items():
        current_node = current_by_id.get(node_id)
        if current_node is None:
            node_state.state = "missing"
            changes.missing_node_ids.append(node_id)
            continue

        node_state.state = "active"
        if current_node.name != node_state.address:
            changes.address_changes[node_id] = (
                node_state.address,
                current_node.name,
            )
            node_state.address = current_node.name

    changes.slot_owner_changes = _slot_owner_changes(client, session, current_by_id)

    for node_id, node in current_by_id.items():
        if node_id not in session.nodes:
            changes.new_node_ids.append(node_id)
            if include_new_nodes:
                session.nodes[node_id] = ClusterScanNode(
                    node_id=node_id,
                    address=node.name,
                    slots=_slot_ranges_for_nodes(
                        client, current_by_id, session.target_scope == "primaries"
                    )[node_id],
                )

    changes.missing_node_ids.sort()
    changes.new_node_ids.sort()

    unexpected_new_nodes = [
        node_id for node_id in changes.new_node_ids if node_id not in session.nodes
    ]
    if (strict and changes.missing_node_ids) or unexpected_new_nodes:
        unfinished_ranges = [
            (node_id, list(session.nodes[node_id].slots))
            for node_id in changes.missing_node_ids
            if not session.nodes[node_id].completed
        ]
        details = []
        if changes.missing_node_ids:
            details.append(f"missing nodes: {', '.join(changes.missing_node_ids)}")
        if unexpected_new_nodes:
            details.append(
                "new nodes require include_new_nodes=True: "
                + ", ".join(unexpected_new_nodes)
            )
        raise ClusterScanTopologyError("; ".join(details), changes, unfinished_ranges)

    return changes


def update_cluster_scan_node(
    session: ClusterScanSession, node_id: str, address: str, cursor: int
) -> None:
    node = session.nodes.get(node_id)
    if node is None or node.state != "active" or node.address != address:
        return
    node.cursor = cursor
    node.completed = cursor == 0


def _scope_nodes(
    client: Any,
    target_nodes: Any,
    explicit_nodes: dict[str, ClusterScanNode] | None = None,
) -> tuple[ClusterScanTargetScope, list[Any]]:
    if explicit_nodes is not None and not isinstance(
        target_nodes, (dict, list, tuple, set)
    ):
        node_ids = set(explicit_nodes)
        nodes = [
            node for node in client.get_nodes() if cluster_node_id(node) in node_ids
        ]
        return "explicit", nodes

    if target_nodes is None or target_nodes == "primaries":
        return "primaries", list(client.get_primaries())
    if target_nodes == "replicas":
        return "replicas", list(client.get_replicas())
    if target_nodes == "all":
        return "all", list(client.get_nodes())
    if isinstance(target_nodes, dict):
        nodes = list(target_nodes.values())
    elif isinstance(target_nodes, (list, tuple, set)):
        nodes = list(target_nodes)
    else:
        if isinstance(target_nodes, str):
            raise ValueError(
                "Cluster scan sessions support primaries, replicas, all or an "
                "explicit set of nodes"
            )
        nodes = [target_nodes]
    return "explicit", nodes


def _slot_ranges_for_nodes(
    client: Any, nodes: dict[str, Any], primary_only: bool
) -> dict[str, list[tuple[int, int]]]:
    ranges = {node_id: [] for node_id in nodes}
    active_id: str | None = None
    start = 0

    def append_range(node_id: str | None, end: int) -> None:
        if node_id in ranges:
            ranges[node_id].append((start, end))

    for slot in range(16384):
        slot_nodes = client.nodes_manager.slots_cache.get(slot)
        if slot_nodes is not None and slot_nodes:
            if primary_only:
                node = slot_nodes[0]
                node_id = cluster_node_id(node)
            else:
                node = next(
                    (
                        candidate
                        for candidate in slot_nodes
                        if cluster_node_id(candidate) in nodes
                    ),
                    None,
                )
                node_id = cluster_node_id(node) if node is not None else None
        else:
            node_id = None

        if node_id != active_id:
            if slot > 0 and active_id is not None:
                append_range(active_id, slot - 1)
            active_id = node_id
            start = slot

    if active_id is not None:
        append_range(active_id, 16383)
    return ranges


def _slot_owner_changes(
    client: Any,
    session: ClusterScanSession,
    current_by_id: dict[str, Any],
) -> list[tuple[int, int, str | None, str | None]]:
    if session.target_scope != "primaries":
        ranges = _slot_ranges_for_nodes(client, current_by_id, False)
        for node_id, slot_ranges in ranges.items():
            if node_id in session.nodes:
                session.nodes[node_id].slots = slot_ranges
        return []

    old_owner: dict[int, str] = {}
    for node_id, node_state in session.nodes.items():
        for start, end in node_state.slots:
            for slot in range(start, end + 1):
                old_owner[slot] = node_id

    changes: list[tuple[int, str | None, str | None]] = []
    start: int | None = None
    previous: int | None = None
    pair: tuple[str | None, str | None] | None = None

    def finish_range(end_slot: int) -> None:
        nonlocal start, previous, pair
        if start is not None and pair is not None:
            changes.append((start, end_slot, pair[0], pair[1]))
        start = None
        previous = None
        pair = None

    for slot in range(16384):
        slot_nodes = client.nodes_manager.slots_cache.get(slot)
        current_node = slot_nodes[0] if slot_nodes is not None and slot_nodes else None
        current_id = cluster_node_id(current_node) if current_node is not None else None
        current_pair = (old_owner.get(slot), current_id)
        different = current_pair[0] != current_pair[1]
        if different:
            adjacent = previous is not None and slot == previous + 1
            if start is None or current_pair != pair or not adjacent:
                if start is not None and previous is not None:
                    finish_range(previous)
                start = slot
                pair = current_pair
            previous = slot
        elif start is not None:
            finish_range(previous or start)

    if start is not None and previous is not None:
        finish_range(previous)

    ranges = _slot_ranges_for_nodes(client, current_by_id, True)
    for node_id, slot_ranges in ranges.items():
        if node_id in session.nodes:
            session.nodes[node_id].slots = slot_ranges
    return changes
