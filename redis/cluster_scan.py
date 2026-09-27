"""
Resumable cluster-wide SCAN sessions.

A single SCAN cursor cannot capture the progress of a cluster-wide key
scan, because every primary owns an independent cursor. This module provides
:class:`ClusterScanSession`, a serializable record of one cursor per
participating node, and the iterators that drive it:

- :func:`cluster_scan` starts a fresh session over the current primaries.
- :func:`resume_cluster_scan` continues a previously saved session.

The session binds the ``MATCH``/``COUNT``/``TYPE`` conditions it was created
with; resuming with different conditions raises
:exc:`ClusterScanSessionMismatch` instead of silently changing the scan.

Guarantees and non-guarantees
-----------------------------

A node is finished only when the server returns cursor ``0`` for it. An
empty batch with a non-zero cursor does **not** complete the node; iteration
continues until the cursor wraps to ``0``.

Redis SCAN itself guarantees only that a key present on a node for the
entire duration of that node's iteration is returned at least once. It may
legitimately return a key more than once (for example while the hash table
rehashes, or when a key migrates back and forth). Those duplicates come from
Redis SCAN itself.

The resumer adds no duplicates of its own: a node's cursor advances only
after a batch has been fetched, and the not-yet-yielded remainder of an
in-flight batch is kept inside the session (``pending``) and replayed
exactly once on resume. Pausing and resuming therefore neither loses nor
repeats keys beyond what Redis SCAN itself would produce in one uninterrupted
run.

Under concurrent slot migration the same key may be returned by two
different nodes (once by the source before the move, once by the target
after). That is inherent to scanning a changing keyspace: this module does
not provide snapshot isolation across the cluster and does not deduplicate
keys, and callers that need exactly-once processing must deduplicate
themselves.
"""

import base64
import json
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from redis.typing import PatternT
from redis.utils import safe_str

if TYPE_CHECKING:
    from redis.cluster import ClusterNode, RedisCluster

SESSION_FORMAT = "redis-py-cluster-scan"
SESSION_VERSION = 1

_UNSET: Any = object()


class ClusterScanError(Exception):
    """Base class for cluster scan session errors."""


class ClusterScanSessionMismatch(ClusterScanError, ValueError):
    """
    A session was resumed with ``MATCH``/``COUNT``/``TYPE`` conditions that
    differ from the ones the session was created with.
    """


class ClusterScanSessionInvalid(ClusterScanError, ValueError):
    """A serialized session payload is malformed or of an unsupported version."""


class ClusterScanTopologyError(ClusterScanError):
    """
    Strict mode: one or more nodes that still had unfinished cursors are no
    longer part of the cluster, so the key ranges they had not scanned yet
    cannot be covered by this session.

    The missing node states are available on :attr:`missing`.
    """

    def __init__(self, missing: list["NodeScanState"]) -> None:
        self.missing = list(missing)
        ranges = ", ".join(
            f"{state.node_id} (last seen at {state.address}, cursor={state.cursor})"
            for state in self.missing
        )
        super().__init__(
            "Cluster topology changed: these nodes left the cluster with "
            f"unfinished scan ranges: {ranges}. Resume with strict=False to "
            "skip them, or start a new scan."
        )


def _encode_item(item: bytes | str) -> dict[str, str]:
    if isinstance(item, bytes):
        return {"b64": base64.b64encode(item).decode("ascii")}
    return {"str": item}


def _decode_item(data: dict[str, str]) -> bytes | str:
    if "b64" in data:
        return base64.b64decode(data["b64"])
    return data["str"]


@dataclass
class NodeScanState:
    """
    Scan progress for one cluster node.

    ``node_id`` is the node's stable identity (``CLUSTER MYID``), which
    survives address changes; ``address`` is the ``host:port`` it was last
    contacted at. ``done`` is set only when the server returns cursor ``0``;
    ``pending`` holds the fetched-but-not-yet-yielded remainder of the
    in-flight batch so that pausing mid-batch loses nothing on resume.
    """

    node_id: str
    address: str
    cursor: int = 0
    done: bool = False
    pending: deque = field(default_factory=deque)

    @property
    def finished(self) -> bool:
        return self.done and not self.pending

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "cursor": self.cursor,
            "done": self.done,
            "pending": [_encode_item(item) for item in self.pending],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeScanState":
        return cls(
            node_id=data["node_id"],
            address=data["address"],
            cursor=data["cursor"],
            done=data["done"],
            pending=deque(_decode_item(item) for item in data["pending"]),
        )


class ClusterScanSession:
    """
    Serializable state of a resumable cluster-wide SCAN.

    Records the bound scan conditions (``match``/``count``/``type``) and one
    :class:`NodeScanState` per participating node. Serialize with
    :meth:`dumps`/:meth:`to_dict`, restore with :meth:`loads`/:meth:`from_dict`,
    and continue scanning with ``resume_cluster_scan``.
    """

    def __init__(
        self,
        match: PatternT | None = None,
        count: int | None = None,
        type: str | None = None,
        node_states: dict[str, NodeScanState] | None = None,
    ) -> None:
        self.match = match
        self.count = count
        self.type = type
        self.nodes: dict[str, NodeScanState] = dict(node_states or {})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusterScanSession):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @property
    def is_complete(self) -> bool:
        """True once every participating node returned cursor ``0`` and all
        buffered batches were consumed."""
        return all(state.finished for state in self.nodes.values())

    @property
    def unfinished_nodes(self) -> list[NodeScanState]:
        """Nodes whose scan has not run to cursor ``0`` yet."""
        return [state for state in self.nodes.values() if not state.done]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SESSION_FORMAT,
            "version": SESSION_VERSION,
            "match": _encode_item(self.match) if self.match is not None else None,
            "count": self.count,
            "type": self.type,
            "nodes": [state.to_dict() for state in self.nodes.values()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClusterScanSession":
        try:
            if payload["format"] != SESSION_FORMAT:
                raise ClusterScanSessionInvalid(
                    f"Not a cluster scan session: {payload['format']!r}"
                )
            if payload["version"] != SESSION_VERSION:
                raise ClusterScanSessionInvalid(
                    f"Unsupported cluster scan session version: "
                    f"{payload['version']!r}"
                )
            match = payload["match"]
            nodes = [NodeScanState.from_dict(state) for state in payload["nodes"]]
            return cls(
                match=_decode_item(match) if match is not None else None,
                count=payload["count"],
                type=payload["type"],
                node_states={state.node_id: state for state in nodes},
            )
        except (KeyError, TypeError, AttributeError) as e:
            raise ClusterScanSessionInvalid(
                f"Malformed cluster scan session payload: {e}"
            ) from e

    def dumps(self) -> str:
        """Serialize the session to a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def loads(cls, payload: str | bytes) -> "ClusterScanSession":
        """Restore a session serialized with :meth:`dumps`."""
        try:
            data = json.loads(payload)
        except (TypeError, ValueError) as e:
            raise ClusterScanSessionInvalid(
                f"Malformed cluster scan session payload: {e}"
            ) from e
        return cls.from_dict(data)


@dataclass(frozen=True)
class AddressChange:
    """A session node whose ``host:port`` changed since it was saved."""

    node_id: str
    old_address: str
    new_address: str


@dataclass(frozen=True)
class TopologyChanges:
    """
    How the current cluster topology differs from a saved session.

    ``continued`` lists node ids found again (possibly at a new address),
    ``address_changed`` details the address moves, ``missing`` the session
    nodes no longer in the cluster, and ``added`` the ``(node_id, address)``
    pairs the session did not know about.
    """

    continued: tuple[str, ...]
    address_changed: tuple[AddressChange, ...]
    missing: tuple[NodeScanState, ...]
    added: tuple[tuple[str, str], ...]


def reconcile(
    session: ClusterScanSession,
    current: dict[str, "ClusterNode"],
    *,
    strict: bool,
    include_new_nodes: bool,
) -> tuple[list[tuple[NodeScanState, "ClusterNode"]], TopologyChanges]:
    """
    Match a saved session against the current ``{node_id: node}`` map.

    Returns the scan plan (node states paired with their current node) and
    the observed topology changes. Address changes are followed silently -
    the node identity is stable, so its cursor stays valid. In strict mode,
    missing nodes with unfinished ranges raise :exc:`ClusterScanTopologyError`;
    new nodes are only added to the session when ``include_new_nodes`` is set.
    """
    plan = []
    continued = []
    address_changed = []
    missing = []
    for node_id, state in session.nodes.items():
        node = current.get(node_id)
        if node is None:
            missing.append(state)
            continue
        if state.address != node.name:
            address_changed.append(AddressChange(node_id, state.address, node.name))
            state.address = node.name
        continued.append(node_id)
        plan.append((state, node))

    unfinished_missing = [state for state in missing if not state.done]
    if strict and unfinished_missing:
        raise ClusterScanTopologyError(unfinished_missing)

    added = []
    for node_id, node in current.items():
        if node_id in session.nodes:
            continue
        added.append((node_id, node.name))
        if include_new_nodes:
            state = NodeScanState(node_id=node_id, address=node.name)
            session.nodes[node_id] = state
            plan.append((state, node))

    changes = TopologyChanges(
        continued=tuple(continued),
        address_changed=tuple(address_changed),
        missing=tuple(missing),
        added=tuple(added),
    )
    return plan, changes


def _current_primaries(client: "RedisCluster") -> dict[str, "ClusterNode"]:
    current = {}
    for node in client.get_primaries():
        node_id = safe_str(client.cluster_myid(target_node=node))
        current[node_id] = node
    return current


def _check_conditions(
    session: ClusterScanSession, match: Any, count: Any, type: Any
) -> None:
    for name, given, bound in (
        ("match", match, session.match),
        ("count", count, session.count),
        ("_type", type, session.type),
    ):
        if given is not _UNSET and given != bound:
            raise ClusterScanSessionMismatch(
                f"Cannot resume a cluster scan session with {name}={given!r}: "
                f"the session is bound to {name}={bound!r}. Start a new scan "
                "to change the scan conditions."
            )


class ClusterScanIterator:
    """
    Drives a :class:`ClusterScanSession` to completion, one key at a time.

    The session is mutated as iteration progresses and can be serialized at
    any point via ``iterator.session.dumps()``; resuming the serialized
    session continues exactly where iteration stopped.
    """

    def __init__(
        self,
        client: "RedisCluster",
        session: ClusterScanSession,
        plan: list[tuple[NodeScanState, "ClusterNode"]],
        topology_changes: TopologyChanges,
    ) -> None:
        self._client = client
        self.session = session
        self.topology_changes = topology_changes
        self._plan = plan
        self._index = 0

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> bytes | str:
        while self._index < len(self._plan):
            state, node = self._plan[self._index]
            if state.pending:
                return state.pending.popleft()
            if state.done:
                self._index += 1
                continue
            self._fetch(state, node)
        raise StopIteration

    def _fetch(self, state: NodeScanState, node: "ClusterNode") -> None:
        cursors, data = self._client.scan(
            cursor=state.cursor,
            match=self.session.match,
            count=self.session.count,
            _type=self.session.type,
            target_nodes=node,
        )
        state.cursor = cursors[node.name]
        # An empty batch with a non-zero cursor does not finish the node;
        # only a cursor of 0 does.
        state.done = state.cursor == 0
        state.pending.extend(data)


def cluster_scan(
    client: "RedisCluster",
    match: PatternT | None = None,
    count: int | None = None,
    _type: str | None = None,
) -> ClusterScanIterator:
    """
    Start a new resumable cluster-wide scan over the current primaries.

    ``match``/``count``/``_type`` behave as in ``RedisCluster.scan`` and are
    bound to the returned session. Iterate the result to yield keys; serialize
    ``iterator.session`` at any point to make the progress resumable.
    """
    session = ClusterScanSession(match=match, count=count, type=_type)
    plan, changes = reconcile(
        session,
        _current_primaries(client),
        strict=False,
        include_new_nodes=True,
    )
    return ClusterScanIterator(client, session, plan, changes)


def resume_cluster_scan(
    client: "RedisCluster",
    session: ClusterScanSession,
    *,
    match: Any = _UNSET,
    count: Any = _UNSET,
    _type: Any = _UNSET,
    strict: bool = True,
    include_new_nodes: bool = False,
) -> ClusterScanIterator:
    """
    Resume a previously saved cluster scan session.

    The scan conditions are bound to the session: passing ``match``/``count``/
    ``_type`` values that differ from the session's raises
    :exc:`ClusterScanSessionMismatch`.

    Topology changes since the session was saved are reconciled by stable
    node identity: a node that merely changed address keeps its cursor; a
    node that left the cluster with an unfinished range raises
    :exc:`ClusterScanTopologyError` in strict mode (the default) and is
    skipped otherwise; nodes that joined since are only scanned when
    ``include_new_nodes`` is set. Observed changes are reported on
    ``iterator.topology_changes``.
    """
    _check_conditions(session, match, count, _type)
    plan, changes = reconcile(
        session,
        _current_primaries(client),
        strict=strict,
        include_new_nodes=include_new_nodes,
    )
    return ClusterScanIterator(client, session, plan, changes)
