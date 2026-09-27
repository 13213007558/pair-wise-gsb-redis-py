"""
Async mirror of :mod:`redis.cluster_scan`.

The session model (:class:`ClusterScanSession`, :class:`NodeScanState`,
exceptions, topology records) is shared with the sync stack and imported
from :mod:`redis.cluster_scan`; only the iterator and the entry points are
reimplemented here around ``await``. See the sync module's docstring for
the guarantees and non-guarantees of resumable cluster scans.
"""

from typing import TYPE_CHECKING, Any, AsyncIterator

from redis.cluster_scan import (
    _UNSET,
    ClusterScanSession,
    NodeScanState,
    TopologyChanges,
    _check_conditions,
    reconcile,
)
from redis.typing import PatternT
from redis.utils import safe_str

if TYPE_CHECKING:
    from redis.asyncio.cluster import RedisCluster
    from redis.cluster import ClusterNode

__all__ = [
    "AsyncClusterScanIterator",
    "cluster_scan",
    "resume_cluster_scan",
]


async def _current_primaries(client: "RedisCluster") -> dict[str, "ClusterNode"]:
    current = {}
    for node in client.get_primaries():
        node_id = safe_str(await client.cluster_myid(target_node=node))
        current[node_id] = node
    return current


class AsyncClusterScanIterator:
    """
    Async variant of :class:`redis.cluster_scan.ClusterScanIterator`.

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

    def __aiter__(self) -> AsyncIterator:
        return self

    async def __anext__(self) -> bytes | str:
        while self._index < len(self._plan):
            state, node = self._plan[self._index]
            if state.pending:
                return state.pending.popleft()
            if state.done:
                self._index += 1
                continue
            await self._fetch(state, node)
        raise StopAsyncIteration

    async def _fetch(self, state: NodeScanState, node: "ClusterNode") -> None:
        cursors, data = await self._client.scan(
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


async def cluster_scan(
    client: "RedisCluster",
    match: PatternT | None = None,
    count: int | None = None,
    _type: str | None = None,
) -> AsyncClusterScanIterator:
    """
    Start a new resumable cluster-wide scan over the current primaries.

    ``match``/``count``/``_type`` behave as in ``RedisCluster.scan`` and are
    bound to the returned session. Iterate the result to yield keys; serialize
    ``iterator.session`` at any point to make the progress resumable.
    """
    session = ClusterScanSession(match=match, count=count, type=_type)
    plan, changes = reconcile(
        session,
        await _current_primaries(client),
        strict=False,
        include_new_nodes=True,
    )
    return AsyncClusterScanIterator(client, session, plan, changes)


async def resume_cluster_scan(
    client: "RedisCluster",
    session: ClusterScanSession,
    *,
    match: Any = _UNSET,
    count: Any = _UNSET,
    _type: Any = _UNSET,
    strict: bool = True,
    include_new_nodes: bool = False,
) -> AsyncClusterScanIterator:
    """
    Resume a previously saved cluster scan session.

    The scan conditions are bound to the session: passing ``match``/``count``/
    ``_type`` values that differ from the session's raises
    :exc:`redis.cluster_scan.ClusterScanSessionMismatch`.

    Topology changes since the session was saved are reconciled by stable
    node identity: a node that merely changed address keeps its cursor; a
    node that left the cluster with an unfinished range raises
    :exc:`redis.cluster_scan.ClusterScanTopologyError` in strict mode (the
    default) and is skipped otherwise; nodes that joined since are only
    scanned when ``include_new_nodes`` is set. Observed changes are reported
    on ``iterator.topology_changes``.
    """
    _check_conditions(session, match, count, _type)
    plan, changes = reconcile(
        session,
        await _current_primaries(client),
        strict=strict,
        include_new_nodes=include_new_nodes,
    )
    return AsyncClusterScanIterator(client, session, plan, changes)
