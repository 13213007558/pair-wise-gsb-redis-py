import pytest
from redis.asyncio.cluster_scan import cluster_scan, resume_cluster_scan
from redis.cluster_scan import (
    AddressChange,
    ClusterScanSession,
    ClusterScanSessionMismatch,
    ClusterScanTopologyError,
)

from tests.test_cluster_scan import FakeCluster


class AsyncFakeCluster:
    """Async variant of the fake cluster client used by the sync tests."""

    def __init__(self):
        self._fake = FakeCluster()

    def __getattr__(self, name):
        # add_node/remove_node/change_address/move_keys/servers stay sync
        return getattr(self._fake, name)

    async def cluster_myid(self, target_node):
        return self._fake.cluster_myid(target_node=target_node)

    async def scan(
        self, cursor=0, match=None, count=None, _type=None, target_nodes=None
    ):
        return self._fake.scan(
            cursor=cursor,
            match=match,
            count=count,
            _type=_type,
            target_nodes=target_nodes,
        )


def make_cluster():
    cluster = AsyncFakeCluster()
    cluster.add_node("id-a", [f"a{i}".encode() for i in range(10)])
    cluster.add_node("id-b", [f"b{i}".encode() for i in range(5)])
    cluster.add_node("id-c", [f"c{i}".encode() for i in range(3)])
    return cluster


def all_keys(cluster):
    return sorted(
        key for server in cluster.servers.values() for key in server.keys if key
    )


async def collect(iterator):
    return [key async for key in iterator]


class TestAsyncClusterScanIterator:
    async def test_full_iteration(self):
        cluster = make_cluster()
        iterator = await cluster_scan(cluster, count=4)
        keys = await collect(iterator)
        assert sorted(keys) == all_keys(cluster)
        assert iterator.session.is_complete

    async def test_empty_batch_with_nonzero_cursor_is_not_done(self):
        cluster = AsyncFakeCluster()
        server = cluster.add_node("id-a", [b"x1", b"x2", b"target"])
        iterator = await cluster_scan(cluster, match=b"target*", count=2)
        assert await collect(iterator) == [b"target"]
        assert server.scan_calls == 2
        assert iterator.session.is_complete

    async def test_pause_and_resume_mid_batch(self):
        cluster = make_cluster()
        iterator = await cluster_scan(cluster, count=4)
        first = [await iterator.__anext__(), await iterator.__anext__()]

        session = ClusterScanSession.loads(iterator.session.dumps())
        assert list(session.nodes["id-a"].pending) == [b"a2", b"a3"]

        resumed = await resume_cluster_scan(cluster, session)
        rest = await collect(resumed)
        assert rest[:2] == [b"a2", b"a3"]

        combined = first + rest
        assert sorted(combined) == all_keys(cluster)
        assert len(combined) == len(all_keys(cluster))
        assert resumed.session.is_complete

    async def test_resume_rejects_changed_conditions(self):
        cluster = make_cluster()
        iterator = await cluster_scan(cluster, match=b"a*", count=4)
        await iterator.__anext__()
        session = ClusterScanSession.loads(iterator.session.dumps())

        with pytest.raises(ClusterScanSessionMismatch):
            await resume_cluster_scan(cluster, session, match=b"b*")
        with pytest.raises(ClusterScanSessionMismatch):
            await resume_cluster_scan(cluster, session, count=5)

    async def test_address_change_keeps_cursor(self):
        cluster = make_cluster()
        iterator = await cluster_scan(cluster, count=4)
        await iterator.__anext__()
        session = ClusterScanSession.loads(iterator.session.dumps())

        cluster.change_address("id-a", "10.0.0.1", 9000)
        resumed = await resume_cluster_scan(cluster, session)
        assert resumed.topology_changes.address_changed == (
            AddressChange("id-a", "127.0.0.1:7000", "10.0.0.1:9000"),
        )
        keys = await collect(resumed)
        combined = [b"a0"] + keys
        assert sorted(combined) == all_keys(cluster)
        assert len(combined) == len(all_keys(cluster))

    async def test_missing_node_strict_and_non_strict(self):
        cluster = make_cluster()
        iterator = await cluster_scan(cluster, count=4)
        await iterator.__anext__()
        session = ClusterScanSession.loads(iterator.session.dumps())

        cluster.remove_node("id-a")
        with pytest.raises(ClusterScanTopologyError) as exc_info:
            await resume_cluster_scan(cluster, session)
        assert [state.node_id for state in exc_info.value.missing] == ["id-a"]

        resumed = await resume_cluster_scan(cluster, session, strict=False)
        keys = await collect(resumed)
        assert sorted(keys) == [k for k in all_keys(cluster) if not k.startswith(b"a")]

    async def test_new_node_excluded_unless_opted_in(self):
        cluster = make_cluster()
        iterator = await cluster_scan(cluster, count=4)
        await iterator.__anext__()
        session = ClusterScanSession.loads(iterator.session.dumps())

        cluster.add_node("id-d", [b"d0", b"d1"])

        resumed = await resume_cluster_scan(cluster, session)
        keys = await collect(resumed)
        assert b"d0" not in keys and b"d1" not in keys
        assert resumed.topology_changes.added == (("id-d", "127.0.0.1:7003"),)

        resumed = await resume_cluster_scan(cluster, session, include_new_nodes=True)
        keys = await collect(resumed)
        assert b"d0" in keys and b"d1" in keys

    async def test_slot_migration_mid_scan(self):
        cluster = make_cluster()
        iterator = await cluster_scan(cluster, count=4)
        first = [await iterator.__anext__(), await iterator.__anext__()]
        session = ClusterScanSession.loads(iterator.session.dumps())

        cluster.move_keys("id-a", "id-b", [b"a0", b"a4", b"a5", b"a6", b"a7"])

        resumed = await resume_cluster_scan(cluster, session)
        rest = await collect(resumed)
        combined = first + rest

        assert sorted(set(combined)) == all_keys(cluster)
        # the duplicate a0 comes from Redis SCAN semantics over a changing
        # keyspace, not from the resumer
        assert combined.count(b"a0") == 2
        for key in all_keys(cluster):
            if key != b"a0":
                assert combined.count(key) == 1
