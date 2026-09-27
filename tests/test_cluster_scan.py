import fnmatch

import pytest
from redis.cluster_scan import (
    AddressChange,
    ClusterScanSession,
    ClusterScanSessionInvalid,
    ClusterScanSessionMismatch,
    ClusterScanTopologyError,
    NodeScanState,
    cluster_scan,
    resume_cluster_scan,
)


class FakeClusterNode:
    """Stand-in for redis.cluster.ClusterNode."""

    def __init__(self, host, port):
        self.host = host
        self.port = port

    @property
    def name(self):
        return f"{self.host}:{self.port}"


class FakeServer:
    """
    One fake cluster node: a keyspace with SCAN cursor semantics.

    The cursor is an index into the keyspace table. Removed keys leave
    tombstones so that positions of unscanned keys never shift, mirroring
    how Redis SCAN stays valid while the hash table mutates. COUNT limits
    the number of raw table entries examined per call, so a batch may come
    back empty while the cursor is still non-zero.
    """

    def __init__(self, node_id, host, port, keys, key_types=None):
        self.node_id = node_id
        self.node = FakeClusterNode(host, port)
        self.keys = list(keys)
        self.key_types = key_types or {}
        self.scan_calls = 0

    def scan(self, cursor, match, count, _type):
        self.scan_calls += 1
        count = count if count is not None else 10
        window = self.keys[cursor : cursor + count]
        next_cursor = cursor + len(window)
        if next_cursor >= len(self.keys):
            next_cursor = 0
        batch = [
            key
            for key in window
            if key is not None and self._matches(key, match, _type)
        ]
        return next_cursor, batch

    def _matches(self, key, match, _type):
        if match is not None:
            pattern = match.decode() if isinstance(match, bytes) else match
            if not fnmatch.fnmatchcase(key.decode(), pattern):
                return False
        if _type is not None and self.key_types.get(key, "string") != _type:
            return False
        return True


class FakeCluster:
    """
    A fake cluster client exposing the interface the scan iterators use:
    get_primaries(), cluster_myid(target_node=...) and scan(...).

    Node ids are stable; addresses and key ownership can change mid-scan.
    """

    def __init__(self):
        self.servers = {}

    def add_node(self, node_id, keys, host="127.0.0.1", port=None, key_types=None):
        port = port if port is not None else 7000 + len(self.servers)
        server = FakeServer(node_id, host, port, keys, key_types)
        self.servers[node_id] = server
        return server

    def remove_node(self, node_id):
        del self.servers[node_id]

    def change_address(self, node_id, host, port):
        server = self.servers[node_id]
        server.node = FakeClusterNode(host, port)

    def move_keys(self, src_id, dst_id, keys):
        """Slot migration: keys leave the source (tombstoned) and are
        appended to the target's scan table."""
        src = self.servers[src_id]
        dst = self.servers[dst_id]
        for key in keys:
            src.keys[src.keys.index(key)] = None
            dst.keys.append(key)
            if key in src.key_types:
                dst.key_types[key] = src.key_types.pop(key)

    # client interface used by the iterators
    def get_primaries(self):
        return [server.node for server in self.servers.values()]

    def cluster_myid(self, target_node):
        return self._server_for(target_node).node_id

    def scan(self, cursor=0, match=None, count=None, _type=None, target_nodes=None):
        server = self._server_for(target_nodes)
        next_cursor, batch = server.scan(cursor, match, count, _type)
        return {target_nodes.name: next_cursor}, batch

    def _server_for(self, node):
        for server in self.servers.values():
            if server.node.name == node.name:
                return server
        raise AssertionError(f"unknown node {node.name}")


def make_cluster():
    cluster = FakeCluster()
    cluster.add_node("id-a", [f"a{i}".encode() for i in range(10)])
    cluster.add_node("id-b", [f"b{i}".encode() for i in range(5)])
    cluster.add_node("id-c", [f"c{i}".encode() for i in range(3)])
    return cluster


def all_keys(cluster):
    return sorted(
        key for server in cluster.servers.values() for key in server.keys if key
    )


class TestClusterScanSession:
    def test_serialization_round_trip(self):
        session = ClusterScanSession(match=b"a*", count=4, type="string")
        state = NodeScanState(node_id="id-a", address="127.0.0.1:7000", cursor=4)
        state.pending.extend([b"a2", b"a3"])
        session.nodes["id-a"] = state
        session.nodes["id-b"] = NodeScanState(
            node_id="id-b", address="127.0.0.1:7001", cursor=0, done=True
        )

        assert ClusterScanSession.loads(session.dumps()) == session
        assert ClusterScanSession.from_dict(session.to_dict()) == session
        # bytes and str payloads both load
        assert ClusterScanSession.loads(session.dumps().encode()) == session

        restored = ClusterScanSession.loads(session.dumps())
        assert restored.match == b"a*"
        assert list(restored.nodes["id-a"].pending) == [b"a2", b"a3"]

    def test_loads_rejects_bad_payloads(self):
        with pytest.raises(ClusterScanSessionInvalid):
            ClusterScanSession.loads("not json")
        with pytest.raises(ClusterScanSessionInvalid):
            ClusterScanSession.loads('{"format": "other", "version": 1}')
        with pytest.raises(ClusterScanSessionInvalid):
            ClusterScanSession.loads(
                '{"format": "redis-py-cluster-scan", "version": 999}'
            )
        with pytest.raises(ClusterScanSessionInvalid):
            ClusterScanSession.loads(
                '{"format": "redis-py-cluster-scan", "version": 1}'
            )

    def test_completion_state(self):
        session = ClusterScanSession()
        state = NodeScanState(node_id="id-a", address="h:1", cursor=3)
        session.nodes["id-a"] = state
        assert not session.is_complete
        assert session.unfinished_nodes == [state]

        state.cursor = 0
        state.done = True
        state.pending.append(b"k")
        # buffered-but-unyielded items mean the session is not complete yet
        assert not session.is_complete
        assert session.unfinished_nodes == []
        state.pending.popleft()
        assert session.is_complete


class TestClusterScanIterator:
    def test_full_iteration(self):
        cluster = make_cluster()
        iterator = cluster_scan(cluster, count=4)
        keys = list(iterator)
        assert sorted(keys) == all_keys(cluster)
        assert iterator.session.is_complete

    def test_match_and_type_filters(self):
        cluster = FakeCluster()
        cluster.add_node(
            "id-a",
            [b"user:1", b"user:2", b"other"],
            key_types={b"user:1": "string", b"user:2": "hash"},
        )
        keys = list(cluster_scan(cluster, match=b"user:*"))
        assert sorted(keys) == [b"user:1", b"user:2"]
        keys = list(cluster_scan(cluster, match=b"user:*", _type="hash"))
        assert keys == [b"user:2"]

    def test_empty_batch_with_nonzero_cursor_is_not_done(self):
        cluster = FakeCluster()
        server = cluster.add_node("id-a", [b"x1", b"x2", b"target"])
        iterator = cluster_scan(cluster, match=b"target*", count=2)
        # first window (x1, x2) is filtered out entirely: empty batch,
        # non-zero cursor; the node must not be considered done
        assert list(iterator) == [b"target"]
        assert server.scan_calls == 2
        assert iterator.session.is_complete

    def test_pause_and_resume_mid_batch(self):
        cluster = make_cluster()
        iterator = cluster_scan(cluster, count=4)
        first = [next(iterator), next(iterator)]

        payload = iterator.session.dumps()
        session = ClusterScanSession.loads(payload)
        # node id-a fetched a batch of 4; 2 were yielded, 2 remain buffered
        assert list(session.nodes["id-a"].pending) == [b"a2", b"a3"]
        assert session.nodes["id-a"].cursor == 4

        resumed = resume_cluster_scan(cluster, session)
        rest = list(resumed)
        # the buffered remainder of the in-flight batch comes first
        assert rest[:2] == [b"a2", b"a3"]

        combined = first + rest
        # the fake SCAN is duplicate-free, so any duplicate or loss here
        # would be caused by the resumer itself
        assert sorted(combined) == all_keys(cluster)
        assert len(combined) == len(all_keys(cluster))
        assert resumed.session.is_complete

    def test_resume_rejects_changed_conditions(self):
        cluster = make_cluster()
        iterator = cluster_scan(cluster, match=b"a*", count=4, _type="string")
        next(iterator)
        session = ClusterScanSession.loads(iterator.session.dumps())

        with pytest.raises(ClusterScanSessionMismatch):
            resume_cluster_scan(cluster, session, match=b"b*")
        with pytest.raises(ClusterScanSessionMismatch):
            resume_cluster_scan(cluster, session, count=5)
        with pytest.raises(ClusterScanSessionMismatch):
            resume_cluster_scan(cluster, session, _type="hash")

        # re-stating the bound conditions explicitly is accepted
        resumed = resume_cluster_scan(
            cluster, session, match=b"a*", count=4, _type="string"
        )
        assert sorted(resumed) == [f"a{i}".encode() for i in range(1, 10)]

    def test_address_change_keeps_cursor(self):
        cluster = make_cluster()
        iterator = cluster_scan(cluster, count=4)
        next(iterator)
        session = ClusterScanSession.loads(iterator.session.dumps())

        cluster.change_address("id-a", "10.0.0.1", 9000)
        resumed = resume_cluster_scan(cluster, session)

        assert resumed.topology_changes.address_changed == (
            AddressChange("id-a", "127.0.0.1:7000", "10.0.0.1:9000"),
        )
        assert session.nodes["id-a"].address == "10.0.0.1:9000"
        assert resumed.topology_changes.missing == ()
        assert resumed.topology_changes.added == ()

        keys = list(resumed)
        combined = [b"a0"] + keys
        assert sorted(combined) == all_keys(cluster)
        assert len(combined) == len(all_keys(cluster))

    def test_missing_node_strict_reports_unfinished_range(self):
        cluster = make_cluster()
        iterator = cluster_scan(cluster, count=4)
        next(iterator)
        session = ClusterScanSession.loads(iterator.session.dumps())

        cluster.remove_node("id-a")
        with pytest.raises(ClusterScanTopologyError) as exc_info:
            resume_cluster_scan(cluster, session)
        missing = exc_info.value.missing
        assert [state.node_id for state in missing] == ["id-a"]
        assert missing[0].cursor == 4

    def test_missing_node_non_strict_skips_and_reports(self):
        cluster = make_cluster()
        iterator = cluster_scan(cluster, count=4)
        next(iterator)
        session = ClusterScanSession.loads(iterator.session.dumps())

        cluster.remove_node("id-a")
        resumed = resume_cluster_scan(cluster, session, strict=False)
        keys = list(resumed)
        assert sorted(keys) == [k for k in all_keys(cluster) if not k.startswith(b"a")]
        assert [s.node_id for s in resumed.topology_changes.missing] == ["id-a"]

    def test_finished_node_leaving_is_not_an_error(self):
        cluster = make_cluster()
        iterator = cluster_scan(cluster, count=4)
        # drain node id-a fully (10 keys, batches of 4)
        for _ in range(10):
            next(iterator)
        session = ClusterScanSession.loads(iterator.session.dumps())
        assert session.nodes["id-a"].done

        cluster.remove_node("id-a")
        resumed = resume_cluster_scan(cluster, session)
        assert sorted(resumed) == [
            k for k in all_keys(cluster) if not k.startswith(b"a")
        ]

    def test_new_node_excluded_unless_opted_in(self):
        cluster = make_cluster()
        iterator = cluster_scan(cluster, count=4)
        next(iterator)
        session = ClusterScanSession.loads(iterator.session.dumps())

        cluster.add_node("id-d", [b"d0", b"d1"])

        resumed = resume_cluster_scan(cluster, session)
        keys = list(resumed)
        assert b"d0" not in keys and b"d1" not in keys
        assert resumed.topology_changes.added == (("id-d", "127.0.0.1:7003"),)
        assert "id-d" not in session.nodes

        resumed = resume_cluster_scan(cluster, session, include_new_nodes=True)
        keys = list(resumed)
        assert b"d0" in keys and b"d1" in keys
        assert "id-d" in session.nodes

    def test_slot_migration_mid_scan(self):
        cluster = make_cluster()
        iterator = cluster_scan(cluster, count=4)
        first = [next(iterator), next(iterator)]
        session = ClusterScanSession.loads(iterator.session.dumps())

        # migrate a0 (already yielded) and a4..a7 (not yet scanned) to id-b
        cluster.move_keys("id-a", "id-b", [b"a0", b"a4", b"a5", b"a6", b"a7"])

        resumed = resume_cluster_scan(cluster, session)
        rest = list(resumed)
        combined = first + rest

        # every key is returned at least once despite the migration
        assert sorted(set(combined)) == all_keys(cluster)
        # a0 is returned twice: once by id-a before the move, once by id-b
        # after. That duplicate is inherent to Redis SCAN over a changing
        # keyspace, not introduced by pausing/resuming.
        assert combined.count(b"a0") == 2
        for key in all_keys(cluster):
            if key != b"a0":
                assert combined.count(key) == 1
