"""
Tests for ``mget_across_slots`` against a scripted three-node cluster.

The scripted peers (see ``tests/scripted_cluster.py``) record every request
they receive, so these tests assert on input ordering, duplicated keys,
partial failures and the exact number of network calls after redirections.
"""

import pytest

from redis.backoff import NoBackoff
from redis.cluster import RedisCluster
from redis.commands.cluster import ClusterBatchGetResult
from redis.crc import key_slot
from redis.exceptions import ConnectionError, RedisClusterException, ResponseError
from redis.retry import Retry

from tests.scripted_cluster import (
    ScriptedCluster,
    ask_reply,
    drop_connection,
    error_reply,
    keys_for_each_node,
    moved_reply,
)


def make_client(cluster: ScriptedCluster, **kwargs) -> RedisCluster:
    # The scripted connection class only reaches every node pool through the
    # from_url code path, so the client is built from the (fake) startup URL.
    return RedisCluster.from_url(
        f"redis://{cluster.nodes[0].name}/0",
        connection_class=cluster.connection_class,
        **kwargs,
    )


@pytest.fixture()
def cluster():
    with ScriptedCluster() as scripted_cluster:
        yield scripted_cluster


@pytest.fixture()
def client(cluster):
    r = make_client(cluster)
    yield r
    r.close()


def value_per_key(values):
    """An MGET script answering ``values[key]`` (None when absent)."""

    def script(keys):
        return [values.get(key) for key in keys]

    return script


class TestMgetAcrossSlots:
    def test_results_follow_input_order_with_duplicates_and_missing_keys(
        self, cluster, client
    ):
        keys = keys_for_each_node(cluster)
        missing = keys[1] + b"-missing"
        values = {key: b"value-" + key for key in keys}
        for node in cluster.nodes:
            node.mget_script = value_per_key(values)

        requested = [keys[2], keys[0], keys[0], missing, keys[1]]
        results = client.mget_across_slots(requested)

        assert results == [
            ClusterBatchGetResult(value=values[keys[2]], error=None),
            ClusterBatchGetResult(value=values[keys[0]], error=None),
            ClusterBatchGetResult(value=values[keys[0]], error=None),
            ClusterBatchGetResult(value=None, error=None),
            ClusterBatchGetResult(value=values[keys[1]], error=None),
        ]
        # Every slot group was read exactly once, duplicates included, with
        # the group keys sent in input order.
        assert cluster.nodes[0].received(b"MGET") == [[b"MGET", keys[0], keys[0]]]
        assert cluster.nodes[1].received(b"MGET") == [[b"MGET", missing, keys[1]]]
        assert cluster.nodes[2].received(b"MGET") == [[b"MGET", keys[2]]]

    def test_empty_keys(self, cluster, client):
        assert client.mget_across_slots([]) == []
        for node in cluster.nodes:
            assert node.received(b"MGET") == []

    def test_single_key_and_args_form(self, cluster, client):
        keys = keys_for_each_node(cluster)
        values = {key: b"value-" + key for key in keys}
        for node in cluster.nodes:
            node.mget_script = value_per_key(values)

        assert client.mget_across_slots(keys[0]) == [
            ClusterBatchGetResult(value=values[keys[0]], error=None)
        ]
        assert client.mget_across_slots(keys[0], keys[1]) == [
            ClusterBatchGetResult(value=values[keys[0]], error=None),
            ClusterBatchGetResult(value=values[keys[1]], error=None),
        ]

    def test_binary_keys_and_hash_tags(self, cluster, client):
        # Binary keys whose slots are pinned by their hash tags.
        keys = [b"\x00{key-%d}\xff" % i for i in (0, 3, 6)]
        assert [cluster.node_for_key(key) for key in keys] == list(cluster.nodes)
        values = {key: b"\x00\x01" + key for key in keys}
        for node in cluster.nodes:
            node.mget_script = value_per_key(values)

        results = client.mget_across_slots(keys)

        assert results == [
            ClusterBatchGetResult(value=values[key], error=None) for key in keys
        ]
        # The keys went on the wire exactly as passed, binary content included.
        for node in cluster.nodes:
            node_keys = [arg for mget in node.received(b"MGET") for arg in mget[1:]]
            assert node_keys == [
                key for key in keys if cluster.node_for_key(key) is node
            ]

    def test_decode_responses(self, cluster):
        client = make_client(cluster, decode_responses=True)
        try:
            keys = keys_for_each_node(cluster)
            values = {key: b"value-" + key for key in keys}
            for node in cluster.nodes:
                node.mget_script = value_per_key(values)

            results = client.mget_across_slots(keys)

            assert results == [
                ClusterBatchGetResult(value="value-" + key.decode(), error=None)
                for key in keys
            ]
        finally:
            client.close()

    def test_partial_failure_is_scoped_to_the_failing_slot_group(self, cluster, client):
        keys = keys_for_each_node(cluster)
        values = {key: b"value-" + key for key in (keys[0], keys[2])}
        cluster.nodes[0].mget_script = value_per_key(values)
        cluster.nodes[1].mget_script = lambda keys: error_reply(
            b"ERR simulated failure"
        )
        cluster.nodes[2].mget_script = value_per_key(values)

        results = client.mget_across_slots(keys)

        assert results[0] == ClusterBatchGetResult(value=values[keys[0]], error=None)
        assert results[1].value is None
        assert isinstance(results[1].error, ResponseError)
        assert "simulated failure" in str(results[1].error)
        assert results[2] == ClusterBatchGetResult(value=values[keys[2]], error=None)
        # The failing group did not cause the successful ones to be re-read.
        for node in cluster.nodes:
            assert len(node.received(b"MGET")) == 1

    def test_moved_redirects_only_the_affected_group(self, cluster, client):
        keys = keys_for_each_node(cluster)
        moved_slot = key_slot(keys[1])
        target = cluster.nodes[2]
        cluster.nodes[0].mget_script = value_per_key({keys[0]: b"value-0"})
        cluster.nodes[1].mget_script = lambda keys: moved_reply(
            moved_slot, target.host.encode(), target.port
        )
        target.mget_script = value_per_key({keys[1]: b"value-1"})

        results = client.mget_across_slots([keys[0], keys[1]])

        assert results == [
            ClusterBatchGetResult(value=b"value-0", error=None),
            ClusterBatchGetResult(value=b"value-1", error=None),
        ]
        # One MGET per group, plus exactly one re-issue of the redirected
        # group on the MOVED target; the successful group was not re-read.
        assert len(cluster.nodes[0].received(b"MGET")) == 1
        assert len(cluster.nodes[1].received(b"MGET")) == 1
        assert target.received(b"MGET") == [[b"MGET", keys[1]]]

    def test_ask_redirects_only_the_affected_group(self, cluster, client):
        keys = keys_for_each_node(cluster)
        ask_slot = key_slot(keys[1])
        target = cluster.nodes[2]
        cluster.nodes[0].mget_script = value_per_key({keys[0]: b"value-0"})
        cluster.nodes[1].mget_script = lambda keys: ask_reply(
            ask_slot, target.host.encode(), target.port
        )
        target.mget_script = value_per_key({keys[1]: b"value-1"})

        results = client.mget_across_slots([keys[0], keys[1]])

        assert results == [
            ClusterBatchGetResult(value=b"value-0", error=None),
            ClusterBatchGetResult(value=b"value-1", error=None),
        ]
        assert len(cluster.nodes[0].received(b"MGET")) == 1
        assert len(cluster.nodes[1].received(b"MGET")) == 1
        # The redirected group was re-issued once, guarded by ASKING.
        assert target.received(b"ASKING") == [[b"ASKING"]]
        assert target.received(b"MGET") == [[b"MGET", keys[1]]]

    def test_connection_drop_is_retried_per_retry_policy(self, cluster):
        client = make_client(cluster, retry=Retry(NoBackoff(), 3))
        try:
            keys = keys_for_each_node(cluster)
            values = {key: b"value-" + key for key in keys}
            for node in cluster.nodes:
                node.mget_script = value_per_key(values)
            # The first read of the middle group drops the connection,
            # leaving the result unknown; the retry policy re-reads it.
            drops = 0

            def drop_once(keys):
                nonlocal drops
                if drops == 0:
                    drops += 1
                    return drop_connection(keys)
                return value_per_key(values)(keys)

            cluster.nodes[1].mget_script = drop_once

            results = client.mget_across_slots(keys)

            assert results == [
                ClusterBatchGetResult(value=values[key], error=None) for key in keys
            ]
            assert len(cluster.nodes[1].received(b"MGET")) == 2
            assert len(cluster.nodes[0].received(b"MGET")) == 1
            assert len(cluster.nodes[2].received(b"MGET")) == 1
        finally:
            client.close()

    def test_unknown_result_after_drop_is_reported_per_position(self, cluster):
        client = make_client(cluster, retry=Retry(NoBackoff(), 2))
        try:
            keys = keys_for_each_node(cluster)
            values = {key: b"value-" + key for key in (keys[0], keys[2])}
            cluster.nodes[0].mget_script = value_per_key(values)
            cluster.nodes[1].mget_script = drop_connection
            cluster.nodes[2].mget_script = value_per_key(values)

            results = client.mget_across_slots(keys)

            assert results[0] == ClusterBatchGetResult(
                value=values[keys[0]], error=None
            )
            assert results[1].value is None
            assert isinstance(results[1].error, ConnectionError)
            assert results[2] == ClusterBatchGetResult(
                value=values[keys[2]], error=None
            )
            # The dropped group was attempted once plus the configured
            # retries; the successful groups were read exactly once.
            assert len(cluster.nodes[1].received(b"MGET")) == 3
            assert len(cluster.nodes[0].received(b"MGET")) == 1
            assert len(cluster.nodes[2].received(b"MGET")) == 1
        finally:
            client.close()


class TestRegularMgetSlotConstraint:
    def test_mget_still_requires_a_single_slot(self, cluster, client):
        with pytest.raises(RedisClusterException):
            client.mget(["{a}1", "{b}1"])

        # Same-slot keys keep working through the regular MGET path.
        assert client.mget(["{a}1", "{a}2"]) == [None, None]

        # The cross-slot attempt above never reached the scripted peers.
        owner = cluster.node_for_key(b"{a}1")
        for node in cluster.nodes:
            expected = [[b"MGET", b"{a}1", b"{a}2"]] if node is owner else []
            assert node.received(b"MGET") == expected
