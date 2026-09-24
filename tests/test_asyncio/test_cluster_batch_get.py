"""
Async tests for ``mget_across_slots`` against a scripted three-node cluster.

Mirrors ``tests/test_cluster_batch_get.py`` for the async cluster client;
the result contract is identical on both stacks.
"""

import pytest

from redis.asyncio.cluster import RedisCluster
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.commands.cluster import ClusterBatchGetResult
from redis.crc import key_slot
from redis.exceptions import ConnectionError, ResponseError

from tests.scripted_cluster import (
    ScriptedCluster,
    drop_connection,
    error_reply,
    keys_for_each_node,
    moved_reply,
)


def make_client(cluster: ScriptedCluster, **kwargs) -> RedisCluster:
    r = RedisCluster(startup_nodes=cluster.startup_nodes(), **kwargs)
    # The async cluster client captures the connection class in its startup
    # ClusterNode objects and connection kwargs at construction time, before
    # the scripted transport can be injected, so point both at it here, ahead
    # of the lazy topology initialization.
    r.connection_kwargs["connection_class"] = cluster.async_connection_class
    for node in r.nodes_manager.startup_nodes.values():
        node.connection_class = cluster.async_connection_class
    return r


@pytest.fixture()
def cluster():
    with ScriptedCluster() as scripted_cluster:
        yield scripted_cluster


@pytest.fixture()
async def client(cluster):
    r = make_client(cluster)
    yield r
    await r.aclose()


def value_per_key(values):
    """An MGET script answering ``values[key]`` (None when absent)."""

    def script(keys):
        return [values.get(key) for key in keys]

    return script


class TestMgetAcrossSlots:
    async def test_results_follow_input_order_with_duplicates_and_missing_keys(
        self, cluster, client
    ):
        keys = keys_for_each_node(cluster)
        missing = keys[1] + b"-missing"
        values = {key: b"value-" + key for key in keys}
        for node in cluster.nodes:
            node.mget_script = value_per_key(values)

        requested = [keys[2], keys[0], keys[0], missing, keys[1]]
        results = await client.mget_across_slots(requested)

        assert results == [
            ClusterBatchGetResult(value=values[keys[2]], error=None),
            ClusterBatchGetResult(value=values[keys[0]], error=None),
            ClusterBatchGetResult(value=values[keys[0]], error=None),
            ClusterBatchGetResult(value=None, error=None),
            ClusterBatchGetResult(value=values[keys[1]], error=None),
        ]
        assert cluster.nodes[0].received(b"MGET") == [[b"MGET", keys[0], keys[0]]]
        assert cluster.nodes[1].received(b"MGET") == [[b"MGET", missing, keys[1]]]
        assert cluster.nodes[2].received(b"MGET") == [[b"MGET", keys[2]]]

    async def test_empty_keys(self, cluster, client):
        assert await client.mget_across_slots([]) == []
        for node in cluster.nodes:
            assert node.received(b"MGET") == []

    async def test_decode_responses(self, cluster):
        client = make_client(cluster, decode_responses=True)
        try:
            keys = keys_for_each_node(cluster)
            values = {key: b"value-" + key for key in keys}
            for node in cluster.nodes:
                node.mget_script = value_per_key(values)

            results = await client.mget_across_slots(keys)

            assert results == [
                ClusterBatchGetResult(value="value-" + key.decode(), error=None)
                for key in keys
            ]
        finally:
            await client.aclose()

    async def test_partial_failure_is_scoped_to_the_failing_slot_group(
        self, cluster, client
    ):
        keys = keys_for_each_node(cluster)
        values = {key: b"value-" + key for key in (keys[0], keys[2])}
        cluster.nodes[0].mget_script = value_per_key(values)
        cluster.nodes[1].mget_script = lambda keys: error_reply(
            b"ERR simulated failure"
        )
        cluster.nodes[2].mget_script = value_per_key(values)

        results = await client.mget_across_slots(keys)

        assert results[0] == ClusterBatchGetResult(value=values[keys[0]], error=None)
        assert results[1].value is None
        assert isinstance(results[1].error, ResponseError)
        assert results[2] == ClusterBatchGetResult(value=values[keys[2]], error=None)
        for node in cluster.nodes:
            assert len(node.received(b"MGET")) == 1

    async def test_moved_redirects_only_the_affected_group(self, cluster, client):
        keys = keys_for_each_node(cluster)
        moved_slot = key_slot(keys[1])
        target = cluster.nodes[2]
        cluster.nodes[0].mget_script = value_per_key({keys[0]: b"value-0"})
        cluster.nodes[1].mget_script = lambda keys: moved_reply(
            moved_slot, target.host.encode(), target.port
        )
        target.mget_script = value_per_key({keys[1]: b"value-1"})

        results = await client.mget_across_slots([keys[0], keys[1]])

        assert results == [
            ClusterBatchGetResult(value=b"value-0", error=None),
            ClusterBatchGetResult(value=b"value-1", error=None),
        ]
        assert len(cluster.nodes[0].received(b"MGET")) == 1
        assert len(cluster.nodes[1].received(b"MGET")) == 1
        assert target.received(b"MGET") == [[b"MGET", keys[1]]]

    async def test_connection_drop_is_retried_per_retry_policy(self, cluster):
        client = make_client(cluster, retry=Retry(NoBackoff(), 3))
        try:
            keys = keys_for_each_node(cluster)
            values = {key: b"value-" + key for key in keys}
            for node in cluster.nodes:
                node.mget_script = value_per_key(values)
            drops = 0

            def drop_once(keys):
                nonlocal drops
                if drops == 0:
                    drops += 1
                    return drop_connection(keys)
                return value_per_key(values)(keys)

            cluster.nodes[1].mget_script = drop_once

            results = await client.mget_across_slots(keys)

            assert results == [
                ClusterBatchGetResult(value=values[key], error=None) for key in keys
            ]
            assert len(cluster.nodes[1].received(b"MGET")) == 2
            assert len(cluster.nodes[0].received(b"MGET")) == 1
            assert len(cluster.nodes[2].received(b"MGET")) == 1
        finally:
            await client.aclose()

    async def test_unknown_result_after_drop_is_reported_per_position(self, cluster):
        client = make_client(cluster, retry=Retry(NoBackoff(), 2))
        try:
            keys = keys_for_each_node(cluster)
            values = {key: b"value-" + key for key in (keys[0], keys[2])}
            cluster.nodes[0].mget_script = value_per_key(values)
            cluster.nodes[1].mget_script = drop_connection
            cluster.nodes[2].mget_script = value_per_key(values)

            results = await client.mget_across_slots(keys)

            assert results[0] == ClusterBatchGetResult(
                value=values[keys[0]], error=None
            )
            assert results[1].value is None
            assert isinstance(results[1].error, ConnectionError)
            assert results[2] == ClusterBatchGetResult(
                value=values[keys[2]], error=None
            )
            assert len(cluster.nodes[1].received(b"MGET")) == 3
            assert len(cluster.nodes[0].received(b"MGET")) == 1
            assert len(cluster.nodes[2].received(b"MGET")) == 1
        finally:
            await client.aclose()
