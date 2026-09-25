from unittest import mock

import pytest

import redis.asyncio as redis
from redis.exceptions import DataError

from tests.test_geo_polygon import ANTIMERIDIAN_RING, HOLE, SQUARE, _row


class TestAsyncGeoSearchPolygon:
    # Mirrors TestGeoSearchPolygon in tests/test_geo_polygon.py: the async
    # helper must filter, expand and reshape exactly like the sync one.
    A = _row(b"a", 1.0, 5, 5)
    B = _row(b"b", 2.0, 1, 1)
    C = _row(b"c", 3.0, 9, 1)
    D = _row(b"d", 4.0, 8, 8)

    def _client(self):
        return redis.Redis()

    async def test_expands_count_until_enough_filtered_results(self):
        client = self._client()
        with mock.patch.object(
            client, "execute_command", new=mock.AsyncMock()
        ) as execute:
            execute.side_effect = [
                [self.A, self.B, self.C],
                [self.A, self.B, self.C, self.D],
            ]
            result = await client.geosearch_polygon(
                "key",
                SQUARE,
                holes=[HOLE],
                count=3,
                sort="ASC",
                withcoord=True,
                withdist=True,
            )
        assert result == [self.B, self.C, self.D]
        assert execute.call_count == 2
        first, second = execute.call_args_list
        assert first.args[:2] == ("GEOSEARCH", "key")
        assert list(first.args[first.args.index(b"COUNT") + 1 :])[0] == 3
        assert list(second.args[second.args.index(b"COUNT") + 1 :])[0] == 6

    async def test_stops_when_bounding_box_is_exhausted(self):
        client = self._client()
        with mock.patch.object(
            client, "execute_command", new=mock.AsyncMock()
        ) as execute:
            execute.side_effect = [[self.A, self.B, self.C]]
            result = await client.geosearch_polygon(
                "key",
                SQUARE,
                holes=[HOLE],
                count=5,
                sort="ASC",
                withcoord=True,
                withdist=True,
            )
        assert result == [self.B, self.C]
        assert execute.call_count == 1

    async def test_plain_response_shape(self):
        client = self._client()
        with mock.patch.object(
            client, "execute_command", new=mock.AsyncMock()
        ) as execute:
            execute.side_effect = [[[b"a", (5, 5)], [b"b", (1, 1)], [b"c", (9, 1)]]]
            result = await client.geosearch_polygon("key", SQUARE, holes=[HOLE])
        assert result == [b"b", b"c"]
        assert b"WITHCOORD" in execute.call_args_list[0].args

    async def test_antimeridian_polygon(self):
        client = self._client()
        near = _row(b"near", 1.0, 179.5, 45)
        across = _row(b"across", 2.0, -179.5, 45)
        far = _row(b"far", 3.0, 0, 45)
        with mock.patch.object(
            client, "execute_command", new=mock.AsyncMock()
        ) as execute:
            execute.side_effect = [[near, across, far]]
            result = await client.geosearch_polygon(
                "key", ANTIMERIDIAN_RING, sort="ASC", withcoord=True, withdist=True
            )
        assert result == [near, across]
        args = execute.call_args_list[0].args
        from_lonlat = args[args.index(b"FROMLONLAT") + 1 :]
        assert list(from_lonlat[:2]) == [-180.0, 45.0]

    async def test_invalid_input_rejected_before_any_command(self):
        client = self._client()
        with mock.patch.object(
            client, "execute_command", new=mock.AsyncMock()
        ) as execute:
            with pytest.raises(DataError):
                await client.geosearch_polygon("key", [(0, 0), (2, 2), (2, 0), (0, 2)])
            with pytest.raises(DataError):
                await client.geosearch_polygon("key", [(0, 0), (1, 0), (1, 90)])
        execute.assert_not_called()
