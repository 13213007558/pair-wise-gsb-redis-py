"""
Async parity tests for the client-side polygon GEOSEARCH helper.

Mirrors tests/test_geosearch_polygon.py using a scripted async fake in
place of the server, proving the async client returns identical results
and issues identical requests.
"""

import pytest

import redis.asyncio as redis
from redis.exceptions import DataError
from redis.geo_polygon import GeoPolygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
HOLE = [[(4, 4), (6, 4), (6, 6), (4, 6)]]
# ring crossing the antimeridian: longitudes 170 .. -170 through 180
DATELINE_RING = [(170, 10), (-170, 10), (-170, 20), (170, 20)]


class AsyncFakeGeoIndex:
    """
    Scripted async stand-in for the server side of GEOSEARCH.

    Candidates are (member, longitude, latitude, distance) tuples in
    ascending distance order; every executed command is recorded in
    the requests list for inspection.
    """

    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.requests = []

    async def __call__(self, *args, **options):
        self.requests.append(args)
        # the helper must always ask for coordinates to filter client side
        assert b"WITHCOORD" in args
        count = args[args.index(b"COUNT") + 1] if b"COUNT" in args else None
        selected = self.candidates[:count] if count is not None else self.candidates
        withdist = b"WITHDIST" in args
        withhash = b"WITHHASH" in args
        response = []
        for i, (member, lon, lat, dist) in enumerate(selected):
            item = [member]
            if withdist:
                item.append(dist)
            if withhash:
                item.append(1000 + i)
            item.append((lon, lat))
            response.append(item)
        return response


def make_client(candidates):
    client = redis.Redis()
    index = AsyncFakeGeoIndex(candidates)
    client.execute_command = index
    return client, index


def requested_counts(index):
    return [
        request[request.index(b"COUNT") + 1]
        for request in index.requests
        if b"COUNT" in request
    ]


async def test_antimeridian_polygon_with_hole():
    client, index = make_client(
        [
            (b"east", 175.0, 15.0, 10.0),
            (b"hole", 178.0, 16.0, 20.0),
            (b"west", -175.0, 15.0, 30.0),
            (b"boundary", -180.0, 10.0, 40.0),
            (b"far", 0.0, 15.0, 50.0),
        ]
    )
    try:
        hole = [[(176, 14), (180, 14), (180, 18), (176, 18)]]
        result = await client.geosearch_polygon("geo", DATELINE_RING, holes=hole)
        assert result == [b"east", b"west", b"boundary"]

        request = index.requests[0]
        i = request.index(b"FROMLONLAT")
        assert abs(request[i + 1]) == pytest.approx(180.0)
        assert request[i + 2] == pytest.approx(15.0)
    finally:
        await client.aclose()


async def test_nearest_count_expands_until_enough_polygon_matches():
    client, index = make_client(
        [
            (b"m1", 5.0, 5.0, 1.0),  # nearest, but inside the hole
            (b"m2", 50.0, 5.0, 2.0),  # second nearest, outside the polygon
            (b"m3", 2.0, 2.0, 3.0),
            (b"m4", 3.0, 3.0, 4.0),
            (b"m5", 1.0, 1.0, 5.0),
        ]
    )
    try:
        result = await client.geosearch_polygon(
            "geo", SQUARE, holes=HOLE, sort="ASC", count=2, withdist=True
        )
        # the nearest 2 inside the polygon, distances as returned by Redis
        assert result == [[b"m3", 3.0], [b"m4", 4.0]]
        assert requested_counts(index) == [2, 4]
        assert all(b"ASC" in request for request in index.requests)
    finally:
        await client.aclose()


async def test_count_stops_when_candidates_exhausted():
    client, index = make_client(
        [
            (b"m1", 5.0, 5.0, 1.0),  # inside the hole
            (b"m2", 2.0, 2.0, 2.0),
            (b"m3", 50.0, 5.0, 3.0),  # outside
            (b"m4", 3.0, 3.0, 4.0),
        ]
    )
    try:
        result = await client.geosearch_polygon(
            "geo", SQUARE, holes=HOLE, sort="ASC", count=5, withdist=True
        )
        assert result == [[b"m2", 2.0], [b"m4", 4.0]]
        assert requested_counts(index) == [5]
    finally:
        await client.aclose()


async def test_withcoord_and_withdist_passthrough():
    client, index = make_client(
        [
            (b"m1", 5.0, 5.0, 1.0),  # inside the hole
            (b"m2", 2.0, 2.0, 2.0),
        ]
    )
    try:
        result = await client.geosearch_polygon(
            "geo", SQUARE, holes=HOLE, withcoord=True, withdist=True
        )
        assert result == [[b"m2", 2.0, (2.0, 2.0)]]
    finally:
        await client.aclose()


async def test_invalid_polygon_rejected_before_any_command():
    client, index = make_client([(b"m1", 2.0, 2.0, 1.0)])
    try:
        with pytest.raises(DataError):
            # self-intersecting (bowtie) outer ring
            await client.geosearch_polygon("geo", [(0, 0), (10, 10), (0, 10), (10, 0)])
        with pytest.raises(DataError):
            # invalid latitude
            await client.geosearch_polygon("geo", [(0, 0), (10, 0), (10, 91)])
        assert index.requests == []
    finally:
        await client.aclose()


async def test_geopolygon_instance_shared_with_sync_client():
    polygon = GeoPolygon(SQUARE, holes=HOLE)
    client, index = make_client([(b"m1", 2.0, 2.0, 1.0), (b"m2", 5.0, 5.0, 2.0)])
    try:
        assert await client.geosearch_polygon("geo", polygon) == [b"m1"]
    finally:
        await client.aclose()
