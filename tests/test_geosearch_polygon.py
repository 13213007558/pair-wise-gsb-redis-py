"""
Tests for the client-side polygon GEOSEARCH helper.

These tests use a scripted fake in place of the server: it records every
GEOSEARCH request and answers with locally controlled candidates, so the
antimeridian crossing, holes, boundary points and the nearest-N count
expansion can be verified without a running Redis server.
"""

import pytest

import redis
from redis.exceptions import DataError
from redis.geo_polygon import GeoPolygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
HOLE = [[(4, 4), (6, 4), (6, 6), (4, 6)]]
# ring crossing the antimeridian: longitudes 170 .. -170 through 180
DATELINE_RING = [(170, 10), (-170, 10), (-170, 20), (170, 20)]


class FakeGeoIndex:
    """
    Scripted stand-in for the server side of GEOSEARCH.

    Candidates are (member, longitude, latitude, distance) tuples in
    ascending distance order; every executed command is recorded in
    the requests list for inspection.
    """

    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.requests = []

    def __call__(self, *args, **options):
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
    index = FakeGeoIndex(candidates)
    client.execute_command = index
    return client, index


def requested_counts(index):
    return [
        request[request.index(b"COUNT") + 1]
        for request in index.requests
        if b"COUNT" in request
    ]


def test_filters_candidates_and_records_bounding_circle():
    client, index = make_client(
        [
            (b"inside", 2.0, 2.0, 10.0),
            (b"outside", 20.0, 20.0, 20.0),
            (b"inside-2", 3.0, 3.0, 30.0),
        ]
    )
    result = client.geosearch_polygon("geo", SQUARE)
    assert result == [b"inside", b"inside-2"]

    assert len(index.requests) == 1
    request = index.requests[0]
    assert request[0] == "GEOSEARCH"
    assert request[1] == "geo"
    # server-side pre-filter: the polygon bounding circle
    i = request.index(b"FROMLONLAT")
    assert request[i + 1] == pytest.approx(5.0)
    assert request[i + 2] == pytest.approx(5.0)
    i = request.index(b"BYRADIUS")
    assert request[i + 1] == pytest.approx(GeoPolygon(SQUARE).bounding_circle()[2])
    assert request[i + 2] == "m"
    # no COUNT: a single request collects every candidate
    assert b"COUNT" not in request


def test_antimeridian_polygon():
    client, index = make_client(
        [
            (b"east", 175.0, 15.0, 10.0),
            (b"west", -175.0, 15.0, 20.0),
            (b"boundary", -180.0, 10.0, 30.0),
            (b"far", 0.0, 15.0, 40.0),
            (b"south", 175.0, 5.0, 50.0),
        ]
    )
    result = client.geosearch_polygon("geo", DATELINE_RING)
    assert result == [b"east", b"west", b"boundary"]

    request = index.requests[0]
    i = request.index(b"FROMLONLAT")
    # the bounding circle center sits on the meridian itself
    assert abs(request[i + 1]) == pytest.approx(180.0)
    assert request[i + 2] == pytest.approx(15.0)


def test_hole_and_boundary_rule():
    client, index = make_client(
        [
            (b"hole", 5.0, 5.0, 10.0),
            (b"hole-edge", 4.0, 5.0, 20.0),
            (b"outer-edge", 0.0, 5.0, 30.0),
            (b"corner", 10.0, 10.0, 40.0),
            (b"inside", 2.0, 2.0, 50.0),
            (b"outside", 15.0, 5.0, 60.0),
        ]
    )
    result = client.geosearch_polygon("geo", SQUARE, holes=HOLE)
    # hole interior is excluded; ring boundaries (outer and hole alike)
    # are consistently treated as inside
    assert result == [b"hole-edge", b"outer-edge", b"corner", b"inside"]
    assert len(index.requests) == 1


def test_nearest_count_expands_until_enough_polygon_matches():
    client, index = make_client(
        [
            (b"m1", 5.0, 5.0, 1.0),  # nearest, but inside the hole
            (b"m2", 50.0, 5.0, 2.0),  # second nearest, outside the polygon
            (b"m3", 2.0, 2.0, 3.0),
            (b"m4", 3.0, 3.0, 4.0),
            (b"m5", 1.0, 1.0, 5.0),
        ]
    )
    result = client.geosearch_polygon(
        "geo", SQUARE, holes=HOLE, sort="ASC", count=2, withdist=True
    )
    # the nearest 2 inside the polygon, with the distances Redis returned
    assert result == [[b"m3", 3.0], [b"m4", 4.0]]
    # COUNT 2 alone would have been eaten by the hole and the outside
    # point, so the helper widened the candidate window instead
    assert requested_counts(index) == [2, 4]
    assert all(b"ASC" in request for request in index.requests)


def test_count_stops_when_candidates_exhausted():
    client, index = make_client(
        [
            (b"m1", 5.0, 5.0, 1.0),  # inside the hole
            (b"m2", 2.0, 2.0, 2.0),
            (b"m3", 50.0, 5.0, 3.0),  # outside
            (b"m4", 3.0, 3.0, 4.0),
        ]
    )
    result = client.geosearch_polygon(
        "geo", SQUARE, holes=HOLE, sort="ASC", count=5, withdist=True
    )
    # only 2 polygon matches exist; the server returned fewer candidates
    # than requested, proving exhaustion, so no further widening happens
    assert result == [[b"m2", 2.0], [b"m4", 4.0]]
    assert requested_counts(index) == [5]


def test_withcoord_and_withdist_passthrough():
    client, index = make_client(
        [
            (b"m1", 5.0, 5.0, 1.0),  # inside the hole
            (b"m2", 2.0, 2.0, 2.0),
        ]
    )
    result = client.geosearch_polygon(
        "geo", SQUARE, holes=HOLE, withcoord=True, withdist=True
    )
    assert result == [[b"m2", 2.0, (2.0, 2.0)]]


def test_radius_is_converted_to_the_requested_unit():
    client, index = make_client([(b"m1", 175.0, 15.0, 1.0)])
    result = client.geosearch_polygon("geo", DATELINE_RING, unit="km")
    assert result == [b"m1"]
    request = index.requests[0]
    i = request.index(b"BYRADIUS")
    radius_m = GeoPolygon(DATELINE_RING).bounding_circle()[2]
    assert request[i + 1] == pytest.approx(radius_m / 1000)
    assert request[i + 2] == "km"


def test_geopolygon_instance_can_be_passed_directly():
    client, index = make_client([(b"m1", 2.0, 2.0, 1.0)])
    polygon = GeoPolygon(SQUARE, holes=HOLE)
    assert client.geosearch_polygon("geo", polygon) == [b"m1"]
    with pytest.raises(DataError):
        client.geosearch_polygon("geo", polygon, holes=HOLE)


@pytest.mark.parametrize(
    "outer,holes",
    [
        # self-intersecting (bowtie) outer ring
        ([(0, 0), (10, 10), (0, 10), (10, 0)], None),
        # self-intersecting hole
        (SQUARE, [[(4, 4), (8, 8), (4, 8), (8, 4)]]),
        # invalid latitudes
        ([(0, 0), (10, 0), (10, 91)], None),
        ([(0, 0), (10, 0), (10, -90.5)], None),
        # invalid longitude
        ([(0, 0), (181, 0), (10, 10)], None),
        # non-finite coordinates
        ([(0, 0), (10, 0), (float("nan"), 5)], None),
        ([(0, 0), (10, 0), (float("inf"), 5)], None),
        # too few vertices
        ([(0, 0), (10, 0)], None),
        # duplicate consecutive vertices
        ([(0, 0), (0, 0), (10, 0), (10, 10)], None),
    ],
)
def test_invalid_polygon_rejected_before_any_command(outer, holes):
    client, index = make_client([(b"m1", 2.0, 2.0, 1.0)])
    with pytest.raises(DataError):
        client.geosearch_polygon("geo", outer, holes=holes)
    assert index.requests == []


def test_invalid_arguments_rejected_before_any_command():
    client, index = make_client([(b"m1", 2.0, 2.0, 1.0)])
    with pytest.raises(DataError):
        client.geosearch_polygon("geo", SQUARE, count=0)
    with pytest.raises(DataError):
        client.geosearch_polygon("geo", SQUARE, unit="yard")
    assert index.requests == []
