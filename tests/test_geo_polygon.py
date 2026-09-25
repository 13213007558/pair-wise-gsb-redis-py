from unittest import mock

import pytest

import redis
from redis.exceptions import DataError
from redis.geo_polygon import GeoPolygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
HOLE = [(4, 4), (6, 4), (6, 6), (4, 6)]
ANTIMERIDIAN_RING = [(170, 40), (-170, 40), (-170, 50), (170, 50)]


def _row(member, dist, lon, lat):
    return [member, dist, (lon, lat)]


class TestGeoPolygon:
    def test_invalid_latitude_rejected(self):
        with pytest.raises(DataError):
            GeoPolygon([(0, 0), (1, 0), (1, 86)])
        with pytest.raises(DataError):
            GeoPolygon([(0, 0), (1, 0), (1, -85.06112878)])

    def test_invalid_longitude_rejected(self):
        with pytest.raises(DataError):
            GeoPolygon([(0, 0), (180.5, 0), (1, 1)])
        with pytest.raises(DataError):
            GeoPolygon([(0, 0), (-181, 0), (1, 1)])

    def test_too_few_vertices_rejected(self):
        with pytest.raises(DataError):
            GeoPolygon([(0, 0), (1, 1)])

    def test_self_intersecting_outer_ring_rejected(self):
        bowtie = [(0, 0), (2, 2), (2, 0), (0, 2)]
        with pytest.raises(DataError):
            GeoPolygon(bowtie)

    def test_self_intersecting_hole_rejected(self):
        bowtie = [(1, 1), (3, 3), (3, 1), (1, 3)]
        with pytest.raises(DataError):
            GeoPolygon(SQUARE, holes=[bowtie])

    def test_closed_ring_matches_open_ring(self):
        closed = GeoPolygon(SQUARE + [SQUARE[0]])
        assert closed.contains(5, 5)
        assert not closed.contains(15, 5)

    def test_hole_excludes_points(self):
        polygon = GeoPolygon(SQUARE, holes=[HOLE])
        assert polygon.contains(1, 1)
        assert not polygon.contains(5, 5)
        assert not polygon.contains(20, 20)

    def test_boundary_rule(self):
        polygon = GeoPolygon(SQUARE, holes=[HOLE])
        # Boundary points belong to their ring: on the outer ring they are
        # inside the polygon, on a hole they are inside the hole.
        assert polygon.contains(0, 5)
        assert polygon.contains(10, 10)
        assert not polygon.contains(4, 5)
        assert not polygon.contains(6, 6)

    def test_antimeridian_crossing(self):
        polygon = GeoPolygon(ANTIMERIDIAN_RING)
        assert polygon.contains(179.9, 45)
        assert polygon.contains(-179.9, 45)
        assert polygon.contains(180, 45)
        assert polygon.contains(-180, 45)
        assert not polygon.contains(0, 45)
        assert not polygon.contains(179.9, 55)

    def test_antimeridian_bounding_box(self):
        polygon = GeoPolygon(ANTIMERIDIAN_RING)
        lon, lat, width, height = polygon.bounding_box("m")
        assert lon == -180.0
        assert lat == 45.0
        # 20 degrees of longitude at 40 degrees latitude (the box edge
        # closest to the equator, where longitude distances are largest).
        assert 1_600_000 < width < 1_800_000
        # 10 degrees of latitude.
        assert 1_100_000 < height < 1_200_000

    def test_invalid_unit_rejected(self):
        with pytest.raises(DataError):
            GeoPolygon(SQUARE).bounding_box("leagues")


class TestGeoSearchPolygon:
    # Candidates sorted by distance from the query center; "a" sits inside
    # the hole and must be filtered out client-side.
    A = _row(b"a", 1.0, 5, 5)
    B = _row(b"b", 2.0, 1, 1)
    C = _row(b"c", 3.0, 9, 1)
    D = _row(b"d", 4.0, 8, 8)

    def _client(self):
        return redis.Redis()

    def test_expands_count_until_enough_filtered_results(self):
        client = self._client()
        with mock.patch.object(client, "execute_command") as execute:
            execute.side_effect = [
                [self.A, self.B, self.C],
                [self.A, self.B, self.C, self.D],
            ]
            result = client.geosearch_polygon(
                "key",
                SQUARE,
                holes=[HOLE],
                count=3,
                sort="ASC",
                withcoord=True,
                withdist=True,
            )
        # nearest 3 *after* filtering: the hole member "a" is skipped and
        # the 4th nearest "d" backfills the result.
        assert result == [self.B, self.C, self.D]
        assert execute.call_count == 2
        first, second = execute.call_args_list
        assert first.args[:2] == ("GEOSEARCH", "key")
        assert b"FROMLONLAT" in first.args
        assert b"BYBOX" in first.args
        assert b"ASC" in first.args
        assert b"WITHCOORD" in first.args
        assert b"WITHDIST" in first.args
        assert list(first.args[first.args.index(b"COUNT") + 1 :])[0] == 3
        assert list(second.args[second.args.index(b"COUNT") + 1 :])[0] == 6

    def test_stops_when_bounding_box_is_exhausted(self):
        client = self._client()
        with mock.patch.object(client, "execute_command") as execute:
            execute.side_effect = [[self.A, self.B, self.C]]
            result = client.geosearch_polygon(
                "key",
                SQUARE,
                holes=[HOLE],
                count=5,
                sort="ASC",
                withcoord=True,
                withdist=True,
            )
        # The server returned fewer rows than requested, so the box is
        # exhausted and the partial filtered result is final.
        assert result == [self.B, self.C]
        assert execute.call_count == 1

    def test_without_count_queries_once(self):
        client = self._client()
        with mock.patch.object(client, "execute_command") as execute:
            execute.side_effect = [[self.A, self.B, self.C, self.D]]
            result = client.geosearch_polygon(
                "key", SQUARE, holes=[HOLE], withcoord=True, withdist=True
            )
        assert result == [self.B, self.C, self.D]
        assert execute.call_count == 1
        assert b"COUNT" not in execute.call_args_list[0].args

    def test_plain_response_shape(self):
        client = self._client()
        with mock.patch.object(client, "execute_command") as execute:
            execute.side_effect = [[[b"a", (5, 5)], [b"b", (1, 1)], [b"c", (9, 1)]]]
            result = client.geosearch_polygon("key", SQUARE, holes=[HOLE])
        # No WITH* flags requested: bare members, like GEOSEARCH.
        assert result == [b"b", b"c"]
        # WITHCOORD is still requested internally to allow filtering.
        assert b"WITHCOORD" in execute.call_args_list[0].args
        assert b"WITHDIST" not in execute.call_args_list[0].args

    def test_withdist_without_withcoord(self):
        client = self._client()
        with mock.patch.object(client, "execute_command") as execute:
            execute.side_effect = [[self.A, self.B]]
            result = client.geosearch_polygon(
                "key", SQUARE, holes=[HOLE], withdist=True
            )
        assert result == [[b"b", 2.0]]

    def test_antimeridian_polygon(self):
        client = self._client()
        near = _row(b"near", 1.0, 179.5, 45)
        across = _row(b"across", 2.0, -179.5, 45)
        far = _row(b"far", 3.0, 0, 45)
        with mock.patch.object(client, "execute_command") as execute:
            execute.side_effect = [[near, across, far]]
            result = client.geosearch_polygon(
                "key", ANTIMERIDIAN_RING, sort="ASC", withcoord=True, withdist=True
            )
        assert result == [near, across]
        args = execute.call_args_list[0].args
        from_lonlat = args[args.index(b"FROMLONLAT") + 1 :]
        assert list(from_lonlat[:2]) == [-180.0, 45.0]

    def test_invalid_input_rejected_before_any_command(self):
        client = self._client()
        with mock.patch.object(client, "execute_command") as execute:
            with pytest.raises(DataError):
                client.geosearch_polygon("key", [(0, 0), (2, 2), (2, 0), (0, 2)])
            with pytest.raises(DataError):
                client.geosearch_polygon("key", [(0, 0), (1, 0), (1, 90)])
            with pytest.raises(DataError):
                client.geosearch_polygon("key", SQUARE, count=0)
            with pytest.raises(DataError):
                client.geosearch_polygon("key", SQUARE, unit="leagues")
        execute.assert_not_called()
