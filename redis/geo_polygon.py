from __future__ import annotations

import math
from typing import Sequence

from redis.exceptions import DataError
from redis.typing import GeoSearchResponse

# Mean Earth radius in meters, matching the value Redis uses for its own
# geospatial distance calculations, so the bounding circle radius is
# expressed on the same scale the server uses.
EARTH_RADIUS_METERS = 6372797.560856

# Meters per GEOSEARCH unit, matching the Redis conversion factors.
UNIT_TO_METERS = {"m": 1.0, "km": 1000.0, "mi": 1609.34, "ft": 0.3048}

# Tolerance used when deciding whether a point lies exactly on a ring
# boundary segment, relative to the segment's squared length in degrees.
_BOUNDARY_EPS = 1e-12

Point = tuple[float, float]


def _wrap_delta(delta: float) -> float:
    """Map a longitude difference into the [-180, 180) range."""
    return (delta + 180.0) % 360.0 - 180.0


def _normalize_lon(lon: float) -> float:
    """Normalize a longitude into the [-180, 180) range."""
    return (lon + 180.0) % 360.0 - 180.0


def _validate_points(ring: Sequence[Sequence[float]], role: str) -> list[Point]:
    points = []
    for point in ring:
        if len(point) != 2:
            raise DataError(
                f"GEOSEARCH polygon {role} ring points must be"
                " (longitude, latitude) pairs"
            )
        lon, lat = point[0], point[1]
        if (
            isinstance(lon, (bool, str, bytes))
            or isinstance(lat, (bool, str, bytes))
            or not isinstance(lon, (int, float))
            or not isinstance(lat, (int, float))
        ):
            raise DataError(
                f"GEOSEARCH polygon {role} ring coordinates must be numbers"
            )
        lon = float(lon)
        lat = float(lat)
        if not math.isfinite(lon) or not math.isfinite(lat):
            raise DataError(f"GEOSEARCH polygon {role} ring coordinates must be finite")
        if lat < -90.0 or lat > 90.0:
            raise DataError(
                f"GEOSEARCH polygon {role} ring latitude must be in the [-90, 90] range"
            )
        if lon < -180.0 or lon > 180.0:
            raise DataError(
                f"GEOSEARCH polygon {role} ring longitude must be"
                " in the [-180, 180] range"
            )
        points.append((lon, lat))
    return points


def _unwrap_ring(points: list[Point], reference_lon: float) -> list[Point]:
    """
    Unwrap ring longitudes around the given reference so that consecutive
    vertices always take the shortest step, letting rings cross the
    antimeridian (+-180 longitude) without a false 360-degree jump.
    """
    unwrapped = []
    prev = reference_lon + _wrap_delta(points[0][0] - reference_lon)
    unwrapped.append((prev, points[0][1]))
    for lon, lat in points[1:]:
        current = prev + _wrap_delta(lon - prev)
        unwrapped.append((current, lat))
        prev = current
    return unwrapped


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(p: Point, a: Point, b: Point) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    scale = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1.0)
    if abs(cross) > _BOUNDARY_EPS * scale * scale:
        return False
    return (
        min(a[0], b[0]) - _BOUNDARY_EPS <= p[0] <= max(a[0], b[0]) + _BOUNDARY_EPS
        and min(a[1], b[1]) - _BOUNDARY_EPS <= p[1] <= max(a[1], b[1]) + _BOUNDARY_EPS
    )


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and d1 != 0 and d2 != 0:
        if ((d3 > 0) != (d4 > 0)) and d3 != 0 and d4 != 0:
            return True
    if d1 == 0 and _point_on_segment(p1, p3, p4):
        return True
    if d2 == 0 and _point_on_segment(p2, p3, p4):
        return True
    if d3 == 0 and _point_on_segment(p3, p1, p2):
        return True
    if d4 == 0 and _point_on_segment(p4, p1, p2):
        return True
    return False


def _ring_self_intersects(ring: list[Point]) -> bool:
    n = len(ring)
    for i in range(n):
        a1, a2 = ring[i], ring[(i + 1) % n]
        for j in range(i + 1, n):
            # skip segment pairs that share a ring vertex
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            b1, b2 = ring[j], ring[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _prepare_ring(
    ring: Sequence[Sequence[float]], role: str, reference_lon: float | None
) -> list[Point]:
    points = _validate_points(ring, role)
    if len(points) < 3:
        raise DataError(f"GEOSEARCH polygon {role} ring must have at least 3 vertices")
    if points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise DataError(
            f"GEOSEARCH polygon {role} ring must have at least 3 distinct vertices"
        )
    if reference_lon is None:
        reference_lon = points[0][0]
    unwrapped = _unwrap_ring(points, reference_lon)
    for i in range(1, len(unwrapped)):
        if unwrapped[i] == unwrapped[i - 1]:
            raise DataError(
                f"GEOSEARCH polygon {role} ring has duplicate consecutive vertices"
            )
    if _ring_self_intersects(unwrapped):
        raise DataError(f"GEOSEARCH polygon {role} ring is self-intersecting")
    return unwrapped


def _point_in_ring(p: Point, ring: list[Point]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > p[1]) != (yj > p[1]):
            x_cross = xi + (p[1] - yi) * (xj - xi) / (yj - yi)
            if p[0] < x_cross:
                inside = not inside
        j = i
    return inside


def _point_on_ring(p: Point, ring: list[Point]) -> bool:
    j = len(ring) - 1
    for i in range(len(ring)):
        if _point_on_segment(p, ring[j], ring[i]):
            return True
        j = i
    return False


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(a))


class GeoPolygon:
    """
    A polygonal geographic area with optional holes, used by
    the geosearch_polygon helper to filter GEOSEARCH candidates
    client side.

    The outer ring is a sequence of at least three (longitude, latitude)
    pairs; the ring may be closed (first point repeated at the end) or
    not. The optional holes are rings in the same format, describing
    areas excluded from the polygon.

    Rings may cross the antimeridian (+-180 longitude); longitudes must
    still be given in the [-180, 180] range. Points lying exactly on a
    ring boundary - of the outer ring or of any hole - are consistently
    treated as inside the polygon. Self-intersecting rings and invalid
    coordinates are rejected with DataError on construction.
    """

    def __init__(
        self,
        outer: Sequence[Sequence[float]],
        holes: Sequence[Sequence[Sequence[float]]] | None = None,
    ) -> None:
        self._outer = _prepare_ring(outer, "outer", None)
        self._reference_lon = self._outer[0][0]
        self._holes = [
            _prepare_ring(hole, "hole", self._reference_lon) for hole in holes or []
        ]

    def bounding_circle(self) -> tuple[float, float, float]:
        """
        Return the (longitude, latitude, radius_meters) circle that
        covers every polygon vertex, suitable for a server-side
        GEOSEARCH ... BYRADIUS pre-filter. The radius carries a small
        epsilon so members sitting exactly on a vertex are not lost to
        floating-point rounding.
        """
        vertices = list(self._outer)
        for hole in self._holes:
            vertices.extend(hole)
        lons = [v[0] for v in vertices]
        lats = [v[1] for v in vertices]
        center_lon = (min(lons) + max(lons)) / 2
        center_lat = (min(lats) + max(lats)) / 2
        radius = max(
            _haversine(center_lon, center_lat, lon, lat) for lon, lat in vertices
        )
        radius += max(radius * 1e-9, 1e-3)
        return _normalize_lon(center_lon), center_lat, radius

    def contains(self, longitude: float, latitude: float) -> bool:
        """
        Return True when the point is inside the polygon. Boundary points
        of the outer ring and of every hole are treated as inside.
        """
        lon = self._reference_lon + _wrap_delta(longitude - self._reference_lon)
        point = (lon, latitude)
        if _point_on_ring(point, self._outer):
            return True
        if not _point_in_ring(point, self._outer):
            return False
        for hole in self._holes:
            if _point_on_ring(point, hole):
                return True
            if _point_in_ring(point, hole):
                return False
        return True


def prepare_polygon_search(
    outer: Sequence[Sequence[float]] | GeoPolygon,
    holes: Sequence[Sequence[Sequence[float]]] | None,
    unit: str,
    count: int | None,
) -> tuple[GeoPolygon, float, float, float]:
    """
    Validate the polygon query arguments and return the polygon together
    with the (longitude, latitude, radius) bounding circle, the radius
    expressed in the given unit. Raises DataError before any command is
    sent when the polygon or the arguments are invalid.
    """
    if isinstance(outer, GeoPolygon):
        if holes is not None:
            raise DataError(
                "GEOSEARCH polygon holes can't be set together"
                " with a GeoPolygon instance"
            )
        polygon = outer
    else:
        polygon = GeoPolygon(outer, holes)
    if unit is None or unit.lower() not in UNIT_TO_METERS:
        raise DataError("GEOSEARCH invalid unit")
    if count is not None and count <= 0:
        raise DataError("GEOSEARCH polygon count must be a positive integer")
    longitude, latitude, radius = polygon.bounding_circle()
    return polygon, longitude, latitude, radius / UNIT_TO_METERS[unit.lower()]


def filter_geosearch_response(
    polygon: GeoPolygon,
    response: GeoSearchResponse,
    withcoord: bool,
    withdist: bool,
    withhash: bool,
) -> GeoSearchResponse:
    """
    Drop GEOSEARCH items outside the polygon and reshape the survivors to
    match the requested withcoord/withdist/withhash flags.

    The response must have been requested with WITHCOORD so every item
    carries its (longitude, latitude) pair as the last element; the
    coordinate is stripped again unless the caller asked for it. Distances
    and hashes are the exact values returned by the server.
    """
    filtered = []
    keep_extras = withcoord or withdist or withhash
    for item in response:
        coord = item[-1]
        if not polygon.contains(coord[0], coord[1]):
            continue
        if not keep_extras:
            filtered.append(item[0])
        elif withcoord:
            filtered.append(item)
        else:
            filtered.append(item[:-1])
    return filtered
