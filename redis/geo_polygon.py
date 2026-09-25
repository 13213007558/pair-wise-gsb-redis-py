"""Client-side polygon filtering on top of GEOSEARCH.

GEOSEARCH only supports circular (`BYRADIUS`) and axis-aligned rectangular
(`BYBOX`) areas.  This module implements the client side of the
`geosearch_polygon` helper: it validates a polygon (an outer ring plus
optional holes), computes the axis-aligned bounding box used to narrow the
server-side GEOSEARCH, and decides whether a candidate point lies inside the
polygon.

Boundary rule (applied consistently to the outer ring and to every hole):
a point lying exactly on a ring boundary is considered part of that ring.
Consequently points on the outer boundary are inside the polygon, while
points on a hole boundary are inside the hole and therefore excluded.

Polygons may cross the +-180 meridian: ring longitudes are unwrapped so that
consecutive vertices never jump by more than 180 degrees, and candidate
longitudes are shifted into the same frame before the containment test.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

from redis.exceptions import DataError

# Valid coordinate ranges, matching the Redis server (geo.c).
MIN_LATITUDE = -85.05112878
MAX_LATITUDE = 85.05112878
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0

# Mean earth radius in meters, the same constant Redis uses (geo.c).
EARTH_RADIUS_METERS = 6372797.560856

# Conversion factors to meters, matching Redis extractUnitOrReply().
UNIT_TO_METERS = {"m": 1.0, "km": 1000.0, "ft": 0.3048, "mi": 1609.34}

# Distance in degrees within which a point is considered to lie exactly on a
# ring boundary (~0.1 mm).  Applied consistently to outer rings and holes.
_BOUNDARY_EPSILON = 1e-9

# Inflation applied to the bounding box so that floating point rounding when
# converting between units can never clip a polygon vertex.
_BOX_INFLATION = 1 + 1e-9

# Minimum bounding box side in meters; GEOSEARCH rejects zero-sized boxes.
_MIN_BOX_SIDE_METERS = 1e-3

GeoPoint = tuple[float, float]
GeoRing = Sequence[GeoPoint]


class GeoPolygon:
    """
    A validated polygon (one outer ring plus optional holes) used by
    `geosearch_polygon` to filter GEOSEARCH candidates client-side.

    `outer` and each ring in `holes` are sequences of
    `(longitude, latitude)` pairs.  Rings may be closed (last vertex equal
    to the first) or open; a duplicated closing vertex is ignored.
    """

    def __init__(
        self,
        outer: GeoRing,
        holes: Sequence[GeoRing] | None = None,
    ) -> None:
        self.outer = self._prepare(outer, "outer ring")
        self.holes = [
            self._prepare(ring, f"hole {index}")
            for index, ring in enumerate(holes or [])
        ]
        lons = [point[0] for point in self.outer]
        lats = [point[1] for point in self.outer]
        self._min_lon = min(lons)
        self._max_lon = max(lons)
        self._min_lat = min(lats)
        self._max_lat = max(lats)
        self._outer_ref = (self._min_lon + self._max_lon) / 2.0
        self._hole_refs = [
            (min(p[0] for p in ring) + max(p[0] for p in ring)) / 2.0
            for ring in self.holes
        ]

    @staticmethod
    def _prepare(ring: GeoRing, what: str) -> list[GeoPoint]:
        points = _validate_ring(ring, what)
        unwrapped = _unwrap_ring(points)
        _check_self_intersections(unwrapped, what)
        return unwrapped

    def contains(self, longitude: float, latitude: float) -> bool:
        """
        Return True if the point is inside the outer ring (boundary
        included) and not inside any hole (hole boundary included).
        """
        x = _unwrap_lon(longitude, self._outer_ref)
        if not _ring_contains(self.outer, x, latitude):
            return False
        for ring, ref in zip(self.holes, self._hole_refs):
            if _ring_contains(ring, _unwrap_lon(longitude, ref), latitude):
                return False
        return True

    def bounding_box(self, unit: str) -> tuple[float, float, float, float]:
        """
        Return `(longitude, latitude, width, height)` of the smallest
        axis-aligned box fully containing the outer ring, with `width`
        and `height` expressed in `unit`.  The box is a superset of
        the polygon, suitable for narrowing a GEOSEARCH BYBOX query.
        """
        if unit not in UNIT_TO_METERS:
            raise DataError("GEOSEARCH invalid unit")
        to_meters = UNIT_TO_METERS[unit]
        height_m = _haversine_m(0.0, self._min_lat, 0.0, self._max_lat)
        # Longitude distances are largest at the latitude in the box that
        # is closest to the equator.
        if self._min_lat <= 0.0 <= self._max_lat:
            equator_lat = 0.0
        elif abs(self._min_lat) < abs(self._max_lat):
            equator_lat = self._min_lat
        else:
            equator_lat = self._max_lat
        width_m = _haversine_m(self._min_lon, equator_lat, self._max_lon, equator_lat)
        width = max(width_m * _BOX_INFLATION, _MIN_BOX_SIDE_METERS) / to_meters
        height = max(height_m * _BOX_INFLATION, _MIN_BOX_SIDE_METERS) / to_meters
        center_lon = _wrap_lon((self._min_lon + self._max_lon) / 2.0)
        center_lat = (self._min_lat + self._max_lat) / 2.0
        return center_lon, center_lat, width, height


def candidate_counts(count: int) -> Iterator[int]:
    """
    Yield successive GEOSEARCH COUNT values for the expansion loop,
    doubling each time so filtering can never starve the requested count.
    """
    while True:
        yield count
        count *= 2


def filter_rows(
    polygon: GeoPolygon,
    rows: list,
    withcoord: bool,
    withdist: bool,
    withhash: bool,
) -> list:
    """
    Drop rows whose coordinates fall outside `polygon` and reshape the
    survivors to the format `geosearch` would have returned for the
    requested `withcoord`/`withdist`/`withhash` flags.  Rows must be
    in the internal format produced by querying with WITHCOORD (plus
    WITHDIST/WITHHASH as requested): `[member, dist?, hash?, coord]`.
    """
    matched = [row for row in rows if polygon.contains(row[-1][0], row[-1][1])]
    if withcoord:
        return matched
    if withdist or withhash:
        return [row[:-1] for row in matched]
    return [row[0] for row in matched]


def _validate_ring(ring: GeoRing, what: str) -> list[GeoPoint]:
    points = []
    for vertex in ring:
        lon, lat = float(vertex[0]), float(vertex[1])
        if not MIN_LATITUDE <= lat <= MAX_LATITUDE:
            raise DataError(
                f"GEOSEARCH polygon {what} has invalid latitude {lat}: "
                f"must be between {MIN_LATITUDE} and {MAX_LATITUDE}"
            )
        if not MIN_LONGITUDE <= lon <= MAX_LONGITUDE:
            raise DataError(
                f"GEOSEARCH polygon {what} has invalid longitude {lon}: "
                f"must be between {MIN_LONGITUDE} and {MAX_LONGITUDE}"
            )
        points.append((lon, lat))
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        raise DataError(f"GEOSEARCH polygon {what} must have at least 3 vertices")
    return points


def _wrap_lon(lon: float) -> float:
    """Wrap a longitude into [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def _unwrap_lon(lon: float, reference: float) -> float:
    """
    Shift `lon` by a multiple of 360 degrees so that it lies within 180
    degrees of `reference`.  A difference of exactly 180 degrees is kept
    as-is, which keeps unwrapping deterministic.
    """
    return lon + 360.0 * round((reference - lon) / 360.0)


def _unwrap_ring(points: list[GeoPoint]) -> list[GeoPoint]:
    """
    Adjust longitudes so consecutive vertices never differ by more than
    180 degrees, giving the ring a consistent frame across the +-180
    meridian.
    """
    unwrapped = [points[0]]
    previous = points[0][0]
    for lon, lat in points[1:]:
        lon = _unwrap_lon(lon, previous)
        unwrapped.append((lon, lat))
        previous = lon
    return unwrapped


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters, computed like Redis (geo.c)."""
    lon1, lat1, lon2, lat2 = (
        math.radians(lon1),
        math.radians(lat1),
        math.radians(lon2),
        math.radians(lat2),
    )
    a = (
        math.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_METERS * math.asin(math.sqrt(a))


def _cross(origin: GeoPoint, a: GeoPoint, b: GeoPoint) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (
        b[0] - origin[0]
    )


def _collinear_on_segment(a: GeoPoint, b: GeoPoint, p: GeoPoint) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[
        1
    ] <= max(a[1], b[1])


def _segments_intersect(p1: GeoPoint, p2: GeoPoint, p3: GeoPoint, p4: GeoPoint) -> bool:
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0):
        return True
    return (
        (d1 == 0 and _collinear_on_segment(p3, p4, p1))
        or (d2 == 0 and _collinear_on_segment(p3, p4, p2))
        or (d3 == 0 and _collinear_on_segment(p1, p2, p3))
        or (d4 == 0 and _collinear_on_segment(p1, p2, p4))
    )


def _check_self_intersections(ring: list[GeoPoint], what: str) -> None:
    n = len(ring)
    for i in range(n):
        for j in range(i + 2, n):
            # Adjacent edges (sharing a vertex) cannot self-intersect.
            if i == 0 and j == n - 1:
                continue
            if _segments_intersect(
                ring[i], ring[(i + 1) % n], ring[j], ring[(j + 1) % n]
            ):
                raise DataError(f"GEOSEARCH polygon {what} is self-intersecting")


def _point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> bool:
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length == 0.0:
        return math.hypot(px - ax, py - ay) <= _BOUNDARY_EPSILON
    cross = (px - ax) * dy - (py - ay) * dx
    if abs(cross) > _BOUNDARY_EPSILON * length:
        return False
    dot = (px - ax) * dx + (py - ay) * dy
    return -_BOUNDARY_EPSILON * length <= dot <= length * (length + _BOUNDARY_EPSILON)


def _ring_contains(ring: list[GeoPoint], x: float, y: float) -> bool:
    """
    Even-odd ray casting against an unwrapped ring.  Points on the ring
    boundary are considered inside the ring.
    """
    inside = False
    n = len(ring)
    for i in range(n):
        ax, ay = ring[i]
        bx, by = ring[(i + 1) % n]
        if _point_on_segment(x, y, ax, ay, bx, by):
            return True
        if (ay > y) != (by > y):
            x_intersect = ax + (y - ay) * (bx - ax) / (by - ay)
            if x_intersect > x:
                inside = not inside
    return inside
