"""Methods for measuring (between) geometries."""

import warnings
from math import prod

import numpy as np

from shapely import lib

__all__ = [
    "area",
    "bounds",
    "distance",
    "frechet_distance",
    "hausdorff_distance",
    "length",
    "minimum_bounding_radius",
    "minimum_clearance",
    "total_bounds",
]


def area(geometry, **kwargs):
    """Compute the area of a (multi)polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the area.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import MultiPolygon, Polygon
    >>> polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    >>> shapely.area(polygon)
    100.0
    >>> polygon2 = Polygon([(10, 10), (10, 20), (20, 20), (20, 10), (10, 10)])
    >>> shapely.area(MultiPolygon([polygon, polygon2]))
    200.0
    >>> shapely.area(Polygon())
    0.0
    >>> shapely.area(None)
    nan

    """
    return lib.area(geometry, **kwargs)


def _distance_point_broadcast(a, b, kwargs):
    # Decode coordinates once when broadcasting reuses them often enough to
    # amortize extraction. Keep custom array dispatch on the geometry ufunc.
    if type(a) is not np.ndarray or type(b) is not np.ndarray:
        return None
    if a.size * b.size < 1024:
        return None
    if a.dtype != object or b.dtype != object or kwargs.keys() - {"out", "where"}:
        return None
    out = kwargs.get("out")
    if type(out) is tuple:
        if len(out) != 1:
            return None
        out = out[0]
    if out is not None and (type(out) is not np.ndarray or out.dtype != np.float64):
        return None
    where = kwargs.get("where", True)
    if not isinstance(where, (bool, np.bool_)) and not (
        type(where) is np.ndarray and where.dtype == bool
    ):
        return None
    try:
        size = prod(np.broadcast_shapes(a.shape, b.shape))
    except ValueError:
        return None
    if size < 1024 or size < 8 * (a.size + b.size):
        return None

    try:
        if not np.all(lib.get_type_id(a) == 0) or not np.all(lib.get_type_id(b) == 0):
            return None
    except TypeError:
        # A non-geometry object may be excluded by `where`; let the original
        # ufunc apply its mask before validating individual objects.
        return None
    x1, y1 = lib.get_x(a), lib.get_y(a)
    x2, y2 = lib.get_x(b), lib.get_y(b)
    if not all(np.all(np.isfinite(c)) for c in (x1, y1, x2, y2)):
        # Empty and non-finite points retain GEOS's handling and error behavior.
        return None
    return lib._distance_points(x1, y1, x2, y2, **kwargs)


def distance(a, b, **kwargs):
    """Compute the Cartesian distance between two geometries.

    Parameters
    ----------
    a, b : Geometry or array_like
        Geometry or geometries to compute the distance between.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point, Polygon
    >>> point = Point(0, 0)
    >>> shapely.distance(Point(10, 0), point)
    10.0
    >>> shapely.distance(LineString([(1, 1), (1, -1)]), point)
    1.0
    >>> shapely.distance(Polygon([(3, 0), (5, 0), (5, 5), (3, 5), (3, 0)]), point)
    3.0
    >>> shapely.distance(Point(), point)
    nan
    >>> shapely.distance(None, point)
    nan

    """
    if (
        type(a) is np.ndarray
        and a.size > 8
        and type(b) is np.ndarray
        and b.size > 8
        and a.shape != b.shape
    ):
        result = _distance_point_broadcast(a, b, kwargs)
        if result is not None:
            return result
    return lib.distance(a, b, **kwargs)


def bounds(geometry, **kwargs):
    """Compute the bounds (extent) of a geometry.

    For each geometry these 4 numbers are returned: min x, min y, max x, max y.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the bounds.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point, Polygon
    >>> shapely.bounds(Point(2, 3)).tolist()
    [2.0, 3.0, 2.0, 3.0]
    >>> shapely.bounds(LineString([(0, 0), (0, 2), (3, 2)])).tolist()
    [0.0, 0.0, 3.0, 2.0]
    >>> shapely.bounds(Polygon()).tolist()
    [nan, nan, nan, nan]
    >>> shapely.bounds(None).tolist()
    [nan, nan, nan, nan]

    """
    return lib.bounds(geometry, **kwargs)


def total_bounds(geometry, **kwargs):
    """Compute the total bounds (extent) of the geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the total bounds.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Returns
    -------
    numpy ndarray of [xmin, ymin, xmax, ymax]

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, Point, Polygon
    >>> shapely.total_bounds(Point(2, 3)).tolist()
    [2.0, 3.0, 2.0, 3.0]
    >>> shapely.total_bounds([Point(2, 3), Point(4, 5)]).tolist()
    [2.0, 3.0, 4.0, 5.0]
    >>> shapely.total_bounds([
    ...     LineString([(0, 1), (0, 2), (3, 2)]),
    ...     LineString([(4, 4), (4, 6), (6, 7)])
    ... ]).tolist()
    [0.0, 1.0, 6.0, 7.0]
    >>> shapely.total_bounds(Polygon()).tolist()
    [nan, nan, nan, nan]
    >>> shapely.total_bounds([Polygon(), Point(2, 3)]).tolist()
    [2.0, 3.0, 2.0, 3.0]
    >>> shapely.total_bounds(None).tolist()
    [nan, nan, nan, nan]

    """
    b = bounds(geometry, **kwargs)
    if b.ndim == 1:
        return b

    with warnings.catch_warnings():
        # ignore 'All-NaN slice encountered' warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array(
            [
                np.nanmin(b[..., 0]),
                np.nanmin(b[..., 1]),
                np.nanmax(b[..., 2]),
                np.nanmax(b[..., 3]),
            ]
        )


def length(geometry, **kwargs):
    """Compute the length of a (multi)linestring or polygon perimeter.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the length.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString, MultiLineString, Polygon
    >>> shapely.length(LineString([(0, 0), (0, 2), (3, 2)]))
    5.0
    >>> shapely.length(MultiLineString([
    ...     LineString([(0, 0), (1, 0)]),
    ...     LineString([(1, 0), (2, 0)])
    ... ]))
    2.0
    >>> shapely.length(Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]))
    40.0
    >>> shapely.length(LineString())
    0.0
    >>> shapely.length(None)
    nan

    """
    return lib.length(geometry, **kwargs)


def hausdorff_distance(a, b, densify=None, **kwargs):
    """Compute the discrete, undirected Hausdorff distance between two geometries.

    The Hausdorff distance is a measure of similarity: it is the greatest
    distance between any point in A and the closest point in B.

    The discrete distance is an approximation of this metric: only vertices are
    considered. The parameter 'densify' makes this approximation less coarse by
    splitting the line segments between vertices before computing the distance.

    This implementation is undirected or symmetric Hausdorff, meaning that the
    distance from A to B is the same as from B to A.

    Parameters
    ----------
    a, b : Geometry or array_like
        Geometry or geometries to compute the distance between.
    densify : float or array_like, optional
        The value of densify is required to be between 0 and 1.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> line1 = LineString([(130, 0), (0, 0), (0, 150)])
    >>> line2 = LineString([(10, 10), (10, 150), (130, 10)])
    >>> shapely.hausdorff_distance(line1, line2)
    14.142135623730951
    >>> shapely.hausdorff_distance(line1, line2, densify=0.5)
    70.0
    >>> shapely.hausdorff_distance(line1, LineString())
    nan
    >>> shapely.hausdorff_distance(line1, None)
    nan

    """
    if densify is None:
        return lib.hausdorff_distance(a, b, **kwargs)
    else:
        return lib.hausdorff_distance_densify(a, b, densify, **kwargs)


def frechet_distance(a, b, densify=None, **kwargs):
    """Compute the discrete Fréchet distance between two geometries.

    The Fréchet distance is a measure of similarity: it is the greatest
    distance between any point in A and the closest point in B. The discrete
    distance is an approximation of this metric: only vertices are considered.
    The parameter 'densify' makes this approximation less coarse by splitting
    the line segments between vertices before computing the distance.

    Fréchet distance sweep continuously along their respective curves
    and the direction of curves is significant. This makes it a better measure
    of similarity than Hausdorff distance for curve or surface matching.

    Parameters
    ----------
    a, b : Geometry or array_like
        Geometry or geometries to compute the distance between.
    densify : float or array_like, optional
        The value of densify is required to be between 0 and 1.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import LineString
    >>> line1 = LineString([(0, 0), (100, 0)])
    >>> line2 = LineString([(0, 0), (50, 50), (100, 0)])
    >>> shapely.frechet_distance(line1, line2)
    70.71067811865476
    >>> shapely.frechet_distance(line1, line2, densify=0.5)
    50.0
    >>> shapely.frechet_distance(line1, LineString())
    nan
    >>> shapely.frechet_distance(line1, None)
    nan

    """
    if densify is None:
        return lib.frechet_distance(a, b, **kwargs)
    return lib.frechet_distance_densify(a, b, densify, **kwargs)


def minimum_clearance(geometry, **kwargs):
    """Compute the Minimum Clearance distance.

    A geometry's "minimum clearance" is the smallest distance by which
    a vertex of the geometry could be moved to produce an invalid geometry.

    If no minimum clearance exists for a geometry (for example, a single
    point, or an empty geometry), infinity is returned.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the minimum clearance.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> polygon = Polygon([(0, 0), (0, 10), (5, 6), (10, 10), (10, 0), (5, 4), (0, 0)])
    >>> shapely.minimum_clearance(polygon)
    2.0
    >>> shapely.minimum_clearance(Polygon())
    inf
    >>> shapely.minimum_clearance(None)
    nan

    See Also
    --------
    minimum_clearance_line

    """
    return lib.minimum_clearance(geometry, **kwargs)


def minimum_bounding_radius(geometry, **kwargs):
    """Compute the radius of the minimum bounding circle of an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
        Geometry or geometries for which to compute the minimum bounding radius.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.


    Examples
    --------
    >>> import shapely
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
    >>> shapely.minimum_bounding_radius(
    ...     Polygon([(0, 5), (5, 10), (10, 5), (5, 0), (0, 5)])
    ... )
    5.0
    >>> shapely.minimum_bounding_radius(LineString([(1, 1), (1, 10)]))
    4.5
    >>> shapely.minimum_bounding_radius(MultiPoint([(2, 2), (4, 2)]))
    1.0
    >>> shapely.minimum_bounding_radius(Point(0, 1))
    0.0
    >>> shapely.minimum_bounding_radius(GeometryCollection())
    0.0

    See Also
    --------
    minimum_bounding_circle

    """
    return lib.minimum_bounding_radius(geometry, **kwargs)
