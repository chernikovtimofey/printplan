"""Small geometric building blocks shared by the geometry stages (numpy only).

scipy is used when it is importable (cKDTree, connected components) unless the
environment variable PRINTPLAN_NO_SCIPY is set; every function has a pure
numpy fallback so the pipeline runs inside any STE Slicer distribution.
"""
import os

import numpy as np

_scipy_checked = False
_cKDTree = None
_csgraph = None
_sparse = None


def _load_scipy():
    global _scipy_checked, _cKDTree, _csgraph, _sparse
    if _scipy_checked:
        return
    _scipy_checked = True
    if os.environ.get("PRINTPLAN_NO_SCIPY"):
        return
    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import csgraph, coo_matrix
        _cKDTree = cKDTree
        _csgraph = csgraph
        _sparse = coo_matrix
    except Exception:  # scipy missing or broken: numpy fallbacks are used
        pass


def has_scipy():
    _load_scipy()
    return _cKDTree is not None


# ----------------------------------------------------------------------------- rotations

def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def axis_angle_matrix(axis, angle):
    """Rotation matrix for a rotation by `angle` (radians) around `axis` (right-hand rule)."""
    x, y, z = unit(axis)
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rotation_between(a, b):
    """Smallest rotation matrix R with R @ unit(a) == unit(b)."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate around any axis perpendicular to a
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        return axis_angle_matrix(np.cross(a, helper), np.pi)
    return axis_angle_matrix(v, np.arctan2(s, c))


# ----------------------------------------------------------------------------- triangles

def triangle_cross(tris):
    return np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])


def triangle_normals_areas(tris):
    cross = triangle_cross(tris)
    length = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    ok = length > 0
    normals[ok] = cross[ok] / length[ok, None]
    return normals, 0.5 * length


def sample_surface(tris, areas, count, seed=0):
    """Area-weighted uniform sample of `count` points on the triangle soup (deterministic)."""
    rng = np.random.RandomState(seed)
    total = float(areas.sum())
    if total <= 0 or len(tris) == 0:
        return np.zeros((0, 3))
    cdf = np.cumsum(areas) / total
    idx = np.searchsorted(cdf, rng.random_sample(count))
    idx = np.minimum(idx, len(tris) - 1)
    r1 = np.sqrt(rng.random_sample(count))
    r2 = rng.random_sample(count)
    t = tris[idx]
    return ((1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1]
            + (r1 * r2)[:, None] * t[:, 2])


# ----------------------------------------------------------------------------- nearest neighbours

class NearestNeighbors(object):
    """Nearest-neighbour distances to a fixed reference point set."""

    def __init__(self, reference):
        _load_scipy()
        self.reference = np.asarray(reference, dtype=np.float64)
        self._tree = _cKDTree(self.reference) if _cKDTree is not None else None
        self._ref_sq = (self.reference ** 2).sum(axis=1)

    def distances(self, query, chunk=512):
        query = np.asarray(query, dtype=np.float64)
        if self._tree is not None:
            return self._tree.query(query)[0]
        out = np.empty(len(query))
        for start in range(0, len(query), chunk):
            q = query[start:start + chunk]
            d2 = (q ** 2).sum(axis=1)[:, None] + self._ref_sq[None, :] - 2.0 * q.dot(self.reference.T)
            out[start:start + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        return out


# ----------------------------------------------------------------------------- rays and lines

def ray_triangle_hits(origin, direction, tris, both_ways=True, eps=1e-12):
    """Möller–Trumbore against all triangles; returns the hit parameters t.

    With `both_ways` the ray is treated as a full line (t may be negative).
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = unit(direction)
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    p = np.cross(direction, e2)
    det = (e1 * p).sum(axis=1)
    ok = np.abs(det) > eps
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    s = origin - v0
    u = (s * p).sum(axis=1) * inv
    q = np.cross(s, e1)
    v = (q * direction).sum(axis=1) * inv
    t = (e2 * q).sum(axis=1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
    if not both_ways:
        hit &= t > eps
    return t[hit]


def point_line_distance(points, origin, direction):
    d = unit(direction)
    rel = np.asarray(points, dtype=np.float64) - origin
    along = rel.dot(d)
    return np.linalg.norm(rel - along[:, None] * d[None, :], axis=1)


# ----------------------------------------------------------------------------- graphs

def connected_components(n, a, b):
    """Label the connected components of an undirected graph with n nodes and edges (a, b)."""
    _load_scipy()
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if n == 0:
        return 0, np.zeros(0, dtype=np.int64)
    if _csgraph is not None:
        graph = _sparse((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
        count, labels = _csgraph.connected_components(graph, directed=False)
        return int(count), labels.astype(np.int64)
    # Pure numpy: hook larger roots onto smaller ones, then pointer jumping.
    parent = np.arange(n, dtype=np.int64)
    while True:
        pa, pb = parent[a], parent[b]
        diff = pa != pb
        if not diff.any():
            break
        lo = np.minimum(pa[diff], pb[diff])
        hi = np.maximum(pa[diff], pb[diff])
        parent[hi] = lo
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
    _, labels = np.unique(parent, return_inverse=True)
    return int(labels.max()) + 1, labels.astype(np.int64)
