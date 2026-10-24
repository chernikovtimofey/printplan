"""Rotational features used to choose between 3D and 5D printing.

For every candidate axis direction (model X/Y/Z and the principal axes) the
module computes:

* axis location — the line that side-face normals point to (robust least
  squares), so shafts with eccentric lobes are handled; falls back to the
  centre of mass;
* radial_fraction — share of side area whose normals pass through that line
  (cylindrical surfaces around the axis);
* continuous symmetry index — how far the surface moves when rotated by
  37°, 71°, 113° around the axis (≈0 for bodies of revolution);
* cyclic order — the largest N ≤ 48 for which rotation by 360°/N maps the
  surface onto itself (impellers, fans); 0 means continuous symmetry;
* through-bore along the axis and its diameter;
* prismatic fraction — share of area whose normal is parallel or perpendicular
  to the axis (2.5D extrusions such as gears);
* axial length and outer diameter.

Symmetry indices are mean nearest-neighbour distances between a rotated
surface sample and a denser reference sample, minus the same statistic for the
unrotated sample (sampling noise), divided by the bounding-box diagonal.
"""
from collections import OrderedDict

import numpy as np

from . import geometry

CONTINUOUS_ANGLES_DEG = (37.0, 71.0, 113.0)
MAX_CYCLIC_ORDER = 48
CONTINUOUS_THRESHOLD = 5e-4   # continuous index below this: body of revolution
CYCLIC_THRESHOLD = 3e-4       # index of a 360/N rotation below this: N-fold symmetry
RADIAL_AXIS_FRACTION = 0.15   # share of area on coaxial cylinders that defines an axis
SIDE_FACE_DOT = 0.15          # |n·d| below this: face is a "side" face for axis location
PRISMATIC_DOT = 0.02          # |n·d| below this or above 1 - this: parallel / perpendicular


def _sample_sizes():
    if geometry.has_scipy():
        return 3000, 30000
    return 1500, 12000


class AxisFeatures(object):
    def __init__(self, label, direction):
        self.label = label
        self.direction = geometry.unit(direction)
        self.point = None
        self.radial_fraction = 0.0
        self.continuous_index = None
        self.cyclic_order = 1         # 1 = none, 0 = continuous (body of revolution)
        self.cyclic_index = None
        self.body_of_revolution = False
        self.bore = None
        self.prismatic_fraction = 0.0
        self.axial_length = 0.0
        self.outer_diameter = 0.0

    def as_dict(self):
        d = OrderedDict()
        d["label"] = self.label
        d["direction"] = [round(float(x), 4) for x in self.direction]
        d["point_mm"] = [round(float(x), 2) for x in self.point]
        d["radial_fraction"] = round(float(self.radial_fraction), 3)
        d["continuous_symmetry_index"] = _r(self.continuous_index, 5)
        d["body_of_revolution"] = bool(self.body_of_revolution)
        d["cyclic_order"] = int(self.cyclic_order)
        d["cyclic_symmetry_index"] = _r(self.cyclic_index, 5)
        d["prismatic_fraction"] = round(float(self.prismatic_fraction), 3)
        d["axial_length_mm"] = round(float(self.axial_length), 2)
        d["outer_diameter_mm"] = round(float(self.outer_diameter), 2)
        d["through_bore"] = self.bore
        return d


def _r(value, digits):
    return None if value is None else round(float(value), digits)


class _Sampler(object):
    """Shared surface samples + nearest-neighbour structure for one mesh."""

    def __init__(self, mesh, seed=1):
        n_query, n_ref = _sample_sizes()
        self.query = geometry.sample_surface(mesh.triangles, mesh.areas, n_query, seed=seed)
        reference = geometry.sample_surface(mesh.triangles, mesh.areas, n_ref, seed=seed + 1)
        self.nn = geometry.NearestNeighbors(reference)
        base = self.nn.distances(self.query)
        self.baseline = float(base.mean())
        self.quick = slice(0, max(len(self.query) // 6, 100))
        self.quick_baseline = float(base[self.quick].mean())
        self.diagonal = mesh.diagonal

    def deviation(self, point, direction, angle, quick=False):
        rot = geometry.axis_angle_matrix(direction, angle)
        query = self.query[self.quick] if quick else self.query
        baseline = self.quick_baseline if quick else self.baseline
        moved = (query - point).dot(rot.T) + point
        mean = float(self.nn.distances(moved).mean())
        return max(0.0, mean - baseline) / self.diagonal


def candidate_directions(mesh, min_angle_deg=3.0):
    """Model axes first (CAD exports are usually axis aligned), then principal axes."""
    cands = [("X", [1.0, 0, 0]), ("Y", [0, 1.0, 0]), ("Z", [0, 0, 1.0])]
    for i, axis in enumerate(mesh.pca_axes):
        cands.append(("PCA%d" % (i + 1), axis))
    out = []
    cos_tol = np.cos(np.radians(min_angle_deg))
    for label, d in cands:
        d = geometry.unit(d)
        if any(abs(float(np.dot(d, o[1]))) > cos_tol for o in out):
            continue
        out.append((label, d))
    return out


def _basis(direction):
    helper = np.array([1.0, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1.0, 0])
    u = geometry.unit(np.cross(direction, helper))
    v = np.cross(direction, u)
    return u, v


def locate_axis(mesh, direction, iterations=8):
    """Robust least-squares point closest to the normal lines of side faces.

    Returns (point, radial_fraction).
    """
    d = direction
    dots = mesh.normals.dot(d)
    side = (np.abs(dots) < SIDE_FACE_DOT) & (mesh.areas > 0)
    com = mesh.center_of_mass
    if side.sum() < 3:
        return com.copy(), 0.0
    u, v = _basis(d)
    p = np.stack([mesh.centroids[side].dot(u), mesh.centroids[side].dot(v)], axis=1)
    n = np.stack([mesh.normals[side].dot(u), mesh.normals[side].dot(v)], axis=1)
    nlen = np.linalg.norm(n, axis=1)
    n = n / np.maximum(nlen, 1e-12)[:, None]
    w0 = mesh.areas[side]
    x = np.array([com.dot(u), com.dot(v)])
    scale = max(0.02 * mesh.diagonal, 0.2)
    weights = w0.copy()
    for _ in range(iterations):
        # minimise sum w * |(I - n n^T)(x - p)|^2
        proj = np.eye(2)[None, :, :] - n[:, :, None] * n[:, None, :]
        A = (weights[:, None, None] * proj).sum(axis=0)
        b = np.einsum("i,ijk,ik->j", weights, proj, p)
        if abs(np.linalg.det(A)) < 1e-12:
            break
        x = np.linalg.solve(A, b)
        rel = x[None, :] - p
        resid = np.abs(rel[:, 0] * n[:, 1] - rel[:, 1] * n[:, 0])
        weights = w0 / (1.0 + (resid / scale) ** 2)
    rel = x[None, :] - p
    resid = np.abs(rel[:, 0] * n[:, 1] - rel[:, 1] * n[:, 0])
    tol = max(0.005 * mesh.diagonal, 0.1)
    side_area = float(w0.sum())
    radial = float(w0[resid < tol].sum()) / max(mesh.surface_area, 1e-12)
    point = x[0] * u + x[1] * v + float(com.dot(d)) * d
    return point, radial if side_area > 0 else 0.0


def through_bore(mesh, point, direction):
    """Through-hole along the axis line: the line must not hit the surface."""
    hits = geometry.ray_triangle_hits(point, direction, mesh.triangles, both_ways=True)
    if len(hits):
        return None
    dist = geometry.point_line_distance(mesh.vertices, point, direction)
    radius = float(dist.min())
    return OrderedDict([("diameter_mm", round(2.0 * radius, 2))])


def analyze_axis(mesh, sampler, label, direction):
    feat = AxisFeatures(label, direction)
    d = feat.direction
    feat.point, feat.radial_fraction = locate_axis(mesh, d)

    dots = np.abs(mesh.normals.dot(d))
    total = max(mesh.surface_area, 1e-12)
    feat.prismatic_fraction = float(mesh.areas[(dots < PRISMATIC_DOT) | (dots > 1 - PRISMATIC_DOT)].sum()) / total

    along = mesh.vertices.dot(d)
    feat.axial_length = float(along.max() - along.min())
    feat.outer_diameter = 2.0 * float(geometry.point_line_distance(mesh.vertices, feat.point, d).max())

    devs = [sampler.deviation(feat.point, d, np.radians(a)) for a in CONTINUOUS_ANGLES_DEG]
    feat.continuous_index = float(np.mean(devs))
    feat.bore = through_bore(mesh, feat.point, d)
    return feat


def scan_cyclic(sampler, feat, threshold=CYCLIC_THRESHOLD):
    """Largest N <= MAX_CYCLIC_ORDER whose 360/N rotation keeps the surface in place.

    A quick pass on a sample subset selects candidates; the full sample confirms.
    """
    feat.cyclic_order, feat.cyclic_index = 1, None
    for n in range(MAX_CYCLIC_ORDER, 1, -1):
        angle = 2.0 * np.pi / n
        if sampler.deviation(feat.point, feat.direction, angle, quick=True) > 4.0 * threshold:
            continue
        index = sampler.deviation(feat.point, feat.direction, angle)
        if index < threshold:
            feat.cyclic_order, feat.cyclic_index = n, index
            return


def analyze(mesh):
    """Rotational features for all candidate axes and the selected axis (or None)."""
    sampler = _Sampler(mesh)
    feats = []
    for label, d in candidate_directions(mesh):
        feat = analyze_axis(mesh, sampler, label, d)
        feat.body_of_revolution = feat.continuous_index < CONTINUOUS_THRESHOLD
        if feat.body_of_revolution:
            feat.cyclic_order, feat.cyclic_index = 0, feat.continuous_index  # 0 = continuous
        elif feat.radial_fraction >= RADIAL_AXIS_FRACTION or feat.continuous_index < 0.02:
            scan_cyclic(sampler, feat)
        feats.append(feat)
    return RotationalAnalysis(feats, select_axis(feats))


def select_axis(feats):
    """Body of revolution > N-fold symmetry > dominant coaxial cylinders > none."""
    revolution = [f for f in feats if f.body_of_revolution]
    if revolution:
        return min(revolution, key=lambda f: f.continuous_index)
    cyclic = [f for f in feats if f.cyclic_order >= 2]
    if cyclic:
        return max(cyclic, key=lambda f: (f.cyclic_order, f.radial_fraction))
    radial = [f for f in feats if f.radial_fraction >= RADIAL_AXIS_FRACTION]
    if radial:
        return max(radial, key=lambda f: f.radial_fraction)
    return None


class RotationalAnalysis(object):
    def __init__(self, axes, selected):
        self.axes = axes
        self.selected = selected

    def as_dict(self):
        d = OrderedDict()
        sel = self.selected
        d["has_axis"] = sel is not None
        if sel is not None:
            kind = "body_of_revolution" if sel.body_of_revolution else (
                "cyclic" if sel.cyclic_order >= 2 else "coaxial_cylinders")
            d["axis_kind"] = kind
            d["axis"] = sel.as_dict()
        d["candidates"] = [OrderedDict([
            ("label", f.label),
            ("radial_fraction", round(float(f.radial_fraction), 3)),
            ("continuous_symmetry_index", _r(f.continuous_index, 5)),
            ("cyclic_order", int(f.cyclic_order)),
        ]) for f in self.axes]
        return d
