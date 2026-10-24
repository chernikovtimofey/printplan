"""Stage 3: orientation scoring for flat-layer (3D) printing.

Every candidate orientation is described by the model-frame direction that
ends up pointing up (+Z of the printer). For each candidate the model is
rotated, dropped onto the bed and four indicators are computed; their
weighted sum S_o (lower is better; weights from Du et al., 2026,
arXiv:2608.22128) ranks them.

5D modes do not use this ranking: the rotation axis is made vertical
(`axis_orientations`), and S_o only decides which end faces down.
"""
from collections import OrderedDict

import numpy as np

from . import geometry

WEIGHTS = OrderedDict([("support", 0.58), ("slender", 0.16), ("base", 0.12), ("height", 0.14)])
CANONICAL = OrderedDict([
    ("+x", (1.0, 0.0, 0.0)), ("-x", (-1.0, 0.0, 0.0)),
    ("+y", (0.0, 1.0, 0.0)), ("-y", (0.0, -1.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)), ("-z", (0.0, 0.0, -1.0)),
])
DEFAULT_SUPPORT_ANGLE = 50.0   # support_angle of the slicer, measured from the vertical
DEFAULT_BED_DELTA = 0.3        # mm, about one first layer
BASE_FACE_DEG = 5.0            # faces this close to horizontal count as bed contact
SUPPORT_FREE_MAX = 0.01        # r_max: R_support at or below this means "no supports needed"


class Orientation(object):
    def __init__(self, oid, up, kind):
        self.id = oid
        self.up = geometry.unit(up)        # model-frame direction that points up
        self.kind = kind                   # "axis_up", "flat_face", "rotation_axis"
        self.rotation = geometry.rotation_between(self.up, (0.0, 0.0, 1.0))
        self.translation = np.zeros(3)     # applied after rotation (STL frame)
        self.metrics = None
        self.score = None
        self.fits = True

    def as_dict(self):
        d = OrderedDict()
        d["id"] = self.id
        d["kind"] = self.kind
        d["up_in_model"] = [round(float(x), 4) for x in self.up]
        for key, value in self.metrics.items():
            d[key] = value
        d["S_o"] = round(float(self.score), 4)
        d["fits_build_volume"] = bool(self.fits)
        return d


def evaluate(mesh, orientation, support_angle=DEFAULT_SUPPORT_ANGLE, bed_delta=DEFAULT_BED_DELTA):
    R = orientation.rotation
    tris = mesh.triangles.dot(R.T)
    normals = mesh.normals.dot(R.T)
    z = tris[:, :, 2]
    z0 = float(z.min())
    on_bed = (z.max(axis=1) - z0) < bed_delta
    down = -normals[:, 2]
    support = (down > np.sin(np.radians(support_angle))) & ~on_bed
    base = (down > np.cos(np.radians(BASE_FACE_DEG))) & on_bed
    points = tris.reshape(-1, 3)
    lo, hi = points.min(axis=0), points.max(axis=0)
    sx, sy, h = (hi - lo)
    total = max(mesh.surface_area, 1e-12)
    a_support = float(mesh.areas[support].sum())
    a_base = float(mesh.areas[base].sum())
    footprint = max(float(sx * sy), 1e-12)

    r_support = a_support / total
    r_base = a_base / footprint
    r_slender = float(h) / max(min(sx, sy), 0.01)
    r_height = float(h) / max(sx, sy, h, 0.01)
    score = (WEIGHTS["support"] * r_support
             + WEIGHTS["slender"] * min(1.0, r_slender / 10.0)
             + WEIGHTS["base"] * (1.0 - min(1.0, r_base))
             + WEIGHTS["height"] * r_height)

    m = OrderedDict()
    m["R_support"] = round(r_support, 4)
    m["R_base"] = round(r_base, 4)
    m["R_slender"] = round(r_slender, 3)
    m["R_height"] = round(r_height, 4)
    m["support_area_mm2"] = round(a_support, 1)
    m["bed_contact_area_mm2"] = round(a_base, 1)
    m["size_xyz_mm"] = [round(float(sx), 2), round(float(sy), 2), round(float(h), 2)]
    orientation.metrics = m
    orientation.score = score
    orientation._bounds = (lo, hi)
    return orientation


def flat_face_directions(mesh, k=3, min_share=0.05, tolerance_deg=5.0):
    """Normals of the largest groups of parallel faces ("lay this face on the bed")."""
    steps = 1.0 / np.radians(tolerance_deg)
    keys = np.round(mesh.normals * steps).astype(np.int64)
    span = 2 * int(steps) + 3
    flat = ((keys[:, 0] + span) * (2 * span) + (keys[:, 1] + span)) * (2 * span) + (keys[:, 2] + span)
    uniq, inverse = np.unique(flat, return_inverse=True)
    group_area = np.bincount(inverse, weights=mesh.areas, minlength=len(uniq))
    order = np.argsort(group_area)[::-1]
    out = []
    for g in order[:4 * k + 8]:
        if group_area[g] < min_share * mesh.surface_area or len(out) >= k:
            break
        members = inverse == g
        n = (mesh.normals[members] * mesh.areas[members, None]).sum(axis=0)
        if np.linalg.norm(n) < 1e-9:
            continue
        out.append((geometry.unit(n), float(group_area[g])))
    return out


def candidates(mesh, k_faces=3):
    cands = [Orientation(oid, up, "axis_up") for oid, up in CANONICAL.items()]
    cos_tol = np.cos(np.radians(5.0))
    i = 0
    for normal, _area in flat_face_directions(mesh, k=k_faces + 6):
        up = -normal  # the face points down onto the bed
        if any(float(np.dot(up, c.up)) > cos_tol for c in cands):
            continue
        i += 1
        cands.append(Orientation("face%d" % i, up, "flat_face"))
        if i >= k_faces:
            break
    return cands


def _fits_classic(orientation, machine):
    if not machine:
        return True
    sx, sy, h = orientation.metrics["size_xyz_mm"]
    w, d, height = machine["width"], machine["depth"], machine["height_classic"]
    return h <= height and ((sx <= w and sy <= d) or (sx <= d and sy <= w))


def rank(mesh, machine=None, supports_forbidden=False, support_angle=DEFAULT_SUPPORT_ANGLE,
         k_faces=3):
    """Evaluate all candidates and choose o* subject to the constraints.

    Returns an OrderedDict with the table (sorted by S_o), the chosen id and
    flags used by the rules.
    """
    table = []
    for o in candidates(mesh, k_faces=k_faces):
        evaluate(mesh, o, support_angle=support_angle)
        o.fits = _fits_classic(o, machine)
        _center_on_bed(o)
        table.append(o)
    table.sort(key=lambda o: o.score)
    feasible = [o for o in table if o.fits] or table
    min_support = min(o.metrics["R_support"] for o in table)
    supports_unavoidable = min_support > SUPPORT_FREE_MAX
    if supports_forbidden:
        free = [o for o in feasible if o.metrics["R_support"] <= SUPPORT_FREE_MAX]
        best = free[0] if free else min(feasible, key=lambda o: (o.metrics["R_support"], o.score))
    else:
        best = feasible[0]
    result = OrderedDict()
    result["best"] = best
    result["table"] = table
    result["min_R_support"] = round(min_support, 4)
    result["supports_unavoidable"] = bool(supports_unavoidable)
    result["by_id"] = dict((o.id, o) for o in table)
    return result


def _center_on_bed(orientation):
    """Translation that puts the rotated model on the bed, centred at x = y = 0."""
    lo, hi = orientation._bounds
    orientation.translation = np.array([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])


def axis_orientations(mesh, axis_point, axis_direction, machine=None, support_angle=DEFAULT_SUPPORT_ANGLE):
    """Both 'rotation axis vertical' orientations for 5D modes, best first.

    The translation puts the rotation axis through the bed centre (x = y = 0),
    where the slicer cuts the central cylinder, and the lowest point on z = 0.
    """
    out = []
    for oid, sign in (("axis+", 1.0), ("axis-", -1.0)):
        o = Orientation(oid, sign * np.asarray(axis_direction, dtype=np.float64), "rotation_axis")
        evaluate(mesh, o, support_angle=support_angle)
        lo, _hi = o._bounds
        p = o.rotation.dot(np.asarray(axis_point, dtype=np.float64))
        o.translation = np.array([-p[0], -p[1], -lo[2]])
        radius = float(geometry.point_line_distance(mesh.vertices, axis_point, axis_direction).max())
        o.metrics["max_radius_mm"] = round(radius, 2)
        if machine:
            semi = min(machine["width"], machine["depth"]) / 2.0
            o.fits = o.metrics["size_xyz_mm"][2] <= machine["height_5d"] and radius <= semi
        out.append(o)
    out.sort(key=lambda o: o.score)
    return out


def radial_support_ratio(mesh, axis_point, axis_direction, base_radius, support_angle=DEFAULT_SUPPORT_ANGLE,
                         delta=DEFAULT_BED_DELTA):
    """Share of surface that needs support in cylindrical (5D) printing.

    Layers grow radially outwards from the rotation axis, so the overhang
    criterion uses the radial direction instead of +Z: a face needs support
    when it faces the axis more steeply than the support angle. Faces on the
    central cylinder (radius <= base_radius + delta) rest on the base.
    """
    d = geometry.unit(axis_direction)
    rel = mesh.centroids - np.asarray(axis_point, dtype=np.float64)
    radial = rel - rel.dot(d)[:, None] * d[None, :]
    r = np.linalg.norm(radial, axis=1)
    rhat = np.zeros_like(radial)
    ok = r > 1e-9
    rhat[ok] = radial[ok] / r[ok, None]
    inward = -(mesh.normals * rhat).sum(axis=1)
    tri_rel = mesh.triangles - np.asarray(axis_point, dtype=np.float64)
    tri_r = np.linalg.norm(tri_rel - (tri_rel.dot(d))[..., None] * d, axis=2)
    on_base = tri_r.max(axis=1) <= base_radius + delta
    support = (inward > np.sin(np.radians(support_angle))) & ~on_base
    return float(mesh.areas[support].sum()) / max(mesh.surface_area, 1e-12)


def transform_matrix(orientation):
    """4x4 homogeneous transform (STL frame): rotate, then translate."""
    m = np.eye(4)
    m[:3, :3] = orientation.rotation
    m[:3, 3] = orientation.translation
    return m
