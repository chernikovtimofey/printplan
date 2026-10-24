"""Stage 2: mesh analysis.

`analyze(triangles)` takes a raw (N, 3, 3) triangle array in millimetres (Z up,
as STL is interpreted by the slicer) and returns a `MeshModel` with the arrays
later stages need plus the evidence dicts F_mesh and Q_topo.
"""
from collections import OrderedDict

import numpy as np

from . import geometry

UNITS_SMALL_MM = 1.0      # max extent below this: probably metres or inches
UNITS_LARGE_MM = 3000.0   # max extent above this: probably micrometres / wrong units


class MeshModel(object):
    """Triangle soup plus derived per-face data and the analysis results."""

    def __init__(self, triangles):
        tris = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        if len(tris) == 0:
            raise ValueError("The mesh has no triangles")
        self.triangles = tris
        self.normals, self.areas = geometry.triangle_normals_areas(tris)
        self.centroids = tris.mean(axis=1)
        points = tris.reshape(-1, 3)
        self.bbox_min = points.min(axis=0)
        self.bbox_max = points.max(axis=0)
        self.extents = self.bbox_max - self.bbox_min
        self.diagonal = float(np.linalg.norm(self.extents))
        self.vertices = None
        self.faces = None
        self.volume = None
        self.center_of_mass = None
        self.pca_axes = None          # rows: principal axes, largest eigenvalue first
        self.pca_eigenvalues = None
        self.F_mesh = None
        self.Q_topo = None

    @property
    def surface_area(self):
        return float(self.areas.sum())


def analyze(triangles, machine=None):
    """Run all stage-2 computations; `machine` is the catalog 'machine' dict (optional)."""
    mesh = MeshModel(triangles)
    _weld(mesh)
    _volume_and_center(mesh)
    _pca(mesh)
    topo = _topology(mesh)
    mesh.Q_topo = topo
    mesh.F_mesh = _mesh_report(mesh, machine)
    return mesh


# ----------------------------------------------------------------------------- 2.1 welding

def _weld(mesh):
    eps = max(1e-6 * mesh.diagonal, 1e-4)
    points = mesh.triangles.reshape(-1, 3)
    keys = np.round((points - mesh.bbox_min) / eps).astype(np.int64)
    # Encode the three integer coordinates into one int64 for a fast 1-D unique.
    span = keys.max(axis=0) + 1
    if np.prod(span.astype(np.float64)) < 2.0 ** 62:
        flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
        _, first, inverse = np.unique(flat, return_index=True, return_inverse=True)
    else:
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    mesh.vertices = points[first]
    mesh.faces = inverse.reshape(-1, 3).astype(np.int64)
    mesh.weld_tolerance = eps


# ----------------------------------------------------------------------------- 2.4 volume, centre of mass

def _volume_and_center(mesh):
    origin = (mesh.bbox_min + mesh.bbox_max) / 2.0
    t = mesh.triangles - origin
    signed = (t[:, 0] * np.cross(t[:, 1], t[:, 2])).sum(axis=1) / 6.0
    volume = float(signed.sum())
    mesh.volume = volume
    if abs(volume) > 1e-9:
        mesh.center_of_mass = origin + (signed[:, None] * t.sum(axis=1) / 4.0).sum(axis=0) / volume
    else:
        mesh.center_of_mass = origin + (mesh.areas[:, None] * (mesh.centroids - origin)).sum(axis=0) \
            / max(mesh.surface_area, 1e-12)


# ----------------------------------------------------------------------------- 2.5 principal axes

def _pca(mesh):
    w = mesh.areas / max(mesh.surface_area, 1e-12)
    mean = (w[:, None] * mesh.centroids).sum(axis=0)
    rel = mesh.centroids - mean
    cov = (w[:, None] * rel).T.dot(rel)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    mesh.pca_eigenvalues = values[order]
    mesh.pca_axes = vectors[:, order].T
    mesh.surface_centroid = mean


# ----------------------------------------------------------------------------- 2.6 topology

def _topology(mesh):
    faces = mesh.faces
    nv = len(mesh.vertices)
    collapsed = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    tiny_area = (mesh.diagonal * 1e-7) ** 2
    degenerate = collapsed | (mesh.areas <= tiny_area)
    good = np.nonzero(~collapsed)[0]
    f = faces[good]
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    face_of_edge = np.concatenate([good, good, good])
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    key = lo * nv + hi
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    direction = np.where(a < b, 1, -1)
    direction_sum = np.bincount(inverse, weights=direction, minlength=len(counts))
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    inconsistent = int(((counts == 2) & (direction_sum != 0)).sum())

    # Shells: faces connected through shared edges.
    order = np.argsort(inverse, kind="mergesort")
    same = inverse[order][1:] == inverse[order][:-1]
    fa = face_of_edge[order][:-1][same]
    fb = face_of_edge[order][1:][same]
    shell_count, labels = geometry.connected_components(len(faces), fa, fb)
    # Degenerate faces are not bodies; count shells over proper faces only.
    if degenerate.any():
        proper = np.nonzero(~degenerate)[0]
        shell_count = len(np.unique(labels[proper])) if len(proper) else 0
    mesh.shell_labels = labels

    watertight = boundary == 0 and nonmanifold == 0 and inconsistent == 0
    topo = OrderedDict()
    topo["n_triangles"] = int(len(faces))
    topo["n_vertices"] = int(nv)
    topo["n_edges"] = int(len(counts))
    topo["boundary_edges"] = boundary
    topo["nonmanifold_edges"] = nonmanifold
    topo["inconsistent_edges"] = inconsistent
    topo["degenerate_triangles"] = int(degenerate.sum())
    topo["shells"] = int(shell_count)
    topo["watertight"] = bool(watertight)
    topo["inverted"] = bool(mesh.volume is not None and mesh.volume < 0)
    return topo


# ----------------------------------------------------------------------------- report

def _round(values, digits=3):
    return [round(float(v), digits) for v in values]


def _mesh_report(mesh, machine):
    ext = mesh.extents
    rep = OrderedDict()
    rep["extents_mm"] = _round(ext, 2)
    rep["bbox_min_mm"] = _round(mesh.bbox_min, 2)
    rep["bbox_max_mm"] = _round(mesh.bbox_max, 2)
    rep["diagonal_mm"] = round(mesh.diagonal, 2)
    rep["surface_area_mm2"] = round(mesh.surface_area, 1)
    rep["volume_mm3"] = round(abs(mesh.volume), 1)
    rep["volume_cm3"] = round(abs(mesh.volume) / 1000.0, 2)
    rep["volume_exact"] = bool(mesh.Q_topo["watertight"])
    rep["center_of_mass_mm"] = _round(mesh.center_of_mass, 2)
    rep["pca_axes"] = [_round(a, 4) for a in mesh.pca_axes]
    ev = mesh.pca_eigenvalues
    rep["pca_eigenvalues"] = _round(ev, 3)
    largest = float(ext.max())
    units = None
    if largest < UNITS_SMALL_MM:
        units = OrderedDict([("problem", "too_small"), ("max_extent_mm", round(largest, 4)),
                             ("suggested_scales", [1000.0, 25.4])])
    elif largest > UNITS_LARGE_MM:
        units = OrderedDict([("problem", "too_large"), ("max_extent_mm", round(largest, 1)),
                             ("suggested_scales", [0.001, 1.0 / 25.4])])
    rep["units_warning"] = units
    if machine:
        dims = sorted([machine["width"], machine["depth"], machine["height_classic"]])
        rep["fits_build_volume_some_orientation"] = bool(np.all(np.sort(ext) <= np.array(dims)))
    return rep
