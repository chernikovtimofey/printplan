import numpy as np
import pytest

import meshes
from printplan import geometry, mesh as M, orientation as O, symmetry as S


def analyzed(tris):
    return M.analyze(tris)


# ----------------------------------------------------------------------------- geometry helpers

def test_rotation_between():
    for a in ([1, 0, 0], [0, -1, 0], [0, 0, -1], [0.3, 0.4, 0.5]):
        R = geometry.rotation_between(a, [0, 0, 1])
        assert np.allclose(R.dot(geometry.unit(a)), [0, 0, 1])
        assert np.allclose(R.T.dot(R), np.eye(3))
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_ray_hits_and_nn():
    m = analyzed(meshes.box((10, 10, 10)))
    hits = geometry.ray_triangle_hits([3, 6, 5], [0, 0, 1], m.triangles)
    assert sorted(np.round(hits, 6)) == [-5.0, 5.0]
    nn = geometry.NearestNeighbors(np.array([[0.0, 0, 0], [10, 0, 0]]))
    assert np.allclose(nn.distances(np.array([[1.0, 0, 0], [9, 1, 0]])), [1.0, np.sqrt(2)])


def test_nn_brute_force_matches(monkeypatch):
    ref = np.random.RandomState(0).rand(500, 3)
    query = np.random.RandomState(1).rand(50, 3)
    fast = geometry.NearestNeighbors(ref).distances(query)
    monkeypatch.setattr(geometry, "_cKDTree", None)
    monkeypatch.setattr(geometry, "_scipy_checked", True)
    slow = geometry.NearestNeighbors(ref).distances(query)
    assert np.allclose(fast, slow)


# ----------------------------------------------------------------------------- stage 3

def test_box_orientation_prefers_lying_flat():
    m = analyzed(meshes.box((50, 40, 5)))
    r = O.rank(m)
    assert r["best"].id in ("+z", "-z")
    best = r["best"].metrics
    assert best["R_support"] == 0.0
    assert best["R_base"] == pytest.approx(1.0, abs=1e-6)
    assert not r["supports_unavoidable"]


def test_t_overhang_needs_support_when_standing():
    m = analyzed(meshes.t_overhang())
    r = O.rank(m)
    standing = r["by_id"]["+y"].metrics
    flat = r["by_id"]["+z"].metrics
    assert standing["R_support"] > 0.05      # arms of the T overhang
    assert flat["R_support"] == 0.0          # extrusion lying on its side face
    assert r["best"].id in ("+z", "-z", "-y")


def test_supports_forbidden_picks_support_free_orientation():
    m = analyzed(meshes.t_overhang())
    r = O.rank(m, supports_forbidden=True)
    assert r["best"].metrics["R_support"] <= O.SUPPORT_FREE_MAX


def test_sphere_has_point_contact():
    r = O.rank(analyzed(meshes.sphere()))
    assert r["best"].metrics["R_base"] < 0.05
    assert r["supports_unavoidable"]


def test_flat_face_candidate_for_tilted_box():
    R = geometry.axis_angle_matrix([1, 1, 0], np.radians(30))
    m = analyzed(meshes.rotate(meshes.box((60, 40, 4)), R))
    r = O.rank(m)
    assert r["best"].kind == "flat_face"
    assert r["best"].metrics["R_base"] > 0.9


def test_radial_support_ratio_tube_vs_ring():
    tube = analyzed(meshes.tube(20, 8, 15))
    # Base radius = bore radius: the inner wall rests on the rod.
    assert O.radial_support_ratio(tube, [0, 0, 0], [0, 0, 1], 8.0) == pytest.approx(0.0)
    # A thin rod: the bore wall faces the axis and needs support in 5D.
    assert O.radial_support_ratio(tube, [0, 0, 0], [0, 0, 1], 3.0) > 0.1


def test_axis_orientation_puts_axis_through_center():
    m = analyzed(meshes.translate(meshes.cylinder(10, 30), (50, 20, 5)))
    opts = O.axis_orientations(m, [50, 20, 0], [0, 0, 1])
    o = opts[0]
    pts = m.vertices.dot(o.rotation.T) + o.translation
    assert pts[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    center = (pts[:, :2].max(axis=0) + pts[:, :2].min(axis=0)) / 2.0
    assert np.allclose(center, [0, 0], atol=1e-6)


# ----------------------------------------------------------------------------- rotational features

def _selected(tris):
    m = analyzed(tris)
    return m, S.analyze(m)


def test_cylinder_and_cone_are_bodies_of_revolution():
    for tris in (meshes.cylinder(), meshes.cone()):
        _, ra = _selected(tris)
        assert ra.selected is not None and ra.selected.body_of_revolution
        assert ra.selected.label in ("Z", "PCA1", "PCA3")
        assert abs(ra.selected.direction[2]) > 0.99


def test_tube_bore_detected():
    _, ra = _selected(meshes.tube(20, 8, 15))
    axis = ra.selected
    assert axis.body_of_revolution
    assert axis.bore["diameter_mm"] == pytest.approx(16.0, abs=0.1)


def test_solid_cylinder_has_no_bore():
    _, ra = _selected(meshes.cylinder())
    assert ra.selected.bore is None


def test_hex_prism_is_sixfold_not_revolution():
    _, ra = _selected(meshes.hex_prism())
    axis = ra.selected
    assert not axis.body_of_revolution
    assert axis.cyclic_order == 6


def test_gear_cyclic_order():
    _, ra = _selected(meshes.gear(teeth=12))
    assert ra.selected.cyclic_order == 12
    assert ra.selected.prismatic_fraction > 0.95


def test_box_and_l_bracket_not_revolution():
    for tris in (meshes.box((10, 20, 30)), meshes.l_bracket()):
        _, ra = _selected(tris)
        assert ra.selected is None or not ra.selected.body_of_revolution


def test_symmetry_without_scipy(monkeypatch):
    monkeypatch.setattr(geometry, "_cKDTree", None)
    monkeypatch.setattr(geometry, "_scipy_checked", True)
    _, ra = _selected(meshes.tube(20, 8, 15))
    assert ra.selected.body_of_revolution
    _, ra = _selected(meshes.hex_prism())
    assert ra.selected.cyclic_order == 6
