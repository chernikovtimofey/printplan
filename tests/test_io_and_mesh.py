import io
import os
import zipfile

import numpy as np
import pytest

import meshes
from printplan import io_utils, mesh as M


def test_binary_stl_roundtrip(tmpdir):
    tris = meshes.box()
    path = str(tmpdir.join("box.stl"))
    io_utils.write_stl(path, tris)
    back = io_utils.read_stl(path)
    assert back.shape == (12, 3, 3)
    assert np.allclose(back, tris)


def test_ascii_stl(tmpdir):
    tris = meshes.box((1, 2, 3))
    lines = ["solid test"]
    for t in tris:
        lines.append("facet normal 0 0 0\n outer loop")
        for v in t:
            lines.append("  vertex %r %r %r" % tuple(v))
        lines.append(" endloop\nendfacet")
    lines.append("endsolid test")
    path = str(tmpdir.join("ascii.stl"))
    with open(path, "w") as f:
        f.write("\n".join(lines))
    assert np.allclose(io_utils.read_stl(path), tris)


def test_invalid_stl(tmpdir):
    path = str(tmpdir.join("bad.stl"))
    with open(path, "wb") as f:
        f.write(b"\x00" * 100)
    with pytest.raises(ValueError):
        io_utils.read_stl(path)


def test_docx_reader(tmpdir):
    xml = ('<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           '<w:body><w:p><w:r><w:t>Кронштейн под нагрузку,</w:t></w:r><w:r><w:t xml:space="preserve"> PETG</w:t></w:r></w:p>'
           '<w:p><w:r><w:t>без поддержек &amp; быстро</w:t></w:r></w:p></w:body></w:document>')
    path = str(tmpdir.join("task.docx"))
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", xml.encode("utf-8"))
    assert io_utils.read_task(path) == "Кронштейн под нагрузку, PETG\nбез поддержек & быстро"


def test_read_task_passes_text_through():
    assert io_utils.read_task("просто текст") == "просто текст"


def test_box_metrics():
    m = M.analyze(meshes.box((10, 20, 30)))
    assert m.volume == pytest.approx(6000.0)
    assert m.surface_area == pytest.approx(2 * (200 + 300 + 600))
    assert np.allclose(m.center_of_mass, [5, 10, 15])
    q = m.Q_topo
    assert q["watertight"] and q["shells"] == 1 and q["n_vertices"] == 8
    assert q["boundary_edges"] == 0 and q["nonmanifold_edges"] == 0 and q["inconsistent_edges"] == 0
    assert m.F_mesh["extents_mm"] == [10.0, 20.0, 30.0]
    assert m.F_mesh["units_warning"] is None


def test_cylinder_volume_close_to_analytic():
    m = M.analyze(meshes.cylinder(10, 30, segments=256))
    assert m.volume == pytest.approx(np.pi * 100 * 30, rel=1e-3)


def test_open_box_has_boundary_edges():
    m = M.analyze(meshes.box()[:-2])
    assert m.Q_topo["boundary_edges"] == 4
    assert not m.Q_topo["watertight"]
    assert m.F_mesh["volume_exact"] is False


def test_flipped_face_is_inconsistent():
    tris = meshes.box().copy()
    tris[0] = tris[0][::-1]
    m = M.analyze(tris)
    assert m.Q_topo["inconsistent_edges"] > 0
    assert not m.Q_topo["watertight"]


def test_nonmanifold_edge_between_touching_boxes():
    a = meshes.box((10, 10, 10))
    b = meshes.box((10, 10, 10), origin=(10, 10, 0))  # shares the vertical edge x=10, y=10
    m = M.analyze(np.concatenate([a, b]))
    assert m.Q_topo["nonmanifold_edges"] >= 1


def test_shells_and_degenerate():
    a = meshes.box((5, 5, 5))
    b = meshes.box((5, 5, 5), origin=(20, 0, 0))
    degenerate = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]]], dtype=float)
    m = M.analyze(np.concatenate([a, b, degenerate]))
    assert m.Q_topo["shells"] == 2
    assert m.Q_topo["degenerate_triangles"] == 1


def test_inverted_mesh():
    m = M.analyze(meshes.box()[:, ::-1])
    assert m.Q_topo["inverted"]
    assert m.volume < 0


def test_units_warning_for_tiny_model(machine):
    m = M.analyze(meshes.box((0.05, 0.08, 0.02)), machine)
    assert m.F_mesh["units_warning"]["problem"] == "too_small"
    assert 1000.0 in m.F_mesh["units_warning"]["suggested_scales"]


def test_fits_build_volume(machine):
    assert M.analyze(meshes.box((100, 100, 100)), machine).F_mesh["fits_build_volume_some_orientation"]
    assert not M.analyze(meshes.box((400, 100, 100)), machine).F_mesh["fits_build_volume_some_orientation"]


def test_pure_numpy_components(monkeypatch):
    from printplan import geometry
    monkeypatch.setattr(geometry, "_csgraph", None)
    monkeypatch.setattr(geometry, "_scipy_checked", True)
    count, labels = geometry.connected_components(6, [0, 1, 3], [1, 2, 4])
    assert count == 3
    assert labels[0] == labels[1] == labels[2] and labels[3] == labels[4] and labels[5] not in (labels[0], labels[3])
