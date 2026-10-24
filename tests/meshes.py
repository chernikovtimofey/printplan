"""Synthetic triangle meshes (N, 3, 3) with outward normals, for tests."""
import numpy as np


def box(size=(10.0, 20.0, 30.0), origin=(0.0, 0.0, 0.0)):
    sx, sy, sz = size
    ox, oy, oz = origin
    v = np.array([[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0],
                  [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]], dtype=float) + [ox, oy, oz]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return v[np.array(faces)]


def extrude(polygon, height, kernel=None):
    """Extrude a star-shaped polygon (counter-clockwise, seen from +Z) along +Z.

    Caps are fans from `kernel` (a point from which every vertex is visible).
    """
    poly = np.asarray(polygon, dtype=float)
    n = len(poly)
    c = np.mean(poly, axis=0) if kernel is None else np.asarray(kernel, dtype=float)
    tris = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        tris.append([[c[0], c[1], 0], [b[0], b[1], 0], [a[0], a[1], 0]])            # bottom (normal -Z)
        tris.append([[c[0], c[1], height], [a[0], a[1], height], [b[0], b[1], height]])  # top (normal +Z)
        tris.append([[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], height]])        # side
        tris.append([[a[0], a[1], 0], [b[0], b[1], height], [a[0], a[1], height]])
    return np.array(tris)


def circle(radius, segments=64, phase=0.0):
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False) + phase
    return np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)


def cylinder(radius=10.0, height=30.0, segments=64):
    return extrude(circle(radius, segments), height, kernel=(0.0, 0.0))


def tube(r_out=20.0, r_in=8.0, height=15.0, segments=64):
    outer = circle(r_out, segments)
    inner = circle(r_in, segments)
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        o0, o1, i0, i1 = outer[i], outer[j], inner[i], inner[j]
        # outer wall
        tris.append([[o0[0], o0[1], 0], [o1[0], o1[1], 0], [o1[0], o1[1], height]])
        tris.append([[o0[0], o0[1], 0], [o1[0], o1[1], height], [o0[0], o0[1], height]])
        # inner wall (normal towards the axis)
        tris.append([[i0[0], i0[1], 0], [i1[0], i1[1], height], [i1[0], i1[1], 0]])
        tris.append([[i0[0], i0[1], 0], [i0[0], i0[1], height], [i1[0], i1[1], height]])
        # top ring
        tris.append([[o0[0], o0[1], height], [o1[0], o1[1], height], [i1[0], i1[1], height]])
        tris.append([[o0[0], o0[1], height], [i1[0], i1[1], height], [i0[0], i0[1], height]])
        # bottom ring
        tris.append([[o0[0], o0[1], 0], [i1[0], i1[1], 0], [o1[0], o1[1], 0]])
        tris.append([[o0[0], o0[1], 0], [i0[0], i0[1], 0], [i1[0], i1[1], 0]])
    return np.array(tris)


def cone(radius=15.0, height=30.0, segments=64):
    ring = circle(radius, segments)
    apex = [0.0, 0.0, height]
    tris = []
    for i in range(segments):
        a, b = ring[i], ring[(i + 1) % segments]
        tris.append([[0.0, 0.0, 0.0], [b[0], b[1], 0], [a[0], a[1], 0]])
        tris.append([[a[0], a[1], 0], [b[0], b[1], 0], apex])
    return np.array(tris)


def gear(teeth=20, r_root=14.0, r_tip=16.0, height=8.0):
    """Gear-like star prism: alternating root and tip radii."""
    pts = []
    for i in range(teeth):
        base = 2 * np.pi * i / teeth
        step = 2 * np.pi / teeth
        for frac, r in ((0.0, r_root), (0.25, r_tip), (0.5, r_tip), (0.75, r_root)):
            a = base + frac * step
            pts.append([r * np.cos(a), r * np.sin(a)])
    return extrude(pts, height, kernel=(0.0, 0.0))


def hex_prism(radius=10.0, height=20.0):
    return extrude(circle(radius, 6), height, kernel=(0.0, 0.0))


def l_bracket(length=40.0, height=30.0, thickness=5.0, width=20.0):
    poly = [(0, 0), (length, 0), (length, thickness), (thickness, thickness), (thickness, height), (0, height)]
    return extrude(poly, width, kernel=(thickness / 2.0, thickness / 2.0))


def t_overhang(stem=10.0, stem_h=30.0, bar=40.0, bar_h=10.0, depth=20.0):
    """T profile in the XY plane extruded along Z; standing on the stem (+y up) it overhangs."""
    s, b = stem / 2.0, bar / 2.0
    poly = [(-s, 0), (s, 0), (s, stem_h), (b, stem_h), (b, stem_h + bar_h), (-b, stem_h + bar_h),
            (-b, stem_h), (-s, stem_h)]
    return extrude(poly, depth, kernel=(0.0, stem_h + bar_h / 2.0))


def sphere(radius=10.0, rings=24, segments=48):
    tris = []
    for i in range(rings):
        t0, t1 = np.pi * i / rings, np.pi * (i + 1) / rings
        for j in range(segments):
            p0, p1 = 2 * np.pi * j / segments, 2 * np.pi * (j + 1) / segments

            def pt(t, p):
                return [radius * np.sin(t) * np.cos(p), radius * np.sin(t) * np.sin(p), radius * np.cos(t)]
            a, b, c, d = pt(t0, p0), pt(t1, p0), pt(t1, p1), pt(t0, p1)
            if i > 0:
                tris.append([a, b, d])
            if i < rings - 1:
                tris.append([b, c, d])
    return np.array(tris)


def translate(tris, offset):
    return np.asarray(tris) + np.asarray(offset, dtype=float)


def rotate(tris, matrix):
    return np.asarray(tris).dot(np.asarray(matrix).T)
