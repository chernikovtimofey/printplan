"""Interface between the tools and a slicer, plus shared helpers.

Two implementations exist: `slicer_live.LiveSlicer` drives STE Slicer through
its Python API, `slicer_fake.FakeSlicer` keeps everything in memory for tests
and the development server. Tools only use the methods below.

Coordinates: tools and the pipeline use printer coordinates — millimetres,
Z up, origin in the centre of the bed (as in STL files). The slicer scene uses
Y up: scene (x, y, z) = printer (x, z, -y).
"""
from collections import OrderedDict

import numpy as np

from ..printplan.slicer_context import CatalogSlicerContext


class SlicerError(Exception):
    """An action the slicer could not perform (shown to the user as is)."""


def scene_to_printer(points):
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def printer_to_scene(points):
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def bbox_info(min_xyz, max_xyz):
    lo = [round(float(v), 2) for v in min_xyz]
    hi = [round(float(v), 2) for v in max_xyz]
    out = OrderedDict()
    out["size_mm"] = [round(h - l, 2) for l, h in zip(lo, hi)]
    out["center_mm"] = [round((l + h) / 2.0, 2) for l, h in zip(lo, hi)]
    out["min_mm"] = lo
    out["max_mm"] = hi
    return out


class LiveCatalogContext(CatalogSlicerContext):
    """Catalog limits, but quality profiles and materials as the slicer reports them."""

    def __init__(self, catalog=None, quality_types=None, material_ids=None, validator=None):
        super(LiveCatalogContext, self).__init__(catalog)
        self._quality_types = quality_types
        self._material_ids = material_ids
        self._validator = validator

    def quality_profiles(self):
        profiles = super(LiveCatalogContext, self).quality_profiles()
        if not self._quality_types:
            return profiles
        known = [p for p in profiles if p["quality_type"] in self._quality_types]
        return known or profiles

    def materials(self):
        materials = super(LiveCatalogContext, self).materials()
        if not self._material_ids:
            return materials
        known = [m for m in materials if m["id"] in self._material_ids]
        return known or materials


class SlicerAPI(object):
    """Methods every slicer adapter implements. All of them are thread-safe."""

    name = "slicer"

    # ------------------------------------------------------------------ state, profiles, settings
    def state(self):
        """Machine, printing mode, quality type, material, objects, slicing state."""
        raise NotImplementedError

    def list_quality_profiles(self):
        raise NotImplementedError

    def list_materials(self):
        raise NotImplementedError

    def get_settings(self, keys):
        """{key: {"value": ..., "stack": "global"|"extruder"}}"""
        raise NotImplementedError

    def set_settings(self, values):
        """Write user values; returns {key: {"value", "stack", "validation"}}."""
        raise NotImplementedError

    def reset_settings(self):
        """Discard all user changes (back to the profile values)."""
        raise NotImplementedError

    def set_quality(self, quality_type):
        raise NotImplementedError

    def set_material(self, material_id, extruder=0):
        raise NotImplementedError

    def set_printing_mode(self, mode):
        raise NotImplementedError

    def save_profile(self, name):
        """Store the current user changes as a custom profile."""
        raise NotImplementedError

    def slicer_context(self):
        """SlicerContext for the pipeline (limits + what the slicer offers)."""
        return CatalogSlicerContext()

    # ------------------------------------------------------------------ scene
    def load_model(self, path):
        """Load a model file; returns the list of new objects."""
        raise NotImplementedError

    def list_objects(self):
        raise NotImplementedError

    def get_triangles(self, object_id):
        """(N, 3, 3) triangles of the object in printer coordinates (world placement)."""
        raise NotImplementedError

    def transform_object(self, object_id, rotation=None, translation=None):
        """Apply v' = R v + t to the object's world geometry (printer coordinates)."""
        raise NotImplementedError

    def move_object(self, object_id, x=None, y=None, z=None):
        """Move the bounding-box centre to (x, y) and the bottom to z (printer coordinates)."""
        raise NotImplementedError

    def scale_object(self, object_id, factors):
        raise NotImplementedError

    def arrange_all(self):
        raise NotImplementedError

    def delete_object(self, object_id):
        raise NotImplementedError

    def clear_scene(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ slicing and files
    def slice(self, progress=None, timeout=900.0):
        """Slice and wait; returns {"state", "print_time_s", "material_length_m", ...}."""
        raise NotImplementedError

    def slice_status(self):
        raise NotImplementedError

    def export_gcode(self, path):
        raise NotImplementedError

    def save_project(self, path):
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers shared by adapters
    def rotate_object(self, object_id, axis, angle_deg):
        """Rotate around the bounding-box centre about a printer axis, then put back on the bed."""
        axis = str(axis).lower()
        if axis not in ("x", "y", "z"):
            raise SlicerError("Ось поворота должна быть x, y или z")
        tris = self.get_triangles(object_id)
        pts = tris.reshape(-1, 3)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        a = np.radians(float(angle_deg))
        c, s = np.cos(a), np.sin(a)
        rot = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]],
               "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
               "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
        rot = np.array(rot, dtype=np.float64)
        moved = (pts - center).dot(rot.T) + center
        translation = center - rot.dot(center)
        translation[2] -= moved[:, 2].min()  # back onto the bed
        return self.transform_object(object_id, rot, translation)
