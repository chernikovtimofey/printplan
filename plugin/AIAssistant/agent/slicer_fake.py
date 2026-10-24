"""In-memory slicer for tests and the development server.

It mirrors the behaviour of STE Slicer that the tools rely on: the machine
starts in the catalog's default printing mode, settings live in global and
extruder user layers, objects are placed on the bed, slicing produces a print
time and a G-code text. G-code and project files it writes are marked as fake.
"""
import io
import json
import os
import threading
import time
import zipfile
from collections import OrderedDict

import numpy as np

from ..printplan import io_utils, mesh as mesh_mod
from ..printplan.slicer_context import CatalogSlicerContext
from .slicer_base import LiveCatalogContext, SlicerAPI, SlicerError, bbox_info


class FakeSlicer(SlicerAPI):
    name = "fake"

    def __init__(self, catalog=None, slice_delay=0.0):
        self._ctx = CatalogSlicerContext(catalog)
        self._catalog = self._ctx.catalog
        self._lock = threading.RLock()
        self._objects = OrderedDict()
        self._next_id = 1
        self._global = OrderedDict()
        self._extruder = OrderedDict()
        self.quality_type = "normal"
        self.material_id = "proto_pla"
        self._slice_state = "not_started"
        self._slice_result = None
        self._slice_delay = slice_delay
        self.saved_profiles = []

    # ------------------------------------------------------------------ helpers
    def _spec(self, key):
        return self._ctx.setting(key)

    def _value(self, key):
        if key in self._global:
            return self._global[key]
        if key in self._extruder:
            return self._extruder[key]
        if key == "layer_height":
            return self._ctx.quality_layer_height(self.quality_type)
        spec = self._spec(key)
        return spec.get("default") if spec else None

    def _stack_for(self, key):
        spec = self._spec(key) or {}
        return "extruder" if spec.get("settable_per_extruder", True) else "global"

    def _invalidate(self):
        self._slice_state = "not_started"
        self._slice_result = None

    def _object(self, object_id):
        try:
            return self._objects[int(object_id)]
        except (KeyError, ValueError, TypeError):
            raise SlicerError("Нет объекта с id %s" % object_id)

    def _object_info(self, oid, obj):
        pts = obj["triangles"].reshape(-1, 3)
        info = OrderedDict([("id", oid), ("name", obj["name"])])
        info.update(bbox_info(pts.min(axis=0), pts.max(axis=0)))
        info["triangles"] = int(len(obj["triangles"]))
        return info

    # ------------------------------------------------------------------ state, profiles, settings
    def state(self):
        with self._lock:
            out = OrderedDict()
            out["slicer"] = "fake"
            out["machine"] = self._catalog["machine"]["name"]
            out["printing_mode"] = self._value("printing_mode")
            out["quality_type"] = self.quality_type
            out["material"] = self.material_id
            out["user_settings"] = OrderedDict(list(self._global.items()) + list(self._extruder.items()))
            out["objects"] = [self._object_info(oid, o) for oid, o in self._objects.items()]
            out["slice_state"] = self._slice_state
            return out

    def list_quality_profiles(self):
        return [OrderedDict([("quality_type", q["quality_type"]), ("name", q["name"]),
                             ("layer_height", q["layer_height"]), ("available", True)])
                for q in self._ctx.quality_profiles()]

    def list_materials(self):
        return [OrderedDict([("id", m["id"]), ("name", m["label"]), ("brand", m.get("brand"))])
                for m in self._ctx.materials()]

    def get_settings(self, keys):
        with self._lock:
            out = OrderedDict()
            for key in keys:
                if self._spec(key) is None:
                    out[key] = OrderedDict([("error", "неизвестный ключ")])
                    continue
                out[key] = OrderedDict([("value", self._value(key)), ("stack", self._stack_for(key))])
            return out

    def set_settings(self, values):
        with self._lock:
            out = OrderedDict()
            for key, value in values.items():
                if self._spec(key) is None:
                    raise SlicerError("Неизвестный ключ настройки: %s" % key)
                stack = self._stack_for(key)
                (self._extruder if stack == "extruder" else self._global)[key] = value
                out[key] = OrderedDict([("value", value), ("stack", stack), ("validation", "Valid")])
            self._invalidate()
            return out

    def reset_settings(self):
        with self._lock:
            self._global.clear()
            self._extruder.clear()
            self._invalidate()
            return True

    def set_quality(self, quality_type):
        if quality_type not in self._ctx.quality_types():
            raise SlicerError("Нет профиля качества %s" % quality_type)
        with self._lock:
            self.quality_type = quality_type
            self._invalidate()
        return quality_type

    def set_material(self, material_id, extruder=0):
        if material_id not in self._ctx.material_ids():
            raise SlicerError("Нет материала %s" % material_id)
        with self._lock:
            self.material_id = material_id
            self._invalidate()
        return material_id

    def set_printing_mode(self, mode):
        if mode not in ("classic", "cylindrical", "cylindrical_full"):
            raise SlicerError("Неизвестный режим печати %s" % mode)
        with self._lock:
            self._global["printing_mode"] = mode
            self._invalidate()
        return mode

    def save_profile(self, name):
        with self._lock:
            self.saved_profiles.append(OrderedDict([("name", name), ("quality_type", self.quality_type),
                                                    ("settings", dict(self._global, **self._extruder))]))
        return name

    def slicer_context(self):
        return LiveCatalogContext(self._catalog)

    # ------------------------------------------------------------------ scene
    def load_model(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext != ".stl":
            raise SlicerError("Тестовый слайсер читает только STL")
        tris = io_utils.read_stl(path)
        pts = tris.reshape(-1, 3)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        offset = np.array([-center[0], -center[1], -pts[:, 2].min()])
        with self._lock:
            oid = self._next_id
            self._next_id += 1
            self._objects[oid] = {"name": os.path.basename(path), "triangles": tris + offset}
            self._invalidate()
            return [self._object_info(oid, self._objects[oid])]

    def add_triangles(self, name, triangles):
        """Test helper: put a mesh on the bed like load_model does."""
        tris = np.asarray(triangles, dtype=np.float64)
        pts = tris.reshape(-1, 3)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        with self._lock:
            oid = self._next_id
            self._next_id += 1
            self._objects[oid] = {"name": name,
                                  "triangles": tris + np.array([-center[0], -center[1], -pts[:, 2].min()])}
            self._invalidate()
            return self._object_info(oid, self._objects[oid])

    def list_objects(self):
        with self._lock:
            return [self._object_info(oid, o) for oid, o in self._objects.items()]

    def get_triangles(self, object_id):
        with self._lock:
            return self._object(object_id)["triangles"].copy()

    def transform_object(self, object_id, rotation=None, translation=None):
        with self._lock:
            obj = self._object(object_id)
            tris = obj["triangles"]
            if rotation is not None:
                tris = tris.dot(np.asarray(rotation, dtype=np.float64).T)
            if translation is not None:
                tris = tris + np.asarray(translation, dtype=np.float64)
            obj["triangles"] = tris
            self._invalidate()
            return self._object_info(int(object_id), obj)

    def move_object(self, object_id, x=None, y=None, z=None):
        with self._lock:
            obj = self._object(object_id)
            pts = obj["triangles"].reshape(-1, 3)
            center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
            shift = np.zeros(3)
            if x is not None:
                shift[0] = float(x) - center[0]
            if y is not None:
                shift[1] = float(y) - center[1]
            if z is not None:
                shift[2] = float(z) - pts[:, 2].min()
            obj["triangles"] = obj["triangles"] + shift
            self._invalidate()
            return self._object_info(int(object_id), obj)

    def scale_object(self, object_id, factors):
        f = np.asarray(factors, dtype=np.float64).reshape(-1)
        if f.size == 1:
            f = np.repeat(f, 3)
        with self._lock:
            obj = self._object(object_id)
            pts = obj["triangles"].reshape(-1, 3)
            center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
            tris = (obj["triangles"] - center) * f + center
            tris[..., 2] -= tris[..., 2].min()
            obj["triangles"] = tris
            self._invalidate()
            return self._object_info(int(object_id), obj)

    def arrange_all(self):
        with self._lock:
            x = 0.0
            for oid, obj in self._objects.items():
                pts = obj["triangles"].reshape(-1, 3)
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                obj["triangles"] = obj["triangles"] + np.array([x - (lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
                x += (hi[0] - lo[0]) + 10.0
            self._invalidate()
            return self.list_objects()

    def delete_object(self, object_id):
        with self._lock:
            self._object(object_id)
            del self._objects[int(object_id)]
            self._invalidate()
            return True

    def clear_scene(self):
        with self._lock:
            self._objects.clear()
            self._invalidate()
            return True

    # ------------------------------------------------------------------ slicing and files
    def slice(self, progress=None, timeout=900.0):
        with self._lock:
            if not self._objects:
                raise SlicerError("На столе нет моделей")
            volume = 0.0
            for obj in self._objects.values():
                volume += abs(mesh_mod.analyze(obj["triangles"]).volume)
            layer = float(self._value("layer_height") or 0.1)
            infill = float(self._value("infill_sparse_density") or 20)
        for step in (0.25, 0.5, 0.75):
            if progress is not None:
                progress(step)
            if self._slice_delay:
                time.sleep(self._slice_delay / 3.0)
        used_mm3 = volume * (0.3 + 0.7 * infill / 100.0)
        length_m = used_mm3 / (np.pi * 0.875 ** 2) / 1000.0
        result = OrderedDict([
            ("state", "done"),
            ("print_time_s", int(used_mm3 / (8.0 * layer / 0.1) + 120)),
            ("material_length_m", [round(length_m, 2)]),
            ("material_weight_g", [round(used_mm3 / 1000.0 * 1.24, 1)]),
            ("material_names", [self.material_id]),
        ])
        with self._lock:
            self._slice_state = "done"
            self._slice_result = result
        if progress is not None:
            progress(1.0)
        return result

    def slice_status(self):
        with self._lock:
            return OrderedDict([("state", self._slice_state), ("progress", 1.0 if self._slice_state == "done" else 0.0),
                                ("result", self._slice_result)])

    def export_gcode(self, path):
        with self._lock:
            if self._slice_state != "done":
                raise SlicerError("Сначала нужно нарезать модель")
            state = self.state()
        with io.open(path, "w", encoding="utf-8") as f:
            f.write(";FAKE G-CODE from the AI Assistant test slicer\n")
            f.write(";MODE:%s\n;QUALITY:%s\n;MATERIAL:%s\n" % (state["printing_mode"], state["quality_type"],
                                                              state["material"]))
            f.write(";SETTINGS:%s\n" % json.dumps(state["user_settings"], ensure_ascii=False))
        return path

    def save_project(self, path):
        with self._lock:
            state = self.state()
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("FAKE_PROJECT.json", json.dumps(state, ensure_ascii=False, indent=1))
        return path
