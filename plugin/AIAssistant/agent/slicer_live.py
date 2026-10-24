"""STE Slicer adapter: the only module that talks to the slicer's Python API.

Every public method may be called from a worker thread; the work itself is
done on the Qt main thread through the bridge (`app.callLater`). API names
follow the STE Slicer 7.x and Uranium sources (steslicer.SteSlicerApplication,
steslicer.Settings.MachineManager, UM.*).
"""
import contextlib
import io
import os
import threading
import time
from collections import OrderedDict

import numpy as np

from PyQt5.QtCore import QUrl

from UM.Backend.Backend import BackendState
from UM.Logger import Logger
from UM.Math.Matrix import Matrix
from UM.Operations.GroupedOperation import GroupedOperation
from UM.Operations.Operation import Operation
from UM.Operations.RemoveSceneNodeOperation import RemoveSceneNodeOperation
from UM.Scene.SceneNode import SceneNode

from ..printplan.io_utils import read_json_resource
from .slicer_base import LiveCatalogContext, SlicerAPI, SlicerError, bbox_info, scene_to_printer

# printer (x, y, z) -> scene (x, z, -y)
_P2S = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_STATE_NAMES = {
    BackendState.NotStarted: "not_started", BackendState.Processing: "processing",
    BackendState.Done: "done", BackendState.Error: "error", BackendState.Disabled: "disabled",
}


def _guard_gcode_buffer(backend):
    """Keep a cleared G-code buffer from closing the slicer (classic backend only).

    CuraEngineBackend's handlers for the G-code prefix and for the end of slicing index
    scene.gcode_dict[build plate] without checking that the entry still exists. If the
    buffer was cleared during slicing, the KeyError reaches the slicer's crash handler,
    which closes the slicer. The guard recreates the entry empty instead (the G-code
    sliced so far is lost anyway) and drops the prefix; `LiveSlicer.slice` then finds no
    G-code and slices again. CuraEngineBackend registers its handlers from the instance
    attributes when a slice starts, so wrapping the attributes is enough. In 5D the
    cylindrical backend registers its own handlers, which check the buffer themselves.
    """
    if backend.getPluginId() != "CuraEngineBackend" or getattr(backend, "_ai_guarded", False):
        return
    on_prefix, on_finished = backend._onGCodePrefixMessage, backend._onSlicingFinishedMessage

    def recreate_missing_entry():
        gcode_dict = getattr(backend._scene, "gcode_dict", None)
        plate = backend._start_slice_job_build_plate
        if gcode_dict is None or plate is None or plate in gcode_dict:
            return False
        Logger.log("w", "[AI] The G-code buffer was cleared during slicing; the slice will be repeated")
        gcode_dict[plate] = []
        return True

    def guarded_prefix(message):
        if not recreate_missing_entry():
            on_prefix(message)

    def guarded_finished(message):
        recreate_missing_entry()
        on_finished(message)

    backend._onGCodePrefixMessage = guarded_prefix
    backend._onSlicingFinishedMessage = guarded_finished
    backend._ai_guarded = True


class _SetTransformationOperation(Operation):
    """Undoable replacement of a node's transformation matrix."""

    def __init__(self, node, transformation):
        super().__init__()
        self._node = node
        self._old = node.getLocalTransformation()
        self._new = transformation

    def undo(self):
        self._node.setTransformation(self._old)

    def redo(self):
        self._node.setTransformation(self._new)


class LiveSlicer(SlicerAPI):
    name = "ste_slicer"

    def __init__(self, app, bridge):
        self._app = app
        self._bridge = bridge
        self._catalog = read_json_resource("slicer_catalog.json")
        self._slice_state = "not_started"
        self._slice_progress = 0.0
        self._slice_result = None

    # ------------------------------------------------------------------ plumbing (main thread)
    def _call(self, func, *args, **kwargs):
        return self._bridge.call(func, *args, **kwargs)

    def _global(self):
        stack = self._app.getGlobalContainerStack()
        if stack is None:
            raise SlicerError("В слайсере не выбран принтер")
        return stack

    def _extruder(self, position=0):
        return self._global().extruders[str(position)]

    def _root(self):
        return self._app.getController().getScene().getRoot()

    def _nodes(self):
        out = []
        for node in self._root().getChildren():
            if not isinstance(node, SceneNode):
                continue
            if node.callDecoration("isSliceable") or node.callDecoration("isGroup"):
                out.append(node)
        return out

    def _node(self, object_id):
        for node in self._nodes():
            if id(node) == int(object_id):
                return node
        raise SlicerError("Нет объекта с id %s" % object_id)

    def _node_info(self, node):
        info = OrderedDict([("id", id(node)), ("name", node.getName())])
        box = node.getBoundingBox()
        if box is not None:
            lo, hi = box.minimum, box.maximum
            # scene (x, y, z) -> printer (x, -z, y); min/max swap on the flipped axis
            info.update(bbox_info([lo.x, -hi.z, lo.y], [hi.x, -lo.z, hi.y]))
        return info

    @contextlib.contextmanager
    def _no_profile_dialog(self):
        """Keep user changes instead of asking «discard or keep?» when switching profiles."""
        prefs = self._app.getPreferences()
        key = "steslicer/choice_on_profile_override"
        old = prefs.getValue(key)
        prefs.setValue(key, "always_keep")
        try:
            yield
        finally:
            prefs.setValue(key, old)

    def _stack_for(self, key):
        per_extruder = self._global().getProperty(key, "settable_per_extruder")
        return ("extruder", self._extruder()) if per_extruder else ("global", self._global())

    @staticmethod
    def _state_name(state):
        value = getattr(state, "value", state)
        return str(value) if value is not None else None

    # ------------------------------------------------------------------ state, profiles, settings
    def state(self):
        return self._call(self._state_main)

    def _state_main(self):
        stack = self._global()
        manager = self._app.getMachineManager()
        out = OrderedDict()
        out["slicer"] = "STE Slicer"
        out["machine"] = stack.getName()
        out["machine_definition"] = stack.definition.getId()
        out["printing_mode"] = stack.getProperty("printing_mode", "value")
        out["quality_type"] = manager.activeQualityType
        material = self._extruder().material
        out["material"] = material.getMetaDataEntry("base_file") or material.getId()
        user = OrderedDict()
        for container in [stack.userChanges] + [e.userChanges for e in stack.extruders.values()]:
            for key in container.getAllKeys():
                user[key] = container.getProperty(key, "value")
        out["user_settings"] = user
        out["objects"] = [self._node_info(n) for n in self._nodes()]
        out["slice_state"] = self._slice_state
        return out

    def list_quality_profiles(self):
        return self._call(self._quality_main)

    def _quality_main(self):
        groups = self._app.getQualityManager().getQualityGroups(self._global())
        heights = dict((q["quality_type"], q["layer_height"]) for q in self._catalog["quality_profiles"])
        return [OrderedDict([("quality_type", qt), ("name", g.name), ("layer_height", heights.get(qt)),
                             ("available", bool(g.is_available))]) for qt, g in groups.items()]

    def list_materials(self):
        return self._call(self._materials_main)

    def _materials_main(self):
        manager = self._app.getMaterialManager()
        nodes = manager.getAvailableMaterialsForMachineExtruder(self._global(), self._extruder()) or {}
        out = []
        for material_id, node in nodes.items():
            out.append(OrderedDict([("id", material_id), ("name", node.getMetaDataEntry("name")),
                                    ("brand", node.getMetaDataEntry("brand"))]))
        return out

    def get_settings(self, keys):
        return self._call(self._get_settings_main, list(keys))

    def _get_settings_main(self, keys):
        out = OrderedDict()
        for key in keys:
            if self._global().getProperty(key, "type") is None:
                out[key] = OrderedDict([("error", "неизвестный ключ")])
                continue
            stack_name, stack = self._stack_for(key)
            out[key] = OrderedDict([("value", stack.getProperty(key, "value")), ("stack", stack_name),
                                    ("validation", self._state_name(stack.getProperty(key, "validationState")))])
        return out

    def set_settings(self, values):
        return self._call(self._set_settings_main, OrderedDict(values))

    def _set_settings_main(self, values):
        out = OrderedDict()
        for key, value in values.items():
            if self._global().getProperty(key, "type") is None:
                raise SlicerError("Неизвестный ключ настройки: %s" % key)
            stack_name, stack = self._stack_for(key)
            stack.userChanges.setProperty(key, "value", value)
            out[key] = OrderedDict([("value", stack.getProperty(key, "value")), ("stack", stack_name),
                                    ("validation", self._state_name(stack.getProperty(key, "validationState")))])
        return out

    def reset_settings(self):
        return self._call(self._reset_main)

    def _reset_main(self):
        stack = self._global()
        for extruder in stack.extruders.values():
            extruder.userChanges.clear()
        stack.userChanges.clear()
        return True

    def set_quality(self, quality_type):
        return self._call(self._set_quality_main, quality_type)

    def _set_quality_main(self, quality_type):
        groups = self._app.getQualityManager().getQualityGroups(self._global())
        if quality_type not in groups or not groups[quality_type].is_available:
            raise SlicerError("Профиль качества %s недоступен для этого принтера и материала" % quality_type)
        with self._no_profile_dialog():
            self._app.getMachineManager().setQualityGroupByQualityType(quality_type)
        return self._app.getMachineManager().activeQualityType

    def set_material(self, material_id, extruder=0):
        return self._call(self._set_material_main, material_id, extruder)

    def _set_material_main(self, material_id, extruder):
        available = self._app.getMaterialManager().getAvailableMaterialsForMachineExtruder(
            self._global(), self._extruder(extruder)) or {}
        if material_id not in available:
            raise SlicerError("Материал %s недоступен для этого экструдера" % material_id)
        with self._no_profile_dialog():
            self._app.getMachineManager().setMaterialById(str(extruder), material_id)
        material = self._extruder(extruder).material
        return material.getMetaDataEntry("base_file") or material.getId()

    def set_printing_mode(self, mode):
        if mode not in ("classic", "cylindrical", "cylindrical_full"):
            raise SlicerError("Неизвестный режим печати %s" % mode)
        return self._call(self._set_mode_main, mode)

    def _set_mode_main(self, mode):
        # The same container the mode switch in the sidebar writes to (storeIndex 0 = user changes).
        self._global().userChanges.setProperty("printing_mode", "value", mode)
        return self._global().getProperty("printing_mode", "value")

    def save_profile(self, name):
        return self._call(self._save_profile_main, name)

    def _save_profile_main(self, name):
        self._app.getQualityManager().createQualityChanges(name)
        return name

    def slicer_context(self):
        quality = [q["quality_type"] for q in self.list_quality_profiles() if q["available"]]
        materials = [m["id"] for m in self.list_materials()]
        return LiveCatalogContext(self._catalog, quality, materials)

    # ------------------------------------------------------------------ scene
    def load_model(self, path):
        if not os.path.isfile(path):
            raise SlicerError("Файл не найден: %s" % path)
        before = set(o["id"] for o in self.list_objects())
        loaded = threading.Event()

        def on_loaded(file_name):
            loaded.set()

        def start():
            self._app.fileLoaded.connect(on_loaded)
            self._app.readLocalFile(QUrl.fromLocalFile(path), skip_project_file_check=True)

        self._call(start)
        try:
            if not loaded.wait(180.0):
                raise SlicerError("Модель не загрузилась за 3 минуты: %s" % path)
        finally:
            try:
                self._call(self._app.fileLoaded.disconnect, on_loaded)
            except Exception:
                pass
        for _ in range(20):  # nodes are added right after fileLoaded is emitted
            fresh = [o for o in self.list_objects() if o["id"] not in before]
            if fresh:
                self._invalidate_slice()
                return fresh
            time.sleep(0.1)
        raise SlicerError("Файл прочитан, но в сцене не появилось объектов: %s" % path)

    def list_objects(self):
        return self._call(lambda: [self._node_info(n) for n in self._nodes()])

    def get_triangles(self, object_id):
        return self._call(self._triangles_main, object_id)

    def _triangles_main(self, object_id):
        node = self._node(object_id)
        parts = []
        nodes = node.getChildren() if node.callDecoration("isGroup") else [node]
        for item in nodes:
            mesh = item.getMeshData()
            if mesh is None:
                continue
            verts = item.getMeshDataTransformedVertices()
            if verts is None or len(verts) == 0:
                continue
            if mesh.hasIndices():
                tris = np.asarray(verts, dtype=np.float64)[np.asarray(mesh.getIndices())]
            else:
                tris = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
            parts.append(tris)
        if not parts:
            raise SlicerError("У объекта нет сетки")
        return scene_to_printer(np.concatenate(parts, axis=0))

    def transform_object(self, object_id, rotation=None, translation=None):
        rot = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        shift = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
        delta = np.eye(4)
        delta[:3, :3] = _P2S.dot(rot).dot(_P2S.T)
        delta[:3, 3] = _P2S.dot(shift)
        return self._call(self._transform_main, object_id, delta)

    def _transform_main(self, object_id, delta):
        node = self._node(object_id)
        world = np.asarray(node.getWorldTransformation().getData(), dtype=np.float64)
        parent = node.getParent()
        new_world = delta.dot(world)
        if parent is not None and parent is not self._root():
            new_world = np.linalg.inv(np.asarray(parent.getWorldTransformation().getData())).dot(new_world)
        _SetTransformationOperation(node, Matrix(new_world.tolist())).push()
        self._invalidate_slice()
        return self._node_info(node)

    def move_object(self, object_id, x=None, y=None, z=None):
        tris = self.get_triangles(object_id)
        pts = tris.reshape(-1, 3)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        shift = np.zeros(3)
        if x is not None:
            shift[0] = float(x) - center[0]
        if y is not None:
            shift[1] = float(y) - center[1]
        if z is not None:
            shift[2] = float(z) - pts[:, 2].min()
        return self.transform_object(object_id, None, shift)

    def scale_object(self, object_id, factors):
        f = np.asarray(factors, dtype=np.float64).reshape(-1)
        if f.size == 1:
            f = np.repeat(f, 3)
        tris = self.get_triangles(object_id)
        pts = tris.reshape(-1, 3)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        scale = np.diag(f)
        shift = center - scale.dot(center)   # keep the centre in place in x and y
        shift[2] = -f[2] * pts[:, 2].min()  # keep the bottom on the bed
        return self.transform_object(object_id, scale, shift)

    def arrange_all(self):
        self._call(self._app.arrangeAll)
        self._invalidate_slice()
        return self.list_objects()

    def delete_object(self, object_id):
        def run():
            RemoveSceneNodeOperation(self._node(object_id)).push()
        self._call(run)
        self._invalidate_slice()
        return True

    def clear_scene(self):
        # Not app.deleteAll(): it does nothing while the tools are disabled (e.g. on the Preview stage).
        def run():
            nodes = self._nodes()
            if nodes:
                op = GroupedOperation()
                for node in nodes:
                    op.addOperation(RemoveSceneNodeOperation(node))
                op.push()  # one step for Ctrl+Z

        self._call(run)
        self._invalidate_slice()
        return True

    # ------------------------------------------------------------------ slicing and files
    def _invalidate_slice(self):
        self._slice_state = "not_started"
        self._slice_progress = 0.0
        self._slice_result = None

    def slice(self, progress=None, timeout=900.0, attempts=3):
        if not self.list_objects():
            raise SlicerError("На столе нет моделей")
        self._slice_state = "processing"
        deadline = time.time() + timeout
        for attempt in range(1, attempts + 1):
            self._wait_until_settled()
            state = self._slice_once(progress, deadline)
            if state == BackendState.Done:
                if self._call(self._has_gcode):
                    break
                Logger.log("w", "[AI] Slicing finished without G-code (attempt %d), slicing again", attempt)
            elif state is None:
                Logger.log("w", "[AI] The slicer stopped the slice (attempt %d), slicing again", attempt)
            else:
                self._slice_state = _STATE_NAMES.get(state, "error")
                raise SlicerError("Слайсинг не удался (состояние: %s). Проверьте ошибки настроек в слайсере"
                                  % self._slice_state)
        else:
            self._slice_state = "error"
            raise SlicerError("Слайсинг %d раза прерывался или завершался без G-code. Проверьте, что модель "
                              "помещается в область печати, и не меняйте сцену во время слайсинга" % attempts)
        time.sleep(1.0)  # PrintInformation is updated right after the backend reports Done
        result = self._call(self._print_info_main)
        self._slice_state = "done"
        self._slice_result = result
        return result

    def _slice_once(self, progress, deadline, start_timeout=30.0):
        """Run one slice. Returns the final backend state, or None if the slicer stopped
        the slice (a scene or setting change) or never started it."""
        done = threading.Event()
        box = {}
        backend = self._call(self._app.getBackend)

        def on_state(state):
            if state == BackendState.Processing:
                box["started"] = True
            elif state in (BackendState.Done, BackendState.Error, BackendState.Disabled):
                box["state"] = state
                done.set()
            elif state == BackendState.NotStarted and box.get("started"):
                done.set()

        def on_progress(amount):
            self._slice_progress = float(amount)
            if progress is not None:
                try:
                    progress(float(amount))
                except Exception:
                    pass

        def start():
            _guard_gcode_buffer(backend)
            backend.backendStateChange.connect(on_state)
            backend.processingProgress.connect(on_progress)
            backend.forceSlice()

        def stop_listening():
            backend.backendStateChange.disconnect(on_state)
            backend.processingProgress.disconnect(on_progress)

        start_by = time.time() + start_timeout
        self._call(start)
        try:
            while not done.wait(0.5):
                now = time.time()
                if now > deadline:
                    self._slice_state = "error"
                    raise SlicerError("Слайсинг не завершился вовремя")
                if not box.get("started") and now > start_by:
                    break  # e.g. the backend was still processing the previous layers
        finally:
            try:
                self._call(stop_listening)
            except Exception:
                Logger.logException("w", "[AI] Could not disconnect from the backend signals")
        return box.get("state")

    def _wait_until_settled(self, timeout=60.0, quiet=1.0):
        """Wait until the slicer has re-checked the settings after our changes.

        When that check finishes, the backends of the other printing modes clear the
        shared G-code buffer (scene.gcode_dict) if they are not slicing themselves, even
        while the active backend is slicing. So a slice starts only after the check has
        stayed idle for `quiet` seconds.
        """
        checker = self._call(self._app.getMachineErrorChecker)
        end = time.time() + timeout
        idle_since = None
        while time.time() < end:
            now = time.time()
            if self._call(lambda: checker.needToWaitForResult):
                idle_since = None
            elif idle_since is None:
                idle_since = now
            elif now - idle_since >= quiet:
                return True
            time.sleep(0.1)
        Logger.log("w", "[AI] The settings check did not finish in %.0f s, slicing anyway", timeout)
        return False

    def _has_gcode(self):
        scene = self._app.getController().getScene()
        plate = self._app.getMultiBuildPlateModel().activeBuildPlate
        return bool(getattr(scene, "gcode_dict", {}).get(plate))

    def _print_info_main(self):
        info = self._app.getPrintInformation()
        out = OrderedDict()
        out["state"] = "done"
        try:
            out["print_time_s"] = int(info.currentPrintTime)
        except Exception:
            out["print_time_s"] = None
        out["material_length_m"] = [round(float(v), 2) for v in (info.materialLengths or [])]
        out["material_weight_g"] = [round(float(v), 1) for v in (info.materialWeights or [])]
        out["material_names"] = list(info.materialNames or [])
        return out

    def slice_status(self):
        return OrderedDict([("state", self._slice_state), ("progress", round(self._slice_progress, 3)),
                            ("result", self._slice_result)])

    def export_gcode(self, path):
        writer, ok = self._write_file(self._app.getMeshFileHandler, "GCodeWriter", path, "G-code", text=True)
        if not ok:
            raise SlicerError(writer.getInformation() or "Не удалось записать G-code: сначала нарежьте модель")
        return path

    def save_project(self, path):
        writer, ok = self._write_file(self._app.getWorkspaceFileHandler, "3MFWriter", path, "проект 3MF", text=False)
        if ok is False:
            raise SlicerError(writer.getInformation() or "Не удалось сохранить проект")
        return path

    def _write_file(self, get_handler, writer_id, path, what, text):
        """Write the way the slicer's own «Save» does (LocalFileOutputDevice -> WriteFileJob):
        the writer runs in a worker thread, not on the Qt main thread. On the main thread a big
        model or G-code froze the slicer for the whole write and outlasted the bridge timeout,
        and GCodeWriter rendered its preview snapshot outside the render loop."""
        writer, nodes = self._call(lambda: (get_handler().getWriter(writer_id), [self._root()]))
        if writer is None:
            raise SlicerError("В слайсере нет модуля записи: %s" % what)
        Logger.log("d", "[AI] Writing %s to %s", what, path)
        started = time.time()
        with (io.open(path, "w", encoding="utf-8") if text else open(path, "wb")) as stream:
            ok = writer.write(stream, nodes)
        Logger.log("d", "[AI] Writing %s took %.1f s (result: %s)", what, time.time() - started, ok)
        return writer, ok
