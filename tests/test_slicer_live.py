"""Slicing logic of the STE Slicer adapter, run against stubbed Uranium/PyQt modules.

Reproduces the crash seen in STE Slicer 7.10: in classic mode the backends of the
other printing modes clear scene.gcode_dict when the settings check finishes, and
CuraEngineBackend then fails with KeyError on the G-code prefix.
"""
import enum
import importlib
import sys
import types

import pytest

from AIAssistant.agent.bridge import DirectBridge


class BackendState(enum.IntEnum):
    NotStarted = 1
    Processing = 2
    Done = 3
    Error = 4
    Disabled = 5


class Signal(object):
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def disconnect(self, slot):
        self._slots.remove(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class FakeRemoveOperation(object):
    def __init__(self, node):
        self.node = node


class FakeGroupedOperation(object):
    pushed = []

    def __init__(self):
        self.operations = []

    def addOperation(self, op):
        self.operations.append(op)

    def push(self):
        FakeGroupedOperation.pushed.append(self)
        for op in self.operations:
            op.node.parent.children.remove(op.node)


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def slicer_live(monkeypatch):
    class Logger(object):
        lines = []

        @classmethod
        def log(cls, level, msg, *args):
            cls.lines.append(msg % args if args else msg)

        @classmethod
        def logException(cls, level, msg, *args):
            cls.log(level, msg, *args)

    stubs = {
        "PyQt5": _stub("PyQt5"), "PyQt5.QtCore": _stub("PyQt5.QtCore", QUrl=object),
        "UM": _stub("UM"), "UM.Backend": _stub("UM.Backend"),
        "UM.Backend.Backend": _stub("UM.Backend.Backend", BackendState=BackendState),
        "UM.Logger": _stub("UM.Logger", Logger=Logger),
        "UM.Math": _stub("UM.Math"), "UM.Math.Matrix": _stub("UM.Math.Matrix", Matrix=object),
        "UM.Operations": _stub("UM.Operations"),
        "UM.Operations.GroupedOperation": _stub("UM.Operations.GroupedOperation", GroupedOperation=FakeGroupedOperation),
        "UM.Operations.Operation": _stub("UM.Operations.Operation", Operation=object),
        "UM.Operations.RemoveSceneNodeOperation": _stub("UM.Operations.RemoveSceneNodeOperation",
                                                        RemoveSceneNodeOperation=FakeRemoveOperation),
        "UM.Scene": _stub("UM.Scene"), "UM.Scene.SceneNode": _stub("UM.Scene.SceneNode", SceneNode=object),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "AIAssistant.agent.slicer_live", raising=False)
    module = importlib.import_module("AIAssistant.agent.slicer_live")
    monkeypatch.setitem(sys.modules, "AIAssistant.agent.slicer_live", module)
    module.Logger.lines = []
    return module


class FakeChecker(object):
    """MachineErrorChecker that stays busy for the first `busy_reads` reads."""

    def __init__(self, busy_reads=0):
        self.busy_reads = busy_reads
        self.reads = 0

    @property
    def needToWaitForResult(self):
        self.reads += 1
        return self.reads <= self.busy_reads


class FakeEngineBackend(object):
    """Mimics CuraEngineBackend: handlers registered from self at slice time, unguarded prefix."""

    def __init__(self, scene, scripts):
        self._scene = scene
        self._start_slice_job_build_plate = None
        self.backendStateChange = Signal()
        self.processingProgress = Signal()
        self.scripts = list(scripts)  # one list of steps per forceSlice call
        self.calls = 0
        self.checker_reads_at_start = []
        self.checker = None

    def getPluginId(self):
        return "CuraEngineBackend"

    def forceSlice(self):
        self.calls += 1
        if self.checker is not None:
            self.checker_reads_at_start.append(self.checker.reads)
        handlers = {"prefix": self._onGCodePrefixMessage, "finished": self._onSlicingFinishedMessage}
        self.backendStateChange.emit(BackendState.NotStarted)  # stopSlicing() at the start of slice()
        self.backendStateChange.emit(BackendState.Processing)
        self._start_slice_job_build_plate = 0
        self._scene.gcode_dict[0] = []
        for step in self.scripts.pop(0):
            if step == "layers":
                self._scene.gcode_dict[0].extend([";LAYER:0\n", ";LAYER:1\n"])
                self.processingProgress.emit(0.5)
            elif step == "clear":  # what CylindricalBackend._clearLayerData does
                self._scene.gcode_dict = {}
            elif step == "cancel":
                self.backendStateChange.emit(BackendState.NotStarted)
                return
            elif step == "error":
                self.backendStateChange.emit(BackendState.Error)
                return
        handlers["prefix"](";FLAVOR:Marlin\n")
        handlers["finished"](None)

    def _onGCodePrefixMessage(self, message):
        self._scene.gcode_dict[self._start_slice_job_build_plate].insert(0, message)

    def _onSlicingFinishedMessage(self, message):
        self.backendStateChange.emit(BackendState.Done)
        gcode_list = self._scene.gcode_dict[self._start_slice_job_build_plate]
        for index, line in enumerate(gcode_list):
            gcode_list[index] = line
        self._start_slice_job_build_plate = None


class FakePrintInformation(object):
    currentPrintTime = 3600
    materialLengths = [1.234]
    materialWeights = [3.21]
    materialNames = ["PETG"]


class FakeApp(object):
    def __init__(self, backend, checker):
        self.backend, self.checker = backend, checker
        backend.checker = checker
        self.scene = types.SimpleNamespace(gcode_dict={})
        backend._scene = self.scene

    def getBackend(self):
        return self.backend

    def getMachineErrorChecker(self):
        return self.checker

    def getController(self):
        return types.SimpleNamespace(getScene=lambda: self.scene)

    def getMultiBuildPlateModel(self):
        return types.SimpleNamespace(activeBuildPlate=0)

    def getPrintInformation(self):
        return FakePrintInformation()


def _slicer(module, scripts, busy_reads=0):
    backend = FakeEngineBackend(None, scripts)
    app = FakeApp(backend, FakeChecker(busy_reads))
    slicer = module.LiveSlicer(app, DirectBridge())
    slicer.list_objects = lambda: [{"id": 1}]
    return slicer, app


def test_slice_starts_only_after_the_settings_check(slicer_live):
    slicer, app = _slicer(slicer_live, [["layers"]], busy_reads=5)
    result = slicer.slice()
    assert result["print_time_s"] == 3600
    assert app.backend.checker_reads_at_start[0] > 5  # the check was idle before forceSlice
    assert app.scene.gcode_dict[0][0] == ";FLAVOR:Marlin\n"


def test_buffer_cleared_during_slicing_does_not_crash_and_slices_again(slicer_live):
    slicer, app = _slicer(slicer_live, [["layers", "clear"], ["layers"]])
    result = slicer.slice()
    assert result["state"] == "done"
    assert app.backend.calls == 2
    assert app.scene.gcode_dict[0] == [";FLAVOR:Marlin\n", ";LAYER:0\n", ";LAYER:1\n"]
    assert any("cleared during slicing" in line for line in slicer_live.Logger.lines)


def test_unguarded_backend_would_crash(slicer_live):
    """Without the guard the stand-in backend fails exactly like the slicer did."""
    backend = FakeEngineBackend(types.SimpleNamespace(gcode_dict={}), [["layers", "clear"]])
    backend.backendStateChange.connect(lambda state: None)
    with pytest.raises(KeyError):
        backend.forceSlice()


def test_cancelled_slice_is_repeated(slicer_live):
    slicer, app = _slicer(slicer_live, [["cancel"], ["layers"]])
    assert slicer.slice()["state"] == "done"
    assert app.backend.calls == 2


def test_slice_gives_up_after_repeated_failures(slicer_live):
    slicer, app = _slicer(slicer_live, [["clear"], ["cancel"], ["clear"]])
    with pytest.raises(slicer_live.SlicerError):
        slicer.slice()
    assert app.backend.calls == 3
    assert slicer.slice_status()["state"] == "error"


def test_slice_error_is_reported(slicer_live):
    slicer, app = _slicer(slicer_live, [["error"]])
    with pytest.raises(slicer_live.SlicerError):
        slicer.slice()
    assert app.backend.calls == 1
    assert slicer.slice_status()["state"] == "error"


class FakeNode(object):
    def __init__(self, parent, sliceable=True):
        self.parent, self.sliceable = parent, sliceable
        parent.children.append(self)

    def callDecoration(self, name):
        return self.sliceable if name == "isSliceable" else None


def test_clear_scene_removes_models_in_one_undo_step(slicer_live):
    """Works without app.deleteAll(), which is a no-op while the slicer's tools are disabled."""
    slicer, app = _slicer(slicer_live, [])
    root = types.SimpleNamespace(children=[])
    root.getChildren = lambda: list(root.children)
    models = [FakeNode(root), FakeNode(root)]
    FakeNode(root, sliceable=False)  # e.g. the build plate or a convex hull: stays
    app.scene.getRoot = lambda: root
    FakeGroupedOperation.pushed = []
    assert slicer.clear_scene() is True
    assert len(FakeGroupedOperation.pushed) == 1
    assert [op.node for op in FakeGroupedOperation.pushed[0].operations] == models
    assert len(root.children) == 1


class DepthBridge(object):
    """Runs calls directly and tracks whether code runs inside a bridge call (on the Qt main thread)."""

    def __init__(self):
        self.depth = 0

    def call(self, func, *args, **kwargs):
        self.depth += 1
        try:
            return func(*args, **kwargs)
        finally:
            self.depth -= 1


class FakeWriter(object):
    def __init__(self, bridge, ok=True, info=""):
        self.bridge, self.ok, self.info = bridge, ok, info
        self.calls = []

    def write(self, stream, nodes):
        self.calls.append((self.bridge.depth, nodes))
        stream.write(b"PK" if "b" in stream.mode else "G1 X0\n")
        return self.ok

    def getInformation(self):
        return self.info


def _writer_app(gcode_writer, project_writer):
    return types.SimpleNamespace(
        getMeshFileHandler=lambda: types.SimpleNamespace(getWriter=lambda w: gcode_writer if w == "GCodeWriter" else None),
        getWorkspaceFileHandler=lambda: types.SimpleNamespace(
            getWriter=lambda w: project_writer if w == "3MFWriter" else None),
        getController=lambda: types.SimpleNamespace(getScene=lambda: types.SimpleNamespace(getRoot=lambda: "root")))


def test_files_are_written_outside_the_main_thread_like_the_slicer_does(slicer_live, tmpdir):
    bridge = DepthBridge()
    gcode_writer, project_writer = FakeWriter(bridge), FakeWriter(bridge)
    slicer = slicer_live.LiveSlicer(_writer_app(gcode_writer, project_writer), bridge)
    gcode, project = str(tmpdir.join("part.gcode")), str(tmpdir.join("part.3mf"))
    assert slicer.export_gcode(gcode) == gcode and slicer.save_project(project) == project
    assert gcode_writer.calls == [(0, ["root"])] and project_writer.calls == [(0, ["root"])]
    with open(gcode) as f:
        assert f.read() == "G1 X0\n"
    with open(project, "rb") as f:
        assert f.read() == b"PK"
    assert any("Writing G-code took" in line for line in slicer_live.Logger.lines)


def test_failed_gcode_write_reports_the_writers_message(slicer_live, tmpdir):
    bridge = DepthBridge()
    writer = FakeWriter(bridge, ok=False, info="Please prepare G-code before exporting.")
    slicer = slicer_live.LiveSlicer(_writer_app(writer, None), bridge)
    with pytest.raises(slicer_live.SlicerError, match="prepare G-code"):
        slicer.export_gcode(str(tmpdir.join("part.gcode")))
    with pytest.raises(slicer_live.SlicerError, match="3MF"):
        slicer.save_project(str(tmpdir.join("part.3mf")))
