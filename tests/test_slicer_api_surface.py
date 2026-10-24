"""The STE Slicer adapter cannot run outside the slicer, so check statically that
every class member it uses exists in the STE Slicer / Uranium sources.

STE_SLICER_SRC (default: third_party/STE-Slicer-src) is a directory with both
sources unpacked as STE-Slicer-master/ and Uranium-steslicer/; the tests are
skipped when it is absent. Members are methods, class-level assignments (Qt
properties, signals, enum values) and attributes assigned as self.<name>
inside the class.
"""
import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("STE_SLICER_SRC") or os.path.join(ROOT, "third_party", "STE-Slicer-src")
U = "Uranium-steslicer/"
S = "STE-Slicer-master/"

pytestmark = pytest.mark.skipif(not os.path.isdir(SRC), reason="STE Slicer / Uranium sources not present")

# (file, class, members used by plugin/AIAssistant/AIAssistant.py and agent/slicer_live.py)
CHECKS = [
    (U + "UM/Extension.py", "Extension", ["setMenuName", "addMenuItem"]),
    (U + "UM/Application.py", "Application", ["callLater", "getPreferences", "savePreferences", "getController",
                                               "getInstance"]),
    (U + "UM/Qt/QtApplication.py", "QtApplication", ["getMeshFileHandler", "getWorkspaceFileHandler"]),
    (S + "steslicer/SteSlicerApplication.py", "SteSlicerApplication",
     ["getGlobalContainerStack", "getMachineManager", "getQualityManager", "getMaterialManager",
      "getPrintInformation", "readLocalFile", "arrangeAll", "getBackend", "fileLoaded", "getMachineErrorChecker",
      "getMultiBuildPlateModel"]),
    (S + "steslicer/Machines/MachineErrorChecker.py", "MachineErrorChecker", ["needToWaitForResult"]),
    (S + "steslicer/Machines/Models/MultiBuildPlateModel.py", "MultiBuildPlateModel", ["activeBuildPlate"]),
    (S + "steslicer/Settings/MachineManager.py", "MachineManager",
     ["setQualityGroupByQualityType", "setMaterialById", "activeQualityType"]),
    (S + "steslicer/Machines/QualityManager.py", "QualityManager", ["getQualityGroups", "createQualityChanges"]),
    (S + "steslicer/Machines/QualityGroup.py", "QualityGroup", ["name", "is_available", "quality_type"]),
    (S + "steslicer/Machines/MaterialManager.py", "MaterialManager", ["getAvailableMaterialsForMachineExtruder"]),
    (S + "steslicer/Machines/ContainerNode.py", "ContainerNode", ["getMetaDataEntry"]),
    (S + "steslicer/PrintInformation.py", "PrintInformation",
     ["currentPrintTime", "materialLengths", "materialWeights", "materialNames"]),
    (S + "steslicer/Settings/SteSlicerContainerStack.py", "SteSlicerContainerStack",
     ["userChanges", "material", "definition"]),
    (S + "steslicer/Settings/GlobalStack.py", "GlobalStack", ["extruders"]),
    (U + "UM/Settings/ContainerStack.py", "ContainerStack", ["getProperty", "getName"]),
    (U + "UM/Settings/InstanceContainer.py", "InstanceContainer",
     ["setProperty", "getProperty", "getAllKeys", "clear", "getMetaDataEntry", "getId"]),
    (U + "UM/Settings/DefinitionContainer.py", "DefinitionContainer", ["getId"]),
    (U + "UM/Scene/SceneNode.py", "SceneNode",
     ["getMeshDataTransformedVertices", "getMeshData", "getWorldTransformation", "getLocalTransformation",
      "setTransformation", "getBoundingBox", "getChildren", "getParent", "getName", "callDecoration"]),
    (U + "UM/Mesh/MeshData.py", "MeshData", ["hasIndices", "getIndices", "getVertices"]),
    (U + "UM/Math/Matrix.py", "Matrix", ["getData"]),
    (U + "UM/Math/AxisAlignedBox.py", "AxisAlignedBox", ["minimum", "maximum"]),
    (U + "UM/Backend/Backend.py", "Backend", ["backendStateChange", "processingProgress"]),
    (U + "UM/Backend/Backend.py", "BackendState", ["NotStarted", "Processing", "Done", "Error", "Disabled"]),
    (S + "plugins/CylindricalBackend/CylindricalBackend.py", "CylindricalBackend", ["forceSlice"]),
    (S + "plugins/CuraEngineBackend/CuraEngineBackend.py", "CuraEngineBackend",
     ["forceSlice", "_onGCodePrefixMessage", "_onSlicingFinishedMessage", "_scene", "_start_slice_job_build_plate"]),
    (U + "UM/PluginObject.py", "PluginObject", ["getPluginId"]),
    (U + "UM/Operations/Operation.py", "Operation", ["push", "undo", "redo"]),
    (U + "UM/Operations/RemoveSceneNodeOperation.py", "RemoveSceneNodeOperation", ["__init__"]),
    (U + "UM/Operations/GroupedOperation.py", "GroupedOperation", ["addOperation"]),
    (U + "UM/FileHandler/FileHandler.py", "FileHandler", ["getWriter"]),
    (U + "UM/FileHandler/FileWriter.py", "FileWriter", ["getInformation"]),
    (S + "plugins/GCodeWriter/GCodeWriter.py", "GCodeWriter", ["write"]),
    (S + "plugins/3MFWriter/ThreeMFWorkspaceWriter.py", "ThreeMFWorkspaceWriter", ["write"]),
    (U + "UM/Preferences.py", "Preferences", ["addPreference", "getValue", "setValue"]),
    (U + "UM/Logger.py", "Logger", ["log", "logException"]),
    (U + "UM/Message.py", "Message", ["show"]),
    (U + "UM/Qt/Duration.py", "Duration", ["__int__"]),
]


_ANN_ASSIGN = getattr(ast, "AnnAssign", None)  # Python 3.6+ only


def class_members(path, class_name):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            members = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.Assign):
                    members.update(t.id for t in item.targets if isinstance(t, ast.Name))
                elif _ANN_ASSIGN is not None and isinstance(item, _ANN_ASSIGN) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
            for sub in ast.walk(node):  # self.<name> = ... inside methods
                if isinstance(sub, ast.Assign) or (_ANN_ASSIGN is not None and isinstance(sub, _ANN_ASSIGN)):
                    targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                    for t in targets:
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                            members.add(t.attr)
            return members
    return None


@pytest.mark.parametrize("path,class_name,names", CHECKS, ids=[c[1] for c in CHECKS])
def test_member_exists(path, class_name, names):
    full = os.path.join(SRC, path)
    assert os.path.isfile(full), "file missing: %s" % path
    members = class_members(full, class_name)
    assert members is not None, "class %s not found in %s" % (class_name, path)
    missing = [n for n in names if n not in members]
    assert not missing, "%s.%s missing in the STE Slicer / Uranium sources" % (class_name, missing)


def test_imported_modules_exist():
    modules = ["UM/Application.py", "UM/Extension.py", "UM/Logger.py", "UM/Message.py", "UM/Backend/Backend.py",
               "UM/Math/Matrix.py", "UM/Operations/Operation.py", "UM/Operations/RemoveSceneNodeOperation.py", "UM/Operations/GroupedOperation.py",
               "UM/Scene/SceneNode.py"]
    for module in modules:
        assert os.path.isfile(os.path.join(SRC, U, module)), module


def test_engine_backend_registers_handlers_from_instance_attributes():
    """slicer_live guards the G-code buffer by replacing CuraEngineBackend's handler attributes
    on the instance; that works only if slice() looks them up on self each time."""
    path = os.path.join(SRC, S, "plugins/CuraEngineBackend/CuraEngineBackend.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    registered = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "slice":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Attribute)
                        and isinstance(sub.value.value, ast.Name) and sub.value.value.id == "self"):
                    registered.add(sub.value.attr)
    assert {"_onGCodePrefixMessage", "_onSlicingFinishedMessage"} <= registered
