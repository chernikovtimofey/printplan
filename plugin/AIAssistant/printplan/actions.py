"""Validated plan -> ordered list of slicer actions for the plugin's MCP layer.

Coordinates: the pipeline works in the STL frame (Z up). Uranium's STL reader
maps file coordinates (x, y, z) to scene coordinates (x, z, -y) — Y is up in
the scene — so a rotation R given in the STL frame becomes M R M^T in the
scene with M below (plugins/FileHandlers/STLReader/STLReader.py, Uranium 3.6).
"""
from collections import OrderedDict

import numpy as np

STL_TO_SCENE = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


def stl_to_scene_rotation(rotation):
    return STL_TO_SCENE.dot(np.asarray(rotation, dtype=np.float64)).dot(STL_TO_SCENE.T)


def stl_to_scene_point(point):
    return STL_TO_SCENE.dot(np.asarray(point, dtype=np.float64))


def _matrix(m):
    return [[round(float(x), 6) for x in row] for row in m]


def _vector(v):
    return [round(float(x), 4) for x in v]


def split_settings(values, ctx):
    """Global-stack and extruder-stack settings, as the slicer stores user changes."""
    global_values, extruder_values = OrderedDict(), OrderedDict()
    for key, value in values.items():
        spec = ctx.setting(key) or {}
        if spec.get("settable_per_extruder", True):
            extruder_values[key] = value
        else:
            global_values[key] = value
    return global_values, extruder_values


def build_actions(plan, settings, orientation, ctx, axis=None):
    """`settings`: OrderedDict key -> {value, reason, ...} (validation.final_settings)."""
    mode = plan["mode"]
    values = OrderedDict((k, v["value"]) for k, v in settings.items() if k != "printing_mode")
    global_values, extruder_values = split_settings(values, ctx)
    actions = []

    def add(tool, args, why):
        actions.append(OrderedDict([("step", len(actions) + 1), ("tool", tool), ("args", args), ("why", why)]))

    add("set_printing_mode", OrderedDict([("printing_mode", mode)]), "режим печати")
    add("set_quality", OrderedDict([("quality_type", plan["quality_type"])]), "профиль качества")
    add("set_material", OrderedDict([("extruder", 0), ("material_id", plan["material"])]), "материал основного экструдера")
    if global_values:
        add("set_settings", OrderedDict([("stack", "global"), ("values", global_values)]), "глобальные настройки")
    if extruder_values:
        add("set_settings", OrderedDict([("stack", "extruder"), ("extruder", 0), ("values", extruder_values)]),
            "настройки экструдера")
    rot = orientation.rotation
    add("rotate_model", OrderedDict([
        ("orientation_id", orientation.id),
        ("rotation_stl", _matrix(rot)),
        ("rotation_scene", _matrix(stl_to_scene_rotation(rot))),
        ("up_in_model_stl", _vector(orientation.up)),
    ]), "ориентация: %s" % plan["orientation"].get("reason", ""))
    if mode == "classic":
        add("arrange_all", OrderedDict(), "расстановка на столе")
    else:
        args = OrderedDict([("center_axis_on_bed", True), ("lay_on_bed", True)])
        if axis is not None:
            args["axis_point_model_stl"] = _vector(axis.point)
            args["axis_direction_model_stl"] = _vector(axis.direction)
            args["axis_point_model_scene"] = _vector(stl_to_scene_point(axis.point))
        args["translation_after_rotation_stl"] = _vector(orientation.translation)
        add("place_model", args, "5D: ось вращения через центр стола, без авторасстановки")
    add("slice", OrderedDict(), "слайсинг")
    return actions
