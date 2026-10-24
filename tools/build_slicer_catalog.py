#!/usr/bin/env python
"""Build printplan/data/slicer_catalog.json from an STE Slicer source checkout.

The catalog is the static picture of the slicer that the pipeline uses when it
runs outside the slicer (CLI, tests) and as the source of allowed values for
the LLM: setting keys with types, defaults and limits for Stereotech Hybrid 530
V5.2.4, quality profiles and 1.75 mm materials.

Usage:
    python tools/build_slicer_catalog.py [--repo PATH_TO_STE_SLICER_SOURCE] [--out ...]

Limits given as expressions (e.g. "0.8 * min(extruderValues('machine_nozzle_size'))")
are evaluated over the default values of the machine; expressions that cannot be
evaluated are stored as null and the raw expression is kept next to them.
"""
from __future__ import print_function

import argparse
import configparser
import io
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# STE Slicer source tree with the Hybrid 530 V5.2.4 definitions (the public repository only has V5.2).
DEFAULT_REPO = os.path.join(ROOT, "third_party", "STE-Slicer")
DEFAULT_OUT = os.path.join(ROOT, "plugin", "AIAssistant", "printplan", "data", "slicer_catalog.json")

MACHINE_ID = "hte530v5.2.4"
EXTRUDER_IDS = ["stereotech_v5.2.4_extruder_main", "stereotech_v5.2.4_extruder_second"]
USED_MODES = ["classic", "cylindrical", "cylindrical_full"]

LIMIT_PROPS = [
    ("minimum_value", "min"),
    ("maximum_value", "max"),
    ("minimum_value_warning", "min_warning"),
    ("maximum_value_warning", "max_warning"),
]

# Material XML setting names -> slicer keys (plugins/XmlMaterialProfile/XmlMaterialProfile.py)
MATERIAL_SETTING_MAP = {
    "print temperature": "default_material_print_temperature",
    "heated bed temperature": "default_material_bed_temperature",
    "standby temperature": "material_standby_temperature",
    "print cooling": "cool_fan_speed",
    "retraction amount": "retraction_amount",
    "retraction speed": "retraction_speed",
}


def load_json(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def find_definition(repo, def_id):
    for sub in ("definitions", "extruders"):
        path = os.path.join(repo, "resources", sub, def_id + ".def.json")
        if os.path.exists(path):
            return path
    raise IOError("definition not found: " + def_id)


def definition_chain(repo, def_id):
    """Return [leaf, ..., root] definition dicts."""
    chain = []
    while def_id:
        data = load_json(find_definition(repo, def_id))
        data["_id"] = def_id
        chain.append(data)
        def_id = data.get("inherits")
    return chain


def flatten_settings(settings, category=None, out=None):
    """Flatten the nested fdmprinter 'settings' tree into key -> properties."""
    if out is None:
        out = OrderedDict()
    for key, node in settings.items():
        props = OrderedDict((k, v) for k, v in node.items() if k != "children")
        if node.get("type") == "category":
            cat = key
        else:
            cat = category
            props["_category"] = category
            out[key] = props
        if "children" in node:
            flatten_settings(node["children"], cat, out)
    return out


def apply_overrides(settings, chain):
    """Apply 'overrides' from root-most to leaf-most definition."""
    for data in reversed(chain):
        for key, override in data.get("overrides", {}).items():
            if key in settings:
                settings[key].update(override)
            else:
                settings[key] = OrderedDict(override)


def make_eval_context(settings):
    """Evaluation namespace: setting names resolve to their default values."""
    ctx = {}
    for key, props in settings.items():
        if "default_value" in props:
            ctx[key] = props["default_value"]
    ctx["math"] = math
    ctx["extruderValues"] = lambda k: [ctx[k]]
    ctx["extruderValue"] = lambda nr, k: ctx[k]
    ctx["resolveOrValue"] = lambda k: ctx[k]
    ctx["defaultExtruderPosition"] = lambda: 0
    ctx["valueFromContainer"] = lambda *a: None
    return ctx


SAFE_BUILTINS = {
    "min": min, "max": max, "round": round, "int": int, "float": float, "abs": abs,
    "sum": sum, "len": len, "any": any, "all": all, "str": str, "bool": bool,
    "True": True, "False": False, "None": None, "map": map, "list": list,
}


def evaluate(expr, ctx):
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return expr
    if not isinstance(expr, str):
        return None
    try:
        value = eval(expr, {"__builtins__": SAFE_BUILTINS}, ctx)  # trusted slicer resources only
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(value, 6) if isinstance(value, float) else value


def build_settings(settings, ctx):
    result = OrderedDict()
    for key, props in settings.items():
        stype = props.get("type")
        if stype in (None, "category"):
            continue
        entry = OrderedDict()
        entry["label"] = props.get("label")
        entry["type"] = stype
        if "unit" in props:
            entry["unit"] = props["unit"]
        entry["category"] = props.get("_category")
        entry["default"] = props.get("default_value")
        if "value" in props:
            entry["value_expr"] = props["value"]
        if "options" in props:
            entry["options"] = list(props["options"].keys())
        for src, dst in LIMIT_PROPS:
            if src in props:
                entry[dst] = evaluate(props[src], ctx)
                if not isinstance(props[src], (int, float)):
                    entry[dst + "_expr"] = props[src]
        entry["settable_per_extruder"] = props.get("settable_per_extruder", True)
        entry["settable_per_mesh"] = props.get("settable_per_mesh", True)
        if "enabled" in props and not isinstance(props["enabled"], bool):
            entry["enabled_expr"] = props["enabled"]
        result[key] = entry
    return result


def parse_quality_profiles(repo, default_layer_height):
    """Quality profiles; a profile without layer_height inherits the machine default."""
    qdir = os.path.join(repo, "resources", "quality")
    profiles = []
    for name in sorted(os.listdir(qdir)):
        if not name.endswith(".inst.cfg"):
            continue
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(os.path.join(qdir, name), encoding="utf-8")
        if parser.get("metadata", "type", fallback="") != "quality":
            continue
        values = OrderedDict()
        if parser.has_section("values"):
            for key, raw in parser.items("values"):
                values[key] = _parse_cfg_value(raw)
        if "layer_height" not in values:
            values["layer_height"] = default_layer_height
        profiles.append(OrderedDict([
            ("quality_type", parser.get("metadata", "quality_type")),
            ("name", parser.get("general", "name")),
            ("definition", parser.get("general", "definition", fallback=None)),
            ("weight", _parse_cfg_value(parser.get("metadata", "weight", fallback="0"))),
            ("layer_height", values.get("layer_height")),
            ("values", values),
        ]))
    profiles.sort(key=lambda p: (p["layer_height"] is None, p["layer_height"]))
    return profiles


def _parse_cfg_value(raw):
    raw = raw.strip()
    if raw.startswith("="):
        return raw  # formula, keep as text
    for conv in (int, float):
        try:
            return conv(raw)
        except ValueError:
            pass
    if raw in ("True", "False"):
        return raw == "True"
    return raw


def _text(node):
    if node is None or node.text is None:
        return None
    return " ".join(node.text.split())


def parse_materials(repo, diameter=1.75):
    mdir = os.path.join(repo, "resources", "materials")
    ns = {"um": "http://www.ultimaker.com/material"}
    materials = []
    for name in sorted(os.listdir(mdir)):
        if not name.endswith(".xml.fdm_material"):
            continue
        root = ET.parse(os.path.join(mdir, name)).getroot()
        meta = root.find("um:metadata", ns)
        props = root.find("um:properties", ns)
        dia = props.find("um:diameter", ns) if props is not None else None
        try:
            dia_value = float(_text(dia))
        except (TypeError, ValueError):
            dia_value = None
        if dia_value is None or abs(dia_value - diameter) > 0.05:
            continue
        density = props.find("um:density", ns)
        raw_settings = OrderedDict()
        mapped = OrderedDict()
        settings = root.find("um:settings", ns)
        if settings is not None:
            for entry in settings.findall("um:setting", ns):
                key = entry.get("key")
                value = _parse_cfg_value(entry.text or "")
                raw_settings[key] = value
                if key in MATERIAL_SETTING_MAP:
                    mapped[MATERIAL_SETTING_MAP[key]] = value
        mat_id = name[:-len(".xml.fdm_material")]
        materials.append(OrderedDict([
            ("id", mat_id),
            ("label", _text(meta.find("um:name/um:label", ns))),
            ("brand", _text(meta.find("um:name/um:brand", ns))),
            ("material", _text(meta.find("um:name/um:material", ns))),
            ("diameter", dia_value),
            ("density", float(_text(density)) if density is not None and _text(density) else None),
            ("print_temperature", mapped.get("default_material_print_temperature")),
            ("bed_temperature", mapped.get("default_material_bed_temperature")),
            ("description", _text(meta.find("um:description", ns))),
            ("adhesion_info", _text(meta.find("um:adhesion_info", ns))),
            ("settings", mapped),
            ("raw_settings", raw_settings),
        ]))
    return materials


def git_commit(repo):
    """Commit of the source tree, or None when it is not a git checkout (e.g. an unpacked archive)."""
    if not os.path.exists(os.path.join(repo, ".git")):
        return None  # git would report the commit of an enclosing repository instead
    try:
        out = subprocess.check_output(["git", "-C", repo, "log", "-1", "--format=%H %cI"])
        return out.decode("utf-8").strip()
    except Exception:
        return None


def value_in_mode(settings, key, mode):
    """Evaluate a setting's value expression with printing_mode fixed to `mode`."""
    props = settings[key]
    expr = props.get("value")
    if not isinstance(expr, str):
        return props.get("default_value")
    ctx = make_eval_context(settings)
    ctx["printing_mode"] = mode
    ctx["resolveOrValue"] = lambda k: mode if k == "printing_mode" else ctx[k]
    try:
        return eval(expr, {"__builtins__": SAFE_BUILTINS}, ctx)
    except Exception:
        return props.get("default_value")


def build(repo):
    chain = definition_chain(repo, MACHINE_ID)
    root = chain[-1]
    settings = flatten_settings(root["settings"])
    apply_overrides(settings, chain[:-1])
    # Extruder overrides (main extruder) — nozzle size, material diameter, ...
    ext_chain = definition_chain(repo, EXTRUDER_IDS[0])
    fdmextruder = ext_chain[-1]
    ext_settings = flatten_settings(fdmextruder["settings"])
    for key, props in ext_settings.items():
        settings.setdefault(key, props)
    apply_overrides(settings, ext_chain[:-1])

    ctx = make_eval_context(settings)
    catalog = OrderedDict()
    catalog["source"] = OrderedDict([
        ("repo", "STE Slicer source with the Hybrid 530 V5.2.4 definitions (base: https://github.com/stereotech/STE-Slicer)"),
        ("commit", git_commit(repo)),
        ("machine", MACHINE_ID),
        ("definition_chain", [d["_id"] for d in chain]),
        ("extruders", EXTRUDER_IDS),
    ])
    s = settings
    catalog["machine"] = OrderedDict([
        ("name", s["machine_name"]["default_value"]),
        ("width", s["machine_width"]["default_value"]),
        ("depth", s["machine_depth"]["default_value"]),
        ("height_classic", value_in_mode(s, "machine_height", "classic")),
        ("height_5d", value_in_mode(s, "machine_height", "cylindrical")),
        ("height_value_expr", s["machine_height"].get("value")),
        ("shape_classic", value_in_mode(s, "machine_shape", "classic")),
        ("shape_5d", value_in_mode(s, "machine_shape", "cylindrical")),
        ("center_is_zero_5d", bool(value_in_mode(s, "machine_center_is_zero", "cylindrical"))),
        ("heated_bed_5d", bool(value_in_mode(s, "machine_heated_bed", "cylindrical"))),
        ("nozzle_size", s["machine_nozzle_size"]["default_value"]),
        ("material_diameter", s["material_diameter"]["default_value"]),
        ("extruder_count", s["machine_extruder_count"]["default_value"]),
        ("five_axis", s["machine_five_axis"]["default_value"]),
        ("hybrid", s["machine_hybrid"]["default_value"]),
    ])
    mode_options = s["printing_mode"].get("options", {})
    catalog["printing_modes"] = OrderedDict((m, mode_options.get(m, m)) for m in USED_MODES)
    catalog["quality_profiles"] = parse_quality_profiles(repo, s["layer_height"]["default_value"])
    catalog["materials"] = parse_materials(repo)
    catalog["settings"] = build_settings(settings, ctx)
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    catalog = build(args.repo)
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with io.open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps(catalog, ensure_ascii=False, indent=1))
        f.write(u"\n")
    print("settings: %d, quality profiles: %d, materials (1.75 mm): %d -> %s" % (
        len(catalog["settings"]), len(catalog["quality_profiles"]), len(catalog["materials"]), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
