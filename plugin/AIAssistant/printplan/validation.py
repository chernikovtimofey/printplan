"""Stage 7: validation of the plan returned by the LLM.

Checks:
1. the JSON matches the schema;
2. mode, quality profile and material exist in the slicer (and are admissible);
3. setting keys are known and values lie within the limits (SlicerContext);
4. explicit user constraints C are respected;
5. decisions are consistent with each other and with the evidence
   (supports vs. overhangs, 5D keys vs. mode, deviations are justified).

Errors make the pipeline ask the model again; warnings are reported only.
"""
from collections import OrderedDict

from . import schema
from .rules import SUPPORT_RISK, MODES_5D
from .slicer_context import coerce_value


def normalize_plan(args, ctx):
    """Coerce setting values to their types, drop a redundant printing_mode and keep
    only the last value of a key given twice."""
    plan = OrderedDict(args)
    settings, invalid = OrderedDict(), []
    for item in plan.get("settings") or []:
        if not isinstance(item, dict) or "key" not in item:
            invalid.append(item)
            continue
        key = item["key"]
        if key == "printing_mode":
            continue
        entry = OrderedDict(item)
        entry["value"] = coerce_value(item.get("value"), ctx.setting(key))
        settings.pop(key, None)
        settings[key] = entry
    plan["settings"] = list(settings.values()) + invalid
    for key in ("risks", "design_suggestions", "questions_to_user", "deviations"):
        if plan.get(key) is None:
            plan[key] = []
    return plan


def final_settings(plan, E_G):
    """Θ0 of the chosen mode overridden by the plan's settings (each with its reason)."""
    out = OrderedDict()
    mode = plan.get("mode")
    theta0 = E_G["modes"].get(mode, {}).get("theta0", {})
    for key, item in theta0.items():
        out[key] = OrderedDict([("value", item["value"]), ("reason", item["reason"]), ("source", "rules")])
    for item in plan.get("settings") or []:
        if isinstance(item, dict) and "key" in item:
            key = item["key"]
            same = key in theta0 and theta0[key]["value"] == item.get("value")
            out[key] = OrderedDict([("value", item.get("value")),
                                    ("reason", item.get("reason") or (theta0[key]["reason"] if same else "")),
                                    ("source", "rules" if same else "plan")])
    out["printing_mode"] = OrderedDict([("value", mode), ("reason", "режим печати"), ("source", "plan")])
    out.move_to_end("printing_mode", last=False)
    return out


def recommended_orientation_ids(E_G, mode):
    item = E_G["modes"].get(mode)
    if not item:
        return []
    if mode in MODES_5D:
        return [o["id"] for o in item.get("orientation_options", [])]
    ids = [o["id"] for o in E_G["orientations_3d"]]
    if item["orientation"]["id"] not in ids:
        ids.append(item["orientation"]["id"])
    return ids


def validate_plan(plan, parse, E_G, E_KG, ctx, allowed_materials, parameters):
    errors, warnings = [], []
    errors.extend(schema.validate(plan, parameters))
    if errors:
        return errors, warnings

    mode = plan["mode"]
    c = parse["constraints"]
    if mode not in E_G["modes"]:
        errors.append("Режим %s недопустим для этой детали; допустимы: %s" % (mode, list(E_G["modes"].keys())))
        return errors, warnings
    if plan["quality_type"] not in ctx.quality_types():
        errors.append("Профиля качества %s нет в слайсере" % plan["quality_type"])
    if plan["material"] not in allowed_materials:
        errors.append("Материал %s не из списка допустимых" % plan["material"])
    if plan["material"] not in ctx.material_ids():
        errors.append("Материала %s нет в слайсере" % plan["material"])

    oid = plan["orientation"]["candidate_id"]
    valid_ids = recommended_orientation_ids(E_G, mode)
    if oid not in valid_ids:
        errors.append("Ориентация %s не подходит для режима %s; допустимы: %s" % (oid, mode, valid_ids))

    settings = final_settings(plan, E_G)
    values = OrderedDict((k, v["value"]) for k, v in settings.items())
    for issue in ctx.validate_settings(values):
        (errors if issue["level"] == "error" else warnings).append(issue["text"])
    if mode == "classic":
        for key in values:
            if "cylindrical" in key:
                errors.append("Ключ 5D «%s» в режиме classic" % key)

    # 4. explicit constraints
    if c.get("supports") == "forbidden":
        for key in ("support_enable", "support_enable_cylindrical", "support_tree_enable"):
            if values.get(key) is True:
                errors.append("Пользователь запретил поддержки, а %s = true" % key)
    if c.get("material") and plan["material"] != c["material"]:
        errors.append("Пользователь задал материал %s, план выбрал %s" % (c["material"], plan["material"]))
    if c.get("mode") in ("classic", "cylindrical", "cylindrical_full") and mode != c["mode"]:
        errors.append("Пользователь задал режим %s, план выбрал %s" % (c["mode"], mode))
    if c.get("mode") == "5d_any" and mode not in MODES_5D:
        errors.append("Пользователь просил 5D, план выбрал %s" % mode)
    lh = c.get("layer_height_mm")
    if lh:
        key = "layer_height" if mode == "classic" else "cylindrical_layer_height"
        got = values.get(key, values.get("layer_height", ctx.quality_layer_height(plan["quality_type"])))
        if got is None or abs(float(got) - lh) > 0.005:
            errors.append("Пользователь задал высоту слоя %.2f мм, в плане %s = %s" % (lh, key, got))

    # 5. consistency with the evidence
    if mode == "classic" and oid in valid_ids and c.get("supports") != "forbidden":
        row = next((o for o in E_G["orientations_3d"] if o["id"] == oid), None) or \
            (E_G["modes"]["classic"]["orientation"] if E_G["modes"]["classic"]["orientation"]["id"] == oid else None)
        if row and row["R_support"] > SUPPORT_RISK and values.get("support_enable") is not True:
            errors.append("В ориентации %s R_support = %.3f > %.2f, но поддержки выключены" % (
                oid, row["R_support"], SUPPORT_RISK))
    if mode in MODES_5D:
        r5d = E_G["mode_candidate"]["features"].get("R_support_5d", 0) or 0
        if r5d > SUPPORT_RISK and c.get("supports") != "forbidden" and values.get("support_enable_cylindrical") is not True:
            warnings.append("В 5D доля поверхностей, обращённых к оси, %.3f, а поддержки 5D выключены" % r5d)
    excluded = dict((e["id"], e["reasons"]) for e in E_KG["excluded"])
    if plan["material"] in excluded and plan["material"] != c.get("material"):
        errors.append("Материал %s не проходит требования задачи: %s" % (plan["material"], "; ".join(excluded[plan["material"]])))

    deviations = set(d.get("what") for d in plan.get("deviations", []) if isinstance(d, dict) and d.get("reason"))
    cand = E_G["mode_candidate"]
    if mode != cand["candidate"] and cand["source"] != "user" and "mode" not in deviations:
        errors.append("Режим %s отличается от кандидата %s — нужна запись в deviations с обоснованием" % (
            mode, cand["candidate"]))
    rec = E_G["modes"][mode]
    if oid != rec["orientation"]["id"] and "orientation" not in deviations:
        errors.append("Ориентация %s отличается от рекомендованной %s — нужна запись в deviations" % (
            oid, rec["orientation"]["id"]))
    top = E_KG["ranking"][0]["id"] if E_KG["ranking"] else None
    if top and plan["material"] not in (top, c.get("material")) and "material" not in deviations:
        errors.append("Материал %s не первый в рейтинге (%s) — нужна запись в deviations" % (plan["material"], top))
    if plan["quality_type"] != rec["quality_type"] and "quality" not in deviations:
        errors.append("Профиль %s отличается от базового %s — нужна запись в deviations" % (
            plan["quality_type"], rec["quality_type"]))
    if mode in MODES_5D:
        kg_mat = E_KG.get("user_material") if c.get("material") == plan["material"] else None
        mat_entry = kg_mat or next((r for r in E_KG["ranking"] if r["id"] == plan["material"]), None)
        if mat_entry and mat_entry.get("penalties", {}).get("mode"):
            warnings.append("Материал %s плохо подходит для 5D" % plan["material"])
    if not plan.get("explanation", "").strip():
        errors.append("Пустое объяснение")
    return errors, warnings
