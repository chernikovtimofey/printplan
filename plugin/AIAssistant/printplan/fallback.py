"""Deterministic plan used without an LLM or when its answers keep failing
validation: mode candidate, its recommended orientation, the user's material
or the top of the KG ranking, Θ0, and a templated explanation."""
from collections import OrderedDict

MODE_NAMES = OrderedDict([("classic", "3D (classic)"), ("cylindrical", "5D Spiral"),
                          ("cylindrical_full", "5D Spiral Full")])


def choose_material(parse, E_KG, allowed_materials):
    user = parse["constraints"].get("material")
    if user:
        return user, "материал задан пользователем"
    for entry in E_KG["ranking"]:
        if entry["id"] in allowed_materials:
            chain = entry["chains"][0] if entry["chains"] else ""
            return entry["id"], "первый в рейтинге графа знаний (%.2f)%s" % (
                entry["score"], ": " + chain if chain else "")
    return "proto_pla", "рейтинг пуст — выбран самый простой в печати материал"


def build_plan(parse, E_G, E_KG, ctx, allowed_materials, reason=None):
    cand = E_G["mode_candidate"]
    mode = cand["candidate"]
    item = E_G["modes"][mode]
    material, material_reason = choose_material(parse, E_KG, allowed_materials)

    plan = OrderedDict()
    risks = [OrderedDict([("severity", r["severity"]), ("text", r["text"]), ("evidence", r["evidence"])])
             for r in item["risks"]]
    for conflict in E_KG["conflicts"]:
        risks.append(OrderedDict([("severity", "medium"), ("text", conflict["text"]), ("evidence", "граф материалов")]))
    high = any(r["severity"] == "high" for r in risks)
    fits = item["orientation"].get("fits_build_volume", True)
    plan["verdict"] = "not_printable" if not fits else ("printable_with_tuning" if high or risks else "printable")
    plan["mode"] = mode
    plan["quality_type"] = item["quality_type"]
    plan["material"] = material
    plan["orientation"] = OrderedDict([("candidate_id", item["orientation"]["id"]),
                                       ("reason", item["orientation_reason"])])
    plan["settings"] = [OrderedDict([("key", k), ("value", v["value"]), ("reason", v["reason"])])
                        for k, v in item["theta0"].items() if k != "printing_mode"]
    plan["risks"] = risks[:8]
    plan["design_suggestions"] = list(item["design_hints"])
    plan["deviations"] = []
    plan["explanation"] = explanation(parse, E_G, E_KG, mode, item, material, material_reason, reason)
    questions = []
    if cand["confidence"] == "low":
        questions.append("Выбор между 3D и 5D неочевиден (оценка %.1f при пороге %.1f). Подтвердите режим печати." % (
            cand["score"], cand["threshold"]))
    if mode == "cylindrical":
        questions.append("Есть ли стержень-основание Ø%s мм для 5D Spiral?" % cand.get("spiral_base_diameter_mm"))
    plan["questions_to_user"] = questions
    return plan


def explanation(parse, E_G, E_KG, mode, item, material, material_reason, reason=None):
    cand = E_G["mode_candidate"]
    parts = []
    if reason:
        parts.append("План построен по правилам без языковой модели (%s)." % reason)
    if cand["source"] == "user":
        parts.append("Режим %s задан пользователем." % MODE_NAMES[mode])
    else:
        top = sorted(cand["evidence"], key=lambda e: -abs(e["weight"]))[:3]
        why = "; ".join(e["text"] for e in top) if top else "у детали нет оси вращения"
        parts.append("Режим %s: %s (оценка %.1f при пороге %.1f)." % (MODE_NAMES[mode], why, cand["score"], cand["threshold"]))
    parts.append("Ориентация %s: %s." % (item["orientation"]["id"], item["orientation_reason"]))
    parts.append("Материал %s: %s." % (material, material_reason))
    q = [x for x in E_KG["conflicts"]]
    if q:
        parts.append("Внимание: " + "; ".join(c["text"] for c in q[:3]) + ".")
    parts.append("Профиль %s: %s." % (item["quality_type"], item["quality_reason"]))
    key_settings = [(k, v) for k, v in item["theta0"].items() if k != "printing_mode"][:5]
    if key_settings:
        parts.append("Настройки: " + "; ".join("%s = %s (%s)" % (k, v["value"], v["reason"]) for k, v in key_settings) + ".")
    main_risks = [r for r in item["risks"] if r["severity"] in ("high", "medium")][:3]
    if main_risks:
        parts.append("Риски: " + "; ".join("%s (%s)" % (r["text"], r["evidence"]) for r in main_risks) + ".")
    return " ".join(parts)
