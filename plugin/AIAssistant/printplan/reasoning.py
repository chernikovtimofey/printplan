"""Stage 6: LLM reasoning over the evidence packages.

The model receives the task, C/F, the compact geometry package E_G (including
the full orientation table and Θ0 for every admissible mode), the material
package E_KG and the slicer context, and must answer with one forced call of
`submit_print_plan` whose parameters follow schema.plan_schema().
"""
import json
from collections import OrderedDict

from . import schema
from .llm_client import stage_options

TOOL_NAME = "submit_print_plan"
MAX_TASK_CHARS = 6000

# Keys the model is offered (with limits); validation accepts any known non-machine key.
OFFERED_KEYS = [
    "layer_height", "layer_height_0", "cylindrical_layer_height", "cylindrical_layer_height_0",
    "support_enable", "support_enable_cylindrical", "support_type", "support_angle", "support_tree_enable",
    "adhesion_type", "brim_width", "meshfix_union_all", "meshfix_extensive_stitching",
    "wall_line_count", "top_bottom_thickness", "infill_sparse_density", "infill_pattern",
    "speed_print", "speed_wall_0", "speed_wall_0_cylindrical", "xy_offset", "xy_offset_layer_0",
    "retraction_enable", "retraction_amount", "cool_fan_enabled", "cool_fan_speed",
    "cool_fan_enabled_cylindrical", "bridge_settings_enabled",
    "cylindrical_mode_base_diameter", "non_printing_base_diameter", "cylindrical_mode_overlap",
    "cylindrical_raft_type", "material_print_temperature", "material_bed_temperature",
]

SYSTEM_PROMPT = """Ты — инженер-технолог аддитивного производства, помощник в слайсере STE Slicer для принтера Stereotech Hybrid 530 V5.2.4 (режимы 3D classic, 5D Spiral = cylindrical, 5D Spiral Full = cylindrical_full).

Тебе передают задание пользователя и готовые факты, вычисленные кодом: анализ сетки, таблицу ориентаций, правила с рисками и базовыми настройками Θ0 для каждого допустимого режима, кандидата режима с доводами, рейтинг материалов по графу знаний с цепочками обоснований и конфликтами, а также списки того, что есть в слайсере. Верни план печати одним вызовом функции submit_print_plan.

Порядок рассуждения:
1. Пойми задачу: назначение детали, нагрузки, среда, требования к поверхности и точности.
2. Оцени геометрические риски по фактам (R_support, R_base, топология, ось вращения).
3. Сопоставь их с данными графа материалов (цепочки, проверки требований, конфликты).
4. Сравни кандидатов: режим, ориентации, материалы, профиль.
5. Найди баланс между печатаемостью и соответствием задаче.
6. Сформируй рекомендации.

Правила:
- Режим, профиль (quality_type), материал и ориентацию (candidate_id) выбирай только из переданных списков. Ориентация берётся из таблицы выбранного режима.
- Числа не придумывай. Настройки — это Θ0 выбранного режима плюс поправки в допустимых пределах (setting_limits); у каждой настройки есть причина. Ключ printing_mode не указывай — он задаётся полем mode.
- Явные ограничения пользователя (constraints) не нарушай. Если пользователь задал материал, сохрани его, а конфликты из пакета материалов обязательно доведи до пользователя в risks и explanation.
- Отказ от кандидата режима, от ориентации, рекомендованной для режима, от первого материала рейтинга или от базового профиля допустим только с записью в deviations, где причина ссылается на данные.
- В режимах 5D меняй 5D-ключи (с суффиксом _cylindrical, cylindrical_layer_height); автоматическую расстановку в 5D не применяют.
- Не придумывай фактов: свойства и числовые характеристики материалов, температуры, требования к камере и т. п. — только из переданных данных; evidence рисков ссылается на переданные факты. Материалы называй их названиями (label) из данных.
- Объяснение (explanation) пиши по-русски для начинающего пользователя, 5–10 предложений: режим и почему, ориентация, материал (с цепочкой обоснования), профиль и ключевые настройки, главные риски. Опирайся на факты: числа, риски, цепочки графа.
- verdict: printable — можно печатать как есть; printable_with_tuning — печатать с учётом рисков и настроек; not_printable — печать невозможна (например, модель не помещается).
- Если не хватает важных сведений, заполни questions_to_user, но план всё равно построй по наиболее вероятному предположению.
- Пиши кратко: причина у настройки — до 15 слов; risks — до 6 пунктов; design_suggestions — до 4; reason ориентации — 1–2 предложения."""


def _compact_orientation(o):
    keys = ("id", "kind", "R_support", "R_base", "R_slender", "R_height", "S_o", "size_xyz_mm", "fits_build_volume")
    return OrderedDict((k, o[k]) for k in keys if k in o)


def build_evidence(task_text, parse, E_G, E_KG, ctx, allowed_materials):
    ev = OrderedDict()
    text = task_text if len(task_text) <= MAX_TASK_CHARS else task_text[:MAX_TASK_CHARS] + "…"
    ev["task_text"] = text
    ev["task_analysis"] = OrderedDict([
        ("part_name", parse.get("part_name")),
        ("summary", parse.get("summary")),
        ("constraints", parse["constraints"]),
        ("flags", OrderedDict((k, v["strength"]) for k, v in parse["flags"].items())),
        ("numbers", OrderedDict((k, v["value"]) for k, v in parse["numbers"].items())),
        ("applications", list(parse["applications"].keys())),
    ])
    mesh = E_G["mesh"]
    topo = E_G["topology"]
    geo = OrderedDict()
    geo["mesh"] = OrderedDict([
        ("extents_mm", mesh["extents_mm"]), ("volume_cm3", mesh["volume_cm3"]),
        ("surface_area_mm2", mesh["surface_area_mm2"]), ("watertight", topo["watertight"]),
        ("shells", topo["shells"]), ("boundary_edges", topo["boundary_edges"]),
        ("nonmanifold_edges", topo["nonmanifold_edges"]), ("units_warning", mesh.get("units_warning")),
    ])
    geo["orientations_3d"] = [_compact_orientation(o) for o in E_G["orientations_3d"]]
    geo["supports_unavoidable_3d"] = E_G["supports_unavoidable_3d"]
    rot = E_G["rotation"]
    if rot.get("has_axis"):
        a = rot["axis"]
        geo["rotation_axis"] = OrderedDict([
            ("kind", rot["axis_kind"]), ("model_axis", a["label"]), ("cyclic_order", a["cyclic_order"]),
            ("through_bore", a["through_bore"]), ("prismatic_fraction", a["prismatic_fraction"]),
            ("axial_length_mm", a["axial_length_mm"]), ("outer_diameter_mm", a["outer_diameter_mm"]),
        ])
    else:
        geo["rotation_axis"] = None
    mc = E_G["mode_candidate"]
    geo["mode_candidate"] = OrderedDict([
        ("candidate", mc["candidate"]), ("source", mc["source"]), ("score", mc["score"]),
        ("threshold", mc["threshold"]), ("confidence", mc["confidence"]),
        ("evidence", ["%+.1f %s" % (e["weight"], e["text"]) for e in mc["evidence"]]),
        ("spiral_or_full", mc["spiral_or_full_reason"]),
        ("features", mc["features"]),
    ])
    modes = OrderedDict()
    for mode, item in E_G["modes"].items():
        m = OrderedDict()
        m["recommended_orientation"] = item["orientation"]["id"]
        m["orientation_reason"] = item["orientation_reason"]
        if item.get("orientation_options"):
            m["orientation_options"] = [_compact_orientation(o) for o in item["orientation_options"]]
        m["quality_type"] = item["quality_type"]
        m["quality_reason"] = item["quality_reason"]
        m["theta0"] = OrderedDict((k, v["value"]) for k, v in item["theta0"].items() if k != "printing_mode")
        m["theta0_reasons"] = OrderedDict((k, v["reason"]) for k, v in item["theta0"].items() if k != "printing_mode")
        m["risks"] = [OrderedDict([("severity", r["severity"]), ("text", r["text"]), ("evidence", r["evidence"])])
                      for r in item["risks"]]
        m["design_hints"] = item["design_hints"]
        modes[mode] = m
    geo["modes"] = modes
    ev["geometry"] = geo

    mats = OrderedDict()
    mats["ranking"] = [OrderedDict([
        ("id", r["id"]), ("label", r["label"]), ("score", r["score"]), ("chains", r["chains"][:3]),
        ("requirement_issues", [c["text"] for c in r["checks"] if c["status"] != "ok"]),
        ("difficulty", r["difficulty_reasons"]),
    ]) for r in E_KG["ranking"]]
    if E_KG.get("user_material"):
        u = E_KG["user_material"]
        mats["user_material"] = OrderedDict([("id", u["id"]), ("label", u["label"]), ("score", u["score"])])
    mats["conflicts"] = [c["text"] for c in E_KG["conflicts"]]
    mats["excluded_count"] = len(E_KG["excluded"])
    ev["materials"] = mats

    sl = OrderedDict()
    sl["modes"] = list(E_G["modes"].keys())
    sl["quality_profiles"] = [OrderedDict([("quality_type", q["quality_type"]), ("ui_name", q["name"]),
                                           ("layer_height", q["layer_height"])]) for q in ctx.quality_profiles()]
    sl["materials_allowed"] = allowed_materials
    limits = OrderedDict()
    for key in OFFERED_KEYS:
        spec = ctx.setting(key)
        if not spec:
            continue
        limits[key] = OrderedDict((k, spec[k]) for k in ("type", "min", "max", "min_warning", "max_warning", "options")
                                  if spec.get(k) is not None)
    sl["setting_limits"] = limits
    ev["slicer"] = sl
    return ev


def orientation_ids(E_G):
    ids = []
    for mode, item in E_G["modes"].items():
        if item.get("orientation_options"):
            ids.extend(o["id"] for o in item["orientation_options"])
        ids.append(item["orientation"]["id"])
    ids.extend(o["id"] for o in E_G["orientations_3d"])
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def build_messages(evidence, images=None):
    content = ("Данные для плана печати (JSON):\n" + json.dumps(evidence, ensure_ascii=False, separators=(",", ":")) +
               "\n\nВызови submit_print_plan.")
    if images:
        parts = [{"type": "text", "text": content}]
        for data_url in images:
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        user = {"role": "user", "content": parts}
    else:
        user = {"role": "user", "content": content}
    return [{"role": "system", "content": SYSTEM_PROMPT}, user]


def retry_messages(messages, previous_args, errors):
    feedback = ("Ответ не прошёл проверку. Ошибки:\n- " + "\n- ".join(errors) +
                "\n\nИсправь их, сохранив остальное, и снова вызови submit_print_plan.")
    return list(messages) + [
        {"role": "assistant", "content": json.dumps(previous_args, ensure_ascii=False)},
        {"role": "user", "content": feedback},
    ]


def tool_parameters(E_G, ctx, allowed_materials):
    return schema.plan_schema(list(E_G["modes"].keys()), ctx.quality_types(), allowed_materials, orientation_ids(E_G))


def ask(llm, messages, parameters, on_progress=None):
    options = stage_options(getattr(llm, "model", None), 6)
    return llm.call_tool(messages, TOOL_NAME, "Итоговый план печати", parameters, extra_body=options,
                         on_progress=on_progress)
