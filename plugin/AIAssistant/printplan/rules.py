"""Stage 4: heuristic rules.

Turns the metrics of stages 2–3 (plus the rotational features) and the task
flags into:

* the printing-mode candidate with weighted evidence (MODE_WEIGHTS and
  MODE_THRESHOLD are hand-calibrated and should be re-checked on new parts);
* the chosen orientation for that mode;
* risks R and design hints D;
* the baseline quality profile and slicer settings Θ0 (with a reason for every
  value), for every mode the LLM may still choose.
"""
from collections import OrderedDict

import numpy as np

from . import orientation as orient

# --------------------------------------------------------------------------- thresholds

SUPPORT_RISK = 0.05          # R_support above this: supports needed
BASE_RISK = 0.3              # R_base below this: weak adhesion
SLENDER_RISK = 4.0           # R_slender above this: tipping
AXIS_VERTICAL_MARGIN = 0.1   # accept an axis-vertical orientation if S_o is within this of the best
AXIS_VERTICAL_MIN_BASE = 0.05
TIE_MARGIN = 0.02            # S_o differences below this are treated as a tie
TIE_SUPPORT_MARGIN = 0.01    # R_support differences below this do not matter in a tie

# --------------------------------------------------------------------------- 3D / 5D evidence

MODE_THRESHOLD = 2.0
MODE_WEIGHTS = OrderedDict([
    ("body_of_revolution", 3.0),     # body of revolution -> 5D
    ("cyclic_symmetry", 1.5),        # impellers, wheels with spokes
    ("supports_unavoidable_3d", 1.5),
    ("supports_saved_5d", 1.5),      # radial (5D) support need lower than the best 3D one
    ("torsion", 2.0),                # shafts: torsion shears flat layers apart
    ("surface_no_layer_steps", 1.0),
    ("load_on_revolution", 0.5),
    ("flexible_material", -4.0),     # soft elastomers: printed in 3D (to confirm with Stereotech)
    ("ring_without_core", -5.0),     # bore > half the diameter: nothing to wind the layers on
    ("supports_added_5d", -0.5),
    ("flat_support_free", -1.0),     # 2.5D disc printable flat without supports (gears)
])
HARD_CONTRAINDICATIONS = ("flexible_material", "ring_without_core")  # 5D is not offered to the LLM then
RING_BORE_RATIO = 0.5
SPIRAL_MIN_BORE_MM = 4.0             # minimum non_printing_base_diameter of the slicer
DEFAULT_BASE_DIAMETER = 6.0

MODES_5D = ("cylindrical", "cylindrical_full")


def _flag(parse, name):
    item = parse["flags"].get(name)
    return float(item["strength"]) if item else 0.0


def _confirmed_flag(parse, name):
    """Flag strength for the mode decision: wording confirmed by the lexicon counts fully,
    a flag inferred only by the LLM counts half (the mode must rest on explicit text)."""
    item = parse["flags"].get(name)
    if not item:
        return 0.0
    factor = 0.5 if item.get("source") == "llm" else 1.0
    return float(item["strength"]) * factor


def _evidence(term, text, factor=1.0):
    return OrderedDict([("term", term), ("weight", round(MODE_WEIGHTS[term] * factor, 3)), ("text", text)])


def choose_mode(parse, mesh, rot, rank3d, machine, flexible_material=False):
    """Mode candidate with the evidence that produced it."""
    out = OrderedDict()
    explicit = parse["constraints"].get("mode", "unspecified")
    axis = rot.selected
    evidence = []
    info = OrderedDict()
    feasible_5d = axis is not None
    r5d = None
    base_radius = DEFAULT_BASE_DIAMETER / 2.0
    bore_d = None
    if axis is not None:
        bore_d = (axis.bore or {}).get("diameter_mm")
        if bore_d and bore_d >= SPIRAL_MIN_BORE_MM:
            base_radius = bore_d / 2.0
        r5d = orient.radial_support_ratio(mesh, axis.point, axis.direction, base_radius)
        info["R_support_5d"] = round(r5d, 4)
        axis_opts = orient.axis_orientations(mesh, axis.point, axis.direction, machine)
        if not any(o.fits for o in axis_opts):
            feasible_5d = False
            info["5d_fit"] = "модель не помещается в область 5D (высота ≤ %s мм, радиус ≤ %s мм)" % (
                machine["height_5d"], min(machine["width"], machine["depth"]) / 2.0) if machine else "не помещается"
    r3d = rank3d["min_R_support"]
    info["R_support_3d_min"] = r3d

    score = 0.0
    if axis is None:
        info["reason_no_5d"] = "нет оси вращения: деталь не тело вращения и не имеет соосных цилиндрических поверхностей"
    else:
        if axis.body_of_revolution:
            evidence.append(_evidence("body_of_revolution", "тело вращения (индекс осевой симметрии %.5f)" % axis.continuous_index))
        elif axis.cyclic_order >= 3:
            evidence.append(_evidence("cyclic_symmetry", "циклическая симметрия порядка %d вокруг оси %s" % (
                axis.cyclic_order, axis.label)))
        if r3d > SUPPORT_RISK:
            evidence.append(_evidence("supports_unavoidable_3d",
                                      "в 3D поддержки нужны в любой ориентации (мин. R_support = %.3f)" % r3d))
        if r5d is not None and r3d > orient.SUPPORT_FREE_MAX and r5d < r3d:
            factor = min(1.0, (r3d - r5d) / 0.02)
            evidence.append(_evidence("supports_saved_5d", "в 5D доля поверхности под поддержки %.3f против %.3f в 3D" % (
                r5d, r3d), factor))
        if r5d is not None and r5d > r3d:
            factor = min(1.0, (r5d - r3d) / 0.1)
            evidence.append(_evidence("supports_added_5d", "в 5D поддержек больше: %.3f против %.3f в 3D" % (r5d, r3d), factor))
        torsion = _confirmed_flag(parse, "torsion")
        if torsion >= 0.5:
            evidence.append(_evidence("torsion", "кручение вала: в 3D срез идёт по межслойным границам", torsion))
        appearance = _confirmed_flag(parse, "appearance")
        if appearance >= 0.5:
            evidence.append(_evidence("surface_no_layer_steps",
                                      "требование к поверхности: в 5D нет ступенек слоёв на поверхностях вокруг оси", appearance))
        load = _confirmed_flag(parse, "load")
        if load >= 0.5 and axis.body_of_revolution:
            evidence.append(_evidence("load_on_revolution", "нагрузка на тело вращения: прочность поперёк слоёв", load))
        flex = _confirmed_flag(parse, "flexible")
        if flex >= 0.7 or flexible_material:
            evidence.append(_evidence("flexible_material", "нужен мягкий эластичный материал — печать в 3D"))
        if bore_d and axis.outer_diameter > 0 and bore_d / axis.outer_diameter > RING_BORE_RATIO:
            evidence.append(_evidence("ring_without_core", "кольцо: отверстие Ø%.1f мм больше половины диаметра %.1f мм" % (
                bore_d, axis.outer_diameter)))
        axial_ratio = axis.axial_length / max(axis.outer_diameter, 1e-6)
        if axis.prismatic_fraction >= 0.7 and axial_ratio <= 0.5 and r3d <= orient.SUPPORT_FREE_MAX:
            evidence.append(_evidence("flat_support_free",
                                      "плоская деталь печатается в 3D без поддержек (призматичность %.2f)" % axis.prismatic_fraction))
        score = sum(e["weight"] for e in evidence)

    candidate = "classic"
    if feasible_5d and score >= MODE_THRESHOLD:
        candidate = _spiral_or_full(bore_d)
    source = "rules"
    if explicit in ("classic", "cylindrical", "cylindrical_full"):
        candidate, source = explicit, "user"
    elif explicit == "5d_any":
        candidate, source = _spiral_or_full(bore_d), "user"
    margin = abs(score - MODE_THRESHOLD)
    out["candidate"] = candidate
    out["source"] = source
    out["score"] = round(score, 3)
    out["threshold"] = MODE_THRESHOLD
    out["confidence"] = "user" if source == "user" else ("high" if margin >= 1.5 else ("medium" if margin >= 0.5 else "low"))
    out["feasible_5d"] = bool(feasible_5d)
    out["evidence"] = evidence
    out["features"] = info
    out["spiral_base_diameter_mm"] = _spiral_base(bore_d)
    out["spiral_or_full_reason"] = (
        "сквозное осевое отверстие Ø%.1f мм: 5D Spiral на стержне-основании этого диаметра" % bore_d
        if bore_d and bore_d >= SPIRAL_MIN_BORE_MM else
        "нет сквозного осевого отверстия ≥ %.0f мм: 5D Spiral Full (центр печатается плоскими слоями)" % SPIRAL_MIN_BORE_MM)
    return out


def _spiral_or_full(bore_d):
    return "cylindrical" if bore_d and bore_d >= SPIRAL_MIN_BORE_MM else "cylindrical_full"


def _spiral_base(bore_d):
    if bore_d and bore_d >= SPIRAL_MIN_BORE_MM:
        return float(np.floor(bore_d * 10.0) / 10.0)
    return None


# --------------------------------------------------------------------------- orientation

def choose_orientation(mode, mesh, rot, rank3d, machine, supports_forbidden, appearance=0.0):
    """Return (Orientation, reason, table_5d)."""
    if mode in MODES_5D and rot.selected is not None:
        axis = rot.selected
        opts = orient.axis_orientations(mesh, axis.point, axis.direction, machine)
        best = opts[0]
        reason = ("5D: ось вращения (%s) вертикально через центр стола; вниз обращён торец с меньшей оценкой S_o "
                  "(%.3f против %.3f)" % (axis.label, opts[0].score, opts[1].score))
        return best, reason, opts
    best = rank3d["best"]
    reason = "минимальная оценка S_o = %.3f" % best.score
    best, tie_reason = _break_tie(best, rank3d, appearance, supports_forbidden)
    if tie_reason:
        reason = tie_reason
    if supports_forbidden:
        if best.metrics["R_support"] <= orient.SUPPORT_FREE_MAX:
            reason += "; поддержки запрещены — выбрана ориентация без нависаний"
        else:
            reason += "; поддержки запрещены, но без нависаний ориентации нет — выбрана с минимальной R_support"
    axis = rot.selected
    if axis is not None and (axis.body_of_revolution or axis.cyclic_order >= 3):
        alt = _axis_vertical_alternative(mesh, axis, machine)
        ok = alt is not None and alt.metrics["R_base"] >= AXIS_VERTICAL_MIN_BASE \
            and alt.score <= best.score + AXIS_VERTICAL_MARGIN \
            and (not supports_forbidden or alt.metrics["R_support"] <= orient.SUPPORT_FREE_MAX
                 or alt.metrics["R_support"] <= best.metrics["R_support"])
        if ok and abs(float(np.dot(alt.up, best.up))) < 0.99:
            reason = ("ось вращения вертикально: круглые сечения печатаются точно, деталь стоит на торце "
                      "(R_base = %.2f); S_o = %.3f при минимуме %.3f" % (alt.metrics["R_base"], alt.score, best.score))
            return alt, reason, []
    return best, reason, []


def _break_tie(best, rank3d, appearance, supports_forbidden):
    """Among orientations within TIE_MARGIN of the best S_o prefer fewer supports when the
    surface matters, otherwise the model's own orientation (+z, no rotation)."""
    if supports_forbidden:
        return best, None
    near = [o for o in rank3d["table"] if o.fits and o.score <= best.score + TIE_MARGIN]
    if len(near) < 2:
        return best, None
    if appearance >= 0.5:
        least = min(o.metrics["R_support"] for o in near)
        fewest = [o for o in near if o.metrics["R_support"] <= least + TIE_SUPPORT_MARGIN]
        pick = next((o for o in fewest if o.id == "+z"), None) or min(fewest, key=lambda o: o.score)
        if pick is not best and pick.metrics["R_support"] < best.metrics["R_support"]:
            return pick, ("оценки S_o почти равны (%.3f и %.3f); при приоритете внешнего вида выбрана ориентация "
                          "с меньшей площадью поддержек (R_support %.3f против %.3f)" % (
                              pick.score, best.score, pick.metrics["R_support"], best.metrics["R_support"]))
        return best, None
    native = next((o for o in near if o.id == "+z"), None)
    if native is not None and native is not best:
        return native, ("оценки S_o почти равны (%.3f и %.3f); сохранена исходная ориентация модели" % (
            native.score, best.score))
    return best, None


def _axis_vertical_alternative(mesh, axis, machine):
    cands = []
    for sign in (1.0, -1.0):
        up = sign * axis.direction
        label = _canonical_label(up)
        o = orient.Orientation(label or ("axis_up%s" % ("+" if sign > 0 else "-")), up, "axis_up" if label else "rotation_axis")
        orient.evaluate(mesh, o)
        o.fits = orient._fits_classic(o, machine)
        orient._center_on_bed(o)
        if o.fits:
            cands.append(o)
    if not cands:
        return None
    return min(cands, key=lambda o: o.score)


def _canonical_label(up):
    for label, vec in orient.CANONICAL.items():
        if float(np.dot(up, vec)) > 0.9999:
            return label
    return None


# --------------------------------------------------------------------------- quality

def baseline_quality(parse, mesh, mode):
    c = parse["constraints"]
    if c.get("quality") == "draft" or _flag(parse, "speed") >= 0.6:
        return "draft", "просьба печатать быстро / черновик"
    reasons = []
    if c.get("quality") == "fine":
        reasons.append("явное требование к качеству поверхности")
    if _flag(parse, "appearance") >= 0.6:
        reasons.append("приоритет внешнего вида")
    if _flag(parse, "fit") >= 0.6:
        reasons.append("точность посадочных размеров")
    if reasons:
        return "normal", "; ".join(reasons) + " — слой 0.1 мм"
    if _flag(parse, "flexible") >= 0.7:
        return "fast", "эластичный материал: слой 0.15 мм, без лишних ретрактов"
    if float(mesh.extents.max()) > 200:
        return "draft", "крупная деталь без требований к поверхности — слой 0.2 мм"
    return "fast", "функциональная деталь без особых требований к поверхности — слой 0.15 мм"


# --------------------------------------------------------------------------- settings Θ0

def _set(theta, key, value, reason):
    theta[key] = OrderedDict([("value", value), ("reason", reason)])


def baseline_settings(mode, parse, mesh, chosen, rank3d, mode_info, material_id=None, catalog=None):
    """Θ0 for one mode: OrderedDict key -> {value, reason}."""
    theta = OrderedDict()
    supports_forbidden = parse["constraints"].get("supports") == "forbidden"
    m = chosen.metrics
    _set(theta, "printing_mode", mode, "режим печати")
    user_lh = parse["constraints"].get("layer_height_mm")
    if user_lh:
        _set(theta, "layer_height", user_lh, "высота слоя задана пользователем")
        if mode != "classic":
            _set(theta, "cylindrical_layer_height", user_lh, "высота слоя задана пользователем")

    if mode == "classic":
        if supports_forbidden:
            _set(theta, "support_enable", False, "пользователь запретил поддержки")
        elif m["R_support"] > SUPPORT_RISK:
            _set(theta, "support_enable", True, "R_support = %.3f > %.2f" % (m["R_support"], SUPPORT_RISK))
            if _flag(parse, "appearance") >= 0.5:
                _set(theta, "support_type", "buildplate", "внешний вид: поддержки только от стола")
            else:
                _set(theta, "support_type", "everywhere", "нависания есть и над самой моделью")
        else:
            _set(theta, "support_enable", False, "R_support = %.3f — поддержки не нужны" % m["R_support"])
        if m["R_base"] < BASE_RISK or m["R_slender"] > SLENDER_RISK:
            _set(theta, "adhesion_type", "brim", "R_base = %.2f, R_slender = %.2f — кайма против отрыва" % (
                m["R_base"], m["R_slender"]))
        if m["R_slender"] > SLENDER_RISK:
            _set(theta, "speed_print", 40, "высокая деталь (R_slender = %.1f): ниже скорость" % m["R_slender"])
    else:
        r5d = mode_info["features"].get("R_support_5d", 0.0)
        if supports_forbidden:
            _set(theta, "support_enable", False, "пользователь запретил поддержки")
            _set(theta, "support_enable_cylindrical", False, "пользователь запретил поддержки")
        elif r5d > SUPPORT_RISK:
            _set(theta, "support_enable", True, "в 5D-части есть поверхности, обращённые к оси (%.3f)" % r5d)
            _set(theta, "support_enable_cylindrical", True, "поддержки 5D-части")
        else:
            _set(theta, "support_enable", False, "в 5D поддержки не нужны (доля %.3f)" % r5d)
            _set(theta, "support_enable_cylindrical", False, "в 5D поддержки не нужны")
        base = mode_info.get("spiral_base_diameter_mm")
        if mode == "cylindrical" and base:
            _set(theta, "cylindrical_mode_base_diameter", base, "диаметр осевого отверстия детали")
            _set(theta, "non_printing_base_diameter", base, "стержень-основание входит в осевое отверстие")

    speed_key = "speed_wall_0" if mode == "classic" else "speed_wall_0_cylindrical"
    if _flag(parse, "fit") >= 0.6:
        _set(theta, speed_key, 20, "посадка: медленнее внешний периметр для точности")
        if mode == "classic":
            _set(theta, "xy_offset_layer_0", -0.1, "посадка: компенсация расширения первого слоя")
    load = max(_flag(parse, "load"), _flag(parse, "torsion"))
    flexible = _flag(parse, "flexible") >= 0.7
    if flexible:
        _set(theta, "speed_print", 25, "эластичный материал: низкая скорость печати")
        _set(theta, "retraction_amount", 1.5, "эластичный материал: минимальные ретракты")
    elif load >= 0.5:
        _set(theta, "wall_line_count", 4, "нагрузка: больше стенок")
        density = 50 if load >= 0.9 or _flag(parse, "fatigue") >= 0.7 else 40
        _set(theta, "infill_sparse_density", density, "нагрузка: плотнее заполнение")
        _set(theta, "infill_pattern", "gyroid", "изотропное заполнение для нагруженной детали")
    elif _flag(parse, "fit") >= 0.6:
        _set(theta, "wall_line_count", 3, "посадка: жёстче стенки")

    if material_id and catalog:
        mat = next((x for x in catalog.get("materials", []) if x["id"] == material_id), None)
        speed = mat and mat.get("raw_settings", {}).get("speed print")
        if speed and "speed_print" not in theta:
            _set(theta, "speed_print", speed, "рекомендация профиля материала Stereotech")
    q = mesh.Q_topo
    if q["boundary_edges"]:
        _set(theta, "meshfix_extensive_stitching", True, "в сетке %d граничных рёбер (дыры)" % q["boundary_edges"])
    if q["nonmanifold_edges"] or q["inconsistent_edges"]:
        _set(theta, "meshfix_union_all", True, "немногообразные рёбра или перевёрнутые нормали")
    return theta


# --------------------------------------------------------------------------- risks and hints

def _risk(rid, severity, text, evidence, tags=()):
    return OrderedDict([("id", rid), ("severity", severity), ("text", text), ("evidence", evidence),
                        ("tags", list(tags))])


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
FLAG_RISK_TAGS = OrderedDict([
    ("appearance", ("supports", "surface")),
    ("fit", ("accuracy", "stability")),
    ("heat", ("material",)),
    ("load", ("layers",)),
    ("torsion", ("layers",)),
])


def risks_and_hints(mode, parse, mesh, rot, rank3d, chosen, mode_info, machine):
    R, D = [], []
    supports_forbidden = parse["constraints"].get("supports") == "forbidden"
    q = mesh.Q_topo
    f = mesh.F_mesh
    m = chosen.metrics
    if f.get("units_warning"):
        u = f["units_warning"]
        R.append(_risk("units", "high", "Похоже, модель сохранена не в миллиметрах (наибольший габарит %s мм)" % u["max_extent_mm"],
                       "предлагаемые масштабы: %s" % u["suggested_scales"]))
    if q["boundary_edges"]:
        R.append(_risk("mesh_holes", "high", "В сетке есть дыры: возможны ошибки слайсинга и пропуски слоёв",
                       "граничных рёбер: %d" % q["boundary_edges"]))
        D.append("Исправить сетку в CAD: закрыть отверстия в поверхности")
    if q["nonmanifold_edges"] or q["inconsistent_edges"]:
        R.append(_risk("mesh_nonmanifold", "medium", "Немногообразные рёбра или перевёрнутые нормали",
                       "немногообразных: %d, несогласованных: %d" % (q["nonmanifold_edges"], q["inconsistent_edges"])))
        D.append("Исправить сетку в CAD: убрать касания тел по ребру и внутренние грани")
    if q["degenerate_triangles"]:
        R.append(_risk("mesh_degenerate", "low", "Вырожденные треугольники — признак некачественного экспорта, обычно безвредны",
                       "вырожденных: %d" % q["degenerate_triangles"]))
    if q["shells"] > 1:
        R.append(_risk("shells", "low", "Модель состоит из нескольких оболочек (тел или полостей)",
                       "оболочек: %d" % q["shells"]))
        D.append("Проверить, задумано ли несколько отдельных тел")
    if not chosen.fits:
        R.append(_risk("build_volume", "high", "Модель не помещается в рабочую область в выбранном режиме",
                       "габариты %s мм" % m["size_xyz_mm"]))
        D.append("Уменьшить модель или разрезать её на части")

    if mode == "classic":
        if m["R_support"] > SUPPORT_RISK:
            if supports_forbidden:
                R.append(_risk("supports_forbidden_unavoidable", "high",
                               "Без поддержек напечатать не получится: нависания есть в любой ориентации",
                               "мин. R_support = %.3f" % rank3d["min_R_support"], ("supports",)))
            else:
                R.append(_risk("supports", "medium", "Нужны поддержки — на поверхности останутся следы",
                               "R_support = %.3f в ориентации %s" % (m["R_support"], chosen.id), ("supports", "surface")))
            D.append("Заменить горизонтальные нависания фасками под 45°")
        if m["R_base"] < BASE_RISK:
            R.append(_risk("adhesion", "medium", "Малая площадь опоры — возможен отрыв от стола",
                           "R_base = %.2f в ориентации %s" % (m["R_base"], chosen.id), ("stability",)))
            D.append("Добавить плоскую опорную площадку")
        if m["R_slender"] > SLENDER_RISK:
            R.append(_risk("tipping", "medium", "Высокая узкая деталь: опрокидывание и раскачивание",
                           "R_slender = %.1f" % m["R_slender"], ("stability",)))
            D.append("Расширить основание или печатать деталь лёжа")
        if max(_flag(parse, "load"), _flag(parse, "torsion")) >= 0.5:
            R.append(_risk("layer_direction", "medium" if _flag(parse, "torsion") >= 0.5 else "low",
                           "FDM-деталь слабее всего на отрыв слоёв: нагрузка не должна разрывать межслойные границы",
                           "слои перпендикулярны направлению %s модели" % _axis_word(chosen.up), ("layers",)))
        if mode_info.get("feasible_5d") and mode_info["score"] >= MODE_THRESHOLD - 1.0 and mode_info["source"] != "user":
            D.append("Деталь близка к критериям 5D (оценка %.1f при пороге %.1f): рассмотрите 5D-печать" % (
                mode_info["score"], MODE_THRESHOLD))
    else:
        axis = rot.selected
        base = mode_info.get("spiral_base_diameter_mm")
        if mode == "cylindrical":
            R.append(_risk("rod", "medium", "Для 5D Spiral нужен стержень-основание Ø%.1f мм по осевому отверстию" % (base or DEFAULT_BASE_DIAMETER),
                           mode_info.get("spiral_or_full_reason", "")))
        else:
            R.append(_risk("core", "low", "Центральный цилиндр Ø%.0f мм печатается плоскими слоями, остальное — цилиндрическими" % DEFAULT_BASE_DIAMETER,
                           "5D Spiral Full"))
        r5d = mode_info["features"].get("R_support_5d")
        if r5d is not None and r5d > SUPPORT_RISK:
            R.append(_risk("supports_5d", "medium" if not supports_forbidden else "high",
                           "В 5D-части есть поверхности, обращённые к оси: нужны поддержки 5D",
                           "доля %.3f" % r5d, ("supports",)))
        R.append(_risk("arrange_5d", "low", "В режимах 5D деталь ставится на ось вращения в центре стола; автоматическую расстановку применять нельзя",
                       "ось %s" % (axis.label if axis else "?")))

    tags = []
    for flag, ftags in FLAG_RISK_TAGS.items():
        if _flag(parse, flag) >= 0.5:
            tags.extend(ftags)
    R.sort(key=lambda r: (SEVERITY_ORDER[r["severity"]], 0 if set(r["tags"]) & set(tags) else 1))
    return R, D


def _axis_word(up):
    label = _canonical_label(up)
    return ("оси %s" % label.upper()) if label else "выбранной вертикали"


# --------------------------------------------------------------------------- evidence package

def material_is_flexible(kg, material_id):
    if not material_id:
        return False
    mat = kg.materials.get(material_id)
    return bool(mat and mat.get("attributes", {}).get("flexible"))


def build(parse, mesh, rot, rank3d, machine, kg=None, catalog=None, material_hint=None):
    """Run all stage-4 rules. Returns (E_G dict, context dict with live objects)."""
    supports_forbidden = parse["constraints"].get("supports") == "forbidden"
    user_mat = parse["constraints"].get("material")
    flexible_material = material_is_flexible(kg, user_mat) if kg else False
    mode_info = choose_mode(parse, mesh, rot, rank3d, machine, flexible_material)
    candidate = mode_info["candidate"]
    modes = [candidate]
    hard_contra = any(e["term"] in HARD_CONTRAINDICATIONS for e in mode_info["evidence"])
    if mode_info["feasible_5d"] and not (hard_contra and mode_info["source"] != "user"):
        for alt in ("classic", "cylindrical_full", "cylindrical"):
            if alt not in modes and not (alt == "cylindrical" and not mode_info.get("spiral_base_diameter_mm")):
                modes.append(alt)
    elif candidate != "classic":
        modes.append("classic")

    per_mode = OrderedDict()
    for mode in modes:
        chosen, reason, table5d = choose_orientation(mode, mesh, rot, rank3d, machine, supports_forbidden,
                                                     _flag(parse, "appearance"))
        quality, quality_reason = baseline_quality(parse, mesh, mode)
        theta = baseline_settings(mode, parse, mesh, chosen, rank3d, mode_info, material_hint, catalog)
        risks, hints = risks_and_hints(mode, parse, mesh, rot, rank3d, chosen, mode_info, machine)
        per_mode[mode] = OrderedDict([
            ("orientation", chosen), ("orientation_reason", reason), ("table5d", table5d),
            ("quality_type", quality), ("quality_reason", quality_reason),
            ("theta0", theta), ("risks", risks), ("design_hints", hints),
        ])

    E_G = OrderedDict()
    E_G["mesh"] = mesh.F_mesh
    E_G["topology"] = mesh.Q_topo
    E_G["orientations_3d"] = [o.as_dict() for o in rank3d["table"]]
    E_G["best_3d_orientation"] = rank3d["best"].id
    E_G["supports_unavoidable_3d"] = rank3d["supports_unavoidable"]
    E_G["rotation"] = rot.as_dict()
    E_G["mode_candidate"] = mode_info
    E_G["modes"] = OrderedDict()
    for mode, item in per_mode.items():
        entry = OrderedDict()
        entry["orientation"] = item["orientation"].as_dict()
        entry["orientation_reason"] = item["orientation_reason"]
        if item["table5d"]:
            entry["orientation_options"] = [o.as_dict() for o in item["table5d"]]
        entry["quality_type"] = item["quality_type"]
        entry["quality_reason"] = item["quality_reason"]
        entry["theta0"] = item["theta0"]
        entry["risks"] = item["risks"]
        entry["design_hints"] = item["design_hints"]
        E_G["modes"][mode] = entry
    context = OrderedDict([("mode_info", mode_info), ("per_mode", per_mode)])
    return E_G, context
