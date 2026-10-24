"""Stage 5: Task–Material–Process graph and material ranking.

Ranking is an algorithm, not an LLM decision:

1. Activation: flags F and applications from stage 1 activate application
   nodes (alpha_a) and property nodes (alpha_p).
2. Filtering: materials that cannot satisfy hard requirements are excluded
   (density from the slicer profile above the limit, soft vs rigid, conductive
   filler when a dielectric is required, far too low service temperature).
3. Scoring (evidence normalised by the total activation):

       s(m) = (sum alpha_a w_am + sum alpha_p w_pm) / (sum alpha_a + sum alpha_p)
              - s_diff(m) - s_printer(m) - s_requirements(m) - s_mode(m)

   s_requirements grows with how far the material misses numeric requirements
   (temperature, modulus, elongation, water absorption, ...).
4. Ranking with evidence chains ("нагрев 85 °C → теплостойкость → PP G30").

A material fixed by the user is never replaced; a conflict is produced instead.
"""
import math
from collections import OrderedDict

from .io_utils import read_json_resource

MAX_NOZZLE_WARNING_C = 285.0
CONFLICT_MARGIN = 0.1
RIGID_MODULUS_MPA = 1000.0   # below this a "rigid" requirement is penalised
SOFT_FILTER_PENALTY = 0.3    # instead of exclusion when the flag comes from the LLM only


class KnowledgeGraph(object):
    def __init__(self, data=None, catalog=None):
        self.data = data or read_json_resource("kg.json")
        self.catalog = catalog
        self._catalog_materials = {}
        nozzle_warn = MAX_NOZZLE_WARNING_C
        if catalog:
            self._catalog_materials = dict((m["id"], m) for m in catalog.get("materials", []))
            setting = catalog.get("settings", {}).get("default_material_print_temperature", {})
            nozzle_warn = setting.get("max_warning") or nozzle_warn
        self.nozzle_warning_c = nozzle_warn

    # ------------------------------------------------------------------ nodes

    @property
    def materials(self):
        return self.data["materials"]

    def material_label(self, mat_id):
        mat = self.materials.get(mat_id)
        if mat:
            return mat["label"]
        cat = self._catalog_materials.get(mat_id)
        return cat["label"] if cat else mat_id

    def density(self, mat_id):
        cat = self._catalog_materials.get(mat_id)
        if cat and cat.get("density"):
            return float(cat["density"])
        return None

    def print_temperature(self, mat_id):
        cat = self._catalog_materials.get(mat_id)
        return cat.get("print_temperature") if cat else None

    # ------------------------------------------------------------------ activation

    def activations(self, parse):
        """alpha for applications and properties, each with the flags that activated it."""
        apps = OrderedDict()
        for app, item in parse.get("applications", {}).items():
            if app in self.data["applications"]:
                apps[app] = OrderedDict([("alpha", float(item["strength"])), ("via", "назначение")])
        props = OrderedDict()
        for flag, item in parse.get("flags", {}).items():
            spec = self.data["flags"].get(flag)
            if not spec:
                continue
            for prop, weight in spec["properties"].items():
                alpha = float(item["strength"]) * float(weight)
                if prop == "heat":
                    alpha = max(alpha, _heat_alpha(parse.get("numbers", {})))
                prev = props.get(prop)
                if prev is None or alpha > prev["alpha"]:
                    props[prop] = OrderedDict([("alpha", round(alpha, 3)), ("via", flag)])
        return apps, props

    # ------------------------------------------------------------------ ranking

    def rank(self, parse, mode="classic", top_k=5):
        apps, props = self.activations(parse)
        numbers = dict((k, v["value"]) for k, v in parse.get("numbers", {}).items())
        flags = dict((k, float(v["strength"])) for k, v in parse.get("flags", {}).items())
        # Hard exclusions need wording confirmed by the text; LLM-only flags only penalise.
        confirmed = set(k for k, v in parse.get("flags", {}).items() if v.get("source") != "llm")
        user_mat = parse.get("constraints", {}).get("material")
        total_alpha = sum(a["alpha"] for a in apps.values()) + sum(p["alpha"] for p in props.values())

        scored, excluded = [], []
        for mat_id, mat in self.materials.items():
            entry = self._score(mat_id, mat, apps, props, total_alpha, numbers, flags, mode, confirmed)
            if entry["excluded"]:
                excluded.append(OrderedDict([("id", mat_id), ("label", mat["label"]),
                                             ("reasons", entry["hard_failures"])]))
            scored.append(entry)
        ranked = sorted([e for e in scored if not e["excluded"]], key=lambda e: -e["score"])
        leader = ranked[0] if ranked else None

        result = OrderedDict()
        result["activated"] = OrderedDict([
            ("applications", apps), ("properties", props)])
        result["ranking"] = [_public(e) for e in ranked[:top_k] if e["score"] > 0 or e is leader]
        result["excluded"] = excluded
        result["user_material"] = None
        result["conflicts"] = []
        if not ranked:
            nearest = sorted(scored, key=lambda e: (len(e["hard_failures"]), -e["score"]))[:3]
            result["conflicts"].append(OrderedDict([
                ("material", None),
                ("text", "Ни один материал не проходит все жёсткие требования; ближайшие: " +
                 ", ".join("%s (%s)" % (e["label"], "; ".join(e["hard_failures"])) for e in nearest)),
            ]))
            result["ranking"] = [_public(e) for e in nearest]
        if user_mat:
            entry = next((e for e in scored if e["id"] == user_mat), None)
            if entry is None:
                entry = self._unknown_material_entry(user_mat)
            result["user_material"] = _public(entry)
            result["conflicts"].extend(self._conflicts(entry, leader, props))
        return result

    def _score(self, mat_id, mat, apps, props, total_alpha, numbers, flags, mode, confirmed=None):
        attrs = mat.get("attributes", {})
        positive, contributions = 0.0, []
        for app, a in apps.items():
            w = float(mat.get("applications", {}).get(app, 0.0))
            positive += a["alpha"] * w
            if w > 0:
                contributions.append((a["alpha"] * w, "назначение «%s» → %s (w=%.2f)" % (
                    self.data["applications"][app]["label"], mat["label"], w)))
        for prop, p in props.items():
            w = float(mat.get("properties", {}).get(prop, 0.0))
            positive += p["alpha"] * w
            if w > 0:
                contributions.append((p["alpha"] * w, "%s → %s → %s (w=%.2f)" % (
                    _flag_label(self.data, p["via"], numbers), self.data["properties"][prop]["label"],
                    mat["label"], w)))
        positive = positive / total_alpha if total_alpha > 0 else 0.5

        checks, hard, req_penalty = self._requirements(mat_id, mat, attrs, numbers, flags, confirmed)
        printer_penalty, printer_notes = self._printer(mat_id, mode)
        mode_penalty = 0.3 if mode in ("cylindrical", "cylindrical_full") and mat.get("mode_5d") == "poor" else 0.0
        difficulty = float(mat.get("difficulty", 0.0))
        score = positive - difficulty - printer_penalty - req_penalty - mode_penalty

        contributions.sort(key=lambda x: -x[0])
        e = OrderedDict()
        e["id"] = mat_id
        e["label"] = mat["label"]
        e["score"] = round(score, 4)
        e["positive"] = round(positive, 4)
        e["penalties"] = OrderedDict([
            ("difficulty", round(difficulty, 3)), ("printer", round(printer_penalty, 3)),
            ("requirements", round(req_penalty, 3)), ("mode", round(mode_penalty, 3))])
        e["chains"] = [text for _, text in contributions[:4]]
        e["checks"] = checks
        e["difficulty_reasons"] = mat.get("difficulty_reasons", [])
        e["printer_notes"] = printer_notes
        e["hard_failures"] = hard
        e["excluded"] = bool(hard)
        e["_props"] = mat.get("properties", {})
        return e

    def _requirements(self, mat_id, mat, attrs, numbers, flags, confirmed=None):
        checks, hard = [], []
        penalty = 0.0

        def flag_hard(name):
            return confirmed is None or name in confirmed

        def check(key, ok, text, pen=0.0, is_hard=False):
            status = "ok" if ok else ("fail" if is_hard else "miss")
            checks.append(OrderedDict([("requirement", key), ("status", status), ("text", text)]))
            if not ok:
                if is_hard:
                    hard.append(text)
                return pen
            return 0.0

        svc = attrs.get("service_temp_c")
        soft = attrs.get("softening_temp_c")
        if "temp_max_c" in numbers and svc is not None:
            t = numbers["temp_max_c"]
            diff = t - svc
            penalty += check("temp_max_c", diff <= 0,
                             "рабочая %g °C, длительная теплостойкость материала ≈%g °C" % (t, svc),
                             min(0.6, diff / 40.0), diff > 25)
        if "heat_resistance_min_c" in numbers and soft is not None:
            t = numbers["heat_resistance_min_c"]
            diff = t - soft
            penalty += check("heat_resistance_min_c", diff <= 0,
                             "требуется теплостойкость ≥%g °C, у материала ≈%g °C" % (t, soft),
                             min(0.6, diff / 40.0), diff > 40)
        if "short_term_temp_c" in numbers and soft is not None:
            t = numbers["short_term_temp_c"]
            diff = t - soft
            penalty += check("short_term_temp_c", diff <= 0,
                             "кратковременно %g °C, размягчение материала ≈%g °C" % (t, soft),
                             min(0.3, diff / 100.0))
        density = self.density(mat_id)
        if "density_max_g_cm3" in numbers and density is not None:
            limit = numbers["density_max_g_cm3"]
            penalty += check("density_max_g_cm3", density <= limit + 0.005,
                             "плотность %.2f г/см³ (профиль слайсера), предел %.2f" % (density, limit),
                             0.0, True)
        for key, attr, label, scale, cap in (
                ("modulus_min_mpa", "modulus_mpa", "модуль упругости", 0.6, 0.5),
                ("tensile_min_mpa", "tensile_mpa", "предел прочности", 0.4, 0.3),
                ("elongation_min_pct", "elongation_pct", "удлинение при разрыве", 0.4, 0.3)):
            if key in numbers and attrs.get(attr) is not None:
                need, have = numbers[key], float(attrs[attr])
                rel = (need - have) / need if need > 0 else 0.0
                penalty += check(key, have >= need, "%s ≈%g, требуется ≥%g" % (label, have, need),
                                 min(cap, rel * scale))
        if "water_absorption_max_pct" in numbers and attrs.get("water_absorption_pct") is not None:
            need, have = numbers["water_absorption_max_pct"], float(attrs["water_absorption_pct"])
            pen = min(0.4, 0.15 * math.log10(max(have / need, 1.0))) if need > 0 else 0.4
            penalty += check("water_absorption_max_pct", have <= need,
                             "водопоглощение ≈%g %%, требуется ≤%g %%" % (have, need), pen)

        filler = attrs.get("filler", "none")
        if flags.get("dielectric", 0) >= 0.5:
            penalty += check("dielectric", filler != "carbon" and not attrs.get("conductive"),
                             "требуется диэлектрик, наполнитель: %s" % filler, SOFT_FILTER_PENALTY,
                             flag_hard("dielectric"))
        if flags.get("flexible", 0) >= 0.7:
            flexible_enough = attrs.get("flexible") or mat.get("properties", {}).get("flexibility", 0) >= 0.5
            penalty += check("flexible", flexible_enough, "требуется эластичный материал", SOFT_FILTER_PENALTY,
                             flag_hard("flexible"))
        if flags.get("rigid", 0) >= 0.7:
            penalty += check("rigid", not attrs.get("flexible"), "требуется жёсткий материал", SOFT_FILTER_PENALTY,
                             flag_hard("rigid"))
            modulus = attrs.get("modulus_mpa")
            if not attrs.get("flexible") and modulus is not None and modulus < RIGID_MODULUS_MPA:
                penalty += check("rigid_modulus", False, "модуль упругости ≈%g МПа — деталь будет податливой" % modulus,
                                 0.25 * flags["rigid"])
        if flags.get("mating_gentle", 0) > 0 and filler != "none":
            penalty += check("mating_gentle", False,
                             "абразивный наполнитель (%s) изнашивает ответную деталь" % filler,
                             0.15 * flags["mating_gentle"])
        if flags.get("food_contact", 0) >= 0.7 and mat.get("properties", {}).get("food_safe", 0) < 0.5:
            penalty += check("food_contact", False, "нет допуска к контакту с пищей", 0.1)
        return checks, hard, penalty

    def _printer(self, mat_id, mode):
        notes, penalty = [], 0.0
        t = self.print_temperature(mat_id)
        if t is not None and t > self.nozzle_warning_c:
            penalty += 0.05
            notes.append("температура сопла %g °C выше порога предупреждения слайсера %g °C" % (
                t, self.nozzle_warning_c))
        return penalty, notes

    def _unknown_material_entry(self, mat_id):
        e = OrderedDict([("id", mat_id), ("label", self.material_label(mat_id)), ("score", None),
                         ("positive", None), ("penalties", OrderedDict()), ("chains", []), ("checks", []),
                         ("difficulty_reasons", []), ("printer_notes", []),
                         ("hard_failures", ["материал отсутствует в графе знаний"]), ("excluded", False),
                         ("_props", {})])
        return e

    def _conflicts(self, entry, leader, props):
        out = []
        label = entry["label"]
        for text in entry["hard_failures"]:
            out.append(OrderedDict([("material", entry["id"]), ("text", "%s: %s" % (label, text))]))
        for c in entry["checks"]:
            if c["status"] == "miss":
                out.append(OrderedDict([("material", entry["id"]), ("text", "%s: %s" % (label, c["text"]))]))
        for prop, p in props.items():
            if p["alpha"] >= 0.7 and float(entry["_props"].get(prop, 1.0)) < 0.3:
                out.append(OrderedDict([("material", entry["id"]), ("text", "%s: слабое свойство «%s», важное для задачи" % (
                    label, self.data["properties"][prop]["label"]))]))
        if leader is not None and entry["score"] is not None and leader["id"] != entry["id"] \
                and leader["score"] - entry["score"] > CONFLICT_MARGIN:
            out.append(OrderedDict([("material", entry["id"]), ("text", "%s заметно уступает лидеру рейтинга %s (%.2f против %.2f)" % (
                label, leader["label"], entry["score"], leader["score"]))]))
        return out

    # ------------------------------------------------------------------ review

    def report(self):
        """Plain-text table of material nodes for manual review."""
        lines = []
        props = list(self.data["properties"].keys())
        for mat_id, mat in self.materials.items():
            attrs = mat.get("attributes", {})
            lines.append("%s — %s [%s], reviewed=%s" % (mat_id, mat["label"], mat.get("family"),
                                                      mat.get("reviewed", self.data.get("reviewed"))))
            lines.append("  density (slicer): %s; %s" % (self.density(mat_id), ", ".join(
                "%s=%s" % (k, v) for k, v in attrs.items())))
            lines.append("  applications: " + ", ".join("%s=%.2f" % kv for kv in mat.get("applications", {}).items()))
            lines.append("  properties: " + ", ".join("%s=%.2f" % (p, mat["properties"].get(p, 0)) for p in props))
            lines.append("  difficulty=%.2f %s; 5D: %s" % (mat.get("difficulty", 0), mat.get("difficulty_reasons", []),
                                                         mat.get("mode_5d")))
        return "\n".join(lines)


def _heat_alpha(numbers):
    temps = [numbers[k]["value"] for k in ("temp_max_c", "heat_resistance_min_c") if k in numbers]
    if not temps:
        return 0.0
    return max(0.0, min(1.0, (max(temps) - 40.0) / 50.0))


def _flag_label(data, flag, numbers):
    spec = data["flags"].get(flag)
    label = spec["label"] if spec else flag
    if flag == "heat":
        temps = [numbers[k] for k in ("temp_max_c", "heat_resistance_min_c") if k in numbers]
        if temps:
            label = "%s %g °C" % (label, max(temps))
    return label


def _public(entry):
    return OrderedDict((k, v) for k, v in entry.items() if not k.startswith("_") and k != "excluded")
