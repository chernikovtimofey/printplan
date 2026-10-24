"""Stage 1: task text -> explicit constraints C, task flags F, numeric requirements.

Two extractors produce the same structure:

* `parse_with_llm` — one forced tool call with enum-only fields (primary);
* `parse_with_lexicon` — regular expressions from data/lexicon.json plus number
  extraction (fallback and cross-check).

`parse_task` runs both and merges them (`merge`). The merge keeps the LLM
result as primary, adds lexicon-only flags with strength 0.5, fills constraints
and numbers the LLM left empty, and records disagreements.
"""
import re
from collections import OrderedDict

from .io_utils import read_json_resource
from .llm_client import stage_options

_KG = None
_LEX = None


def _kg():
    global _KG
    if _KG is None:
        _KG = read_json_resource("kg.json")
    return _KG


def _lex():
    global _LEX
    if _LEX is None:
        _LEX = read_json_resource("lexicon.json")
    return _LEX


def flag_ids():
    return list(_kg()["flags"].keys())


def application_ids():
    return list(_kg()["applications"].keys())


def material_ids():
    return list(_kg()["materials"].keys())


NUMBER_KEYS = OrderedDict([
    ("temp_max_c", "максимальная рабочая (длительная) температура, °C"),
    ("temp_min_c", "минимальная рабочая температура, °C"),
    ("heat_resistance_min_c", "требуемая теплостойкость материала (Tg, Вика, HDT, «не размягчаться при»), °C"),
    ("short_term_temp_c", "кратковременный нагрев (капли, касание), °C"),
    ("density_max_g_cm3", "максимальная плотность материала, г/см³"),
    ("modulus_min_mpa", "минимальный модуль упругости, МПа"),
    ("tensile_min_mpa", "минимальный предел прочности при растяжении, МПа"),
    ("elongation_min_pct", "минимальное относительное удлинение при разрыве, %"),
    ("water_absorption_max_pct", "максимальное водопоглощение, %"),
    ("tolerance_mm", "наименьший допуск размеров, мм"),
])
SUPPORTS = ["forbidden", "allowed", "unspecified"]
MODES = ["classic", "cylindrical", "cylindrical_full", "5d_any", "unspecified"]
QUALITY = ["draft", "balanced", "fine", "unspecified"]


# ----------------------------------------------------------------------------- normalisation

def normalize(text):
    text = (text or "").lower().replace("ё", "е")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace(" ", " ").replace(" ", " ")
    return re.sub(r"[ \t]+", " ", text)


# A period ends a sentence only before whitespace and not inside "1.5" or "...".
_SENTENCE_SPLIT = re.compile(r"(?<![.\d])\.(?![.\d])(?=\s|$)|[!?;]|\n")


def sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT.split(normalize(text)) if s.strip()]


# ----------------------------------------------------------------------------- empty result

def empty_parse():
    p = OrderedDict()
    p["constraints"] = OrderedDict([
        ("material", None), ("supports", "unspecified"), ("mode", "unspecified"),
        ("quality", "unspecified"), ("layer_height_mm", None),
    ])
    p["flags"] = OrderedDict()
    p["applications"] = OrderedDict()
    p["numbers"] = OrderedDict()
    p["part_name"] = None
    p["summary"] = None
    p["discrepancies"] = []
    p["source"] = None
    return p


# ----------------------------------------------------------------------------- lexicon

def _negated(sentence, start, end):
    before = sentence[max(0, start - 40):start]
    after = sentence[end:end + 40]
    if re.search(r"(\bне\b|\bбез\b)\s*(\w+\s+){0,3}$", before):
        return True
    return bool(re.search(r"^\w*\s*(\w+\s+){0,2}(не\s+допуска|недопуст|исключ|запрещ)", after))


def _find(patterns, sents):
    """Return list of (sentence, match) for non-negated matches."""
    hits = []
    for entry in patterns:
        rx = re.compile(entry["re"] if isinstance(entry, dict) else entry)
        negatable = isinstance(entry, dict) and entry.get("negatable")
        for s in sents:
            for m in rx.finditer(s):
                if negatable and _negated(s, m.start(), m.end()):
                    continue
                hits.append((s, m.group(0)))
    return hits


def _strength(hit_count):
    return {0: 0.0, 1: 0.6, 2: 0.8}.get(hit_count, 1.0)


def _snippet(sentence, limit=120):
    return sentence if len(sentence) <= limit else sentence[:limit - 1] + "…"


def parse_with_lexicon(text):
    lex = _lex()
    sents = sentences(text)
    result = empty_parse()
    result["source"] = "lexicon"

    for flag, patterns in lex["flags"].items():
        hits = _find(patterns, sents)
        if hits:
            distinct = max(len(set(s for s, _ in hits)), len(set(m for _, m in hits)))
            result["flags"][flag] = OrderedDict([
                ("strength", _strength(distinct)), ("evidence", _snippet(hits[0][0])), ("source", "lexicon")])
    for app, patterns in lex["applications"].items():
        hits = _find(patterns, sents)
        if hits:
            result["applications"][app] = OrderedDict([
                ("strength", _strength(len(set(s for s, _ in hits)))), ("evidence", _snippet(hits[0][0])),
                ("source", "lexicon")])

    c = result["constraints"]
    c["material"] = _explicit_material(sents, lex)
    for value in ("forbidden", "allowed"):
        if _find(lex["supports"][value], sents):
            c["supports"] = value
            break
    for value in ("cylindrical_full", "cylindrical", "5d_any", "classic"):
        if _find(lex["modes"][value], sents):
            c["mode"] = value
            break
    for value in ("draft", "fine"):
        if _find(lex["quality"][value], sents):
            c["quality"] = value
            break
    lh = re.search(r"(?:высот\w*\s+сло\w*|сло\w*)\s*(?:-|:)?\s*(0[.,]\d+)\s*мм", normalize(text))
    if lh:
        c["layer_height_mm"] = float(lh.group(1).replace(",", "."))

    result["numbers"] = extract_numbers(text)
    _derive_flags_from_numbers(result)
    resolve_contradictions(result)
    return result


def _explicit_material(sents, lex):
    """A material named as the print material (not the material of the replaced part)."""
    skip = re.compile(lex["material_context_skip"])
    found = []
    for mat_id, patterns in lex["materials"].items():
        for pattern in patterns:
            rx = re.compile(pattern)
            for s in sents:
                for m in rx.finditer(s):
                    if skip.search(s[max(0, m.start() - 40):m.start()]):
                        continue
                    found.append((m.start(), len(m.group(0)), mat_id))
    if not found:
        return None
    # Prefer the most specific (longest) alias, then the first mention.
    found.sort(key=lambda x: (-x[1], x[0]))
    return found[0][2]


# ----------------------------------------------------------------------------- numbers

_TEMP_RE = re.compile(r"(?<![\d.,])([+-]?\d{1,3}(?:[.,]\d+)?)\s*°\s*[cс]")
_HEAT_CTX = re.compile(r"стеклован|\btg\b|вика|размягч|теплостойк|термостойк|\bhdt\b|не\s+ниже")
_SHORT_CTX = re.compile(r"кратковрем|капани|капл|попадани\w*\s+горяч|расплав")
_PRINT_CTX = re.compile(r"сопл|температур\w*\s+печат|стол\w*\s+(нагре|температ)")


def _num(text):
    return float(text.replace(",", ".").replace(" ", ""))


def extract_numbers(text):
    numbers = OrderedDict()

    def put(key, value, sentence, mode):
        old = numbers.get(key)
        if old is None or (mode == "max" and value > old["value"]) or (mode == "min" and value < old["value"]):
            numbers[key] = OrderedDict([("value", value), ("evidence", _snippet(sentence)), ("source", "regex")])

    for s in sentences(text):
        temps = [_num(m.group(1)) for m in _TEMP_RE.finditer(s)
                 if not re.search(r"(±|\+/-)\s*$", s[:m.start()])]
        if temps and not _PRINT_CTX.search(s):
            if _SHORT_CTX.search(s):
                put("short_term_temp_c", max(temps), s, "max")
            elif _HEAT_CTX.search(s):
                put("heat_resistance_min_c", max(temps), s, "max")
            else:
                put("temp_max_c", max(temps), s, "max")
                if len(temps) > 1:
                    put("temp_min_c", min(temps), s, "min")
        m = re.search(r"(\d+[.,]\d+)\s*г\s*/?\s*см", s)
        if m and re.search(r"не\s+более|не\s+тяжелее|не\s+выше|≤|максимум|до\s+\d", s):
            put("density_max_g_cm3", _num(m.group(1)), s, "min")
        if re.search(r"модул\w*\s+упругост", s):
            m = re.search(r"(\d[\d ]{2,6})\s*мпа", s)
            if m:
                put("modulus_min_mpa", _num(m.group(1)), s, "max")
        if re.search(r"предел\w*\s+прочност|прочност\w*\s+при\s+растяж", s):
            m = re.search(r"(\d[\d ]{0,5})\s*мпа", s)
            if m:
                put("tensile_min_mpa", _num(m.group(1)), s, "max")
        if re.search(r"удлинени", s):
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", s)
            if m:
                put("elongation_min_pct", _num(m.group(1)), s, "max")
        if re.search(r"водопоглощ", s):
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", s)
            if m:
                put("water_absorption_max_pct", _num(m.group(1)), s, "min")
        for m in re.finditer(r"(?:±|\+/-)\s*(\d+[.,]\d+)\s*мм", s):
            put("tolerance_mm", _num(m.group(1)), s, "min")
    return numbers


def _derive_flags_from_numbers(result):
    """Numbers imply flags even when no keyword matched (e.g. '+85 °C' -> heat)."""
    n = result["numbers"]
    flags = result["flags"]
    hot = max([n[k]["value"] for k in ("temp_max_c", "heat_resistance_min_c") if k in n] or [0])
    if hot >= 50 and "heat" not in flags:
        flags["heat"] = OrderedDict([("strength", 0.8 if hot >= 70 else 0.6),
                                     ("evidence", "температура до %g °C" % hot), ("source", "regex")])
    if "density_max_g_cm3" in n and "lightweight" not in flags:
        flags["lightweight"] = OrderedDict([("strength", 0.8), ("evidence", n["density_max_g_cm3"]["evidence"]),
                                            ("source", "regex")])
    if "modulus_min_mpa" in n and "rigid" not in flags:
        flags["rigid"] = OrderedDict([("strength", 0.8), ("evidence", n["modulus_min_mpa"]["evidence"]),
                                      ("source", "regex")])
    if "tolerance_mm" in n and "fit" not in flags:
        flags["fit"] = OrderedDict([("strength", 0.6), ("evidence", n["tolerance_mm"]["evidence"]),
                                    ("source", "regex")])


# ----------------------------------------------------------------------------- LLM extraction

TOOL_NAME = "submit_task_analysis"


def llm_schema():
    num = {"type": ["number", "null"]}
    return {
        "type": "object",
        "properties": OrderedDict([
            ("part_name", {"type": "string", "description": "Короткое название детали"}),
            ("summary", {"type": "string", "description": "1–2 предложения: назначение и главные требования"}),
            ("constraints", {
                "type": "object",
                "properties": OrderedDict([
                    ("material", {"type": "string", "enum": material_ids() + ["none"],
                                  "description": "Материал печати, только если пользователь прямо его требует"}),
                    ("supports", {"type": "string", "enum": SUPPORTS}),
                    ("mode", {"type": "string", "enum": MODES}),
                    ("quality", {"type": "string", "enum": QUALITY}),
                    ("layer_height_mm", num),
                ]),
                "required": ["material", "supports", "mode", "quality"],
            }),
            ("flags", {
                "type": "array",
                "items": {"type": "object", "properties": OrderedDict([
                    ("id", {"type": "string", "enum": flag_ids()}),
                    ("strength", {"type": "number", "minimum": 0, "maximum": 1}),
                    ("quote", {"type": "string", "description": "Короткая цитата из текста"}),
                ]), "required": ["id", "strength"]},
            }),
            ("applications", {"type": "array", "items": {"type": "string", "enum": application_ids()}}),
            ("numbers", {"type": "object", "properties": OrderedDict((k, num) for k in NUMBER_KEYS)}),
        ]),
        "required": ["part_name", "summary", "constraints", "flags", "applications", "numbers"],
    }


def _system_prompt():
    kg = _kg()
    flags = "\n".join("- %s: %s" % (k, v["label"]) for k, v in kg["flags"].items())
    apps = "\n".join("- %s: %s" % (k, v["label"]) for k, v in kg["applications"].items())
    nums = "\n".join("- %s: %s" % (k, v) for k, v in NUMBER_KEYS.items())
    mats = ", ".join("%s (%s)" % (k, v["label"]) for k, v in kg["materials"].items())
    return (
        "Ты разбираешь задание на FDM-печать детали (принтер Stereotech Hybrid 530, режимы 3D и 5D) "
        "и возвращаешь структурированные данные вызовом функции %s. Ничего не придумывай: заполняй только то, "
        "что следует из текста.\n\n"
        "Явные ограничения (constraints) — только то, что пользователь требует прямо:\n"
        "- material: материал печати, если он прямо назван как материал для печати. Материал исходной, "
        "штатной или заменяемой детали (литой полиуретан, латунь, POM и т. п.) — НЕ ограничение, ставь none.\n"
        "- supports: forbidden только при прямом запрете поддержек («без поддержек»). Требование «без следов "
        "поддержек» — это флаг appearance, а не запрет.\n"
        "- mode: только если прямо назван режим печати (3D/5D, 5D Spiral, 5D Spiral Full), иначе unspecified.\n"
        "- quality: draft — просят быстро/черновик; fine — требуют гладкую поверхность, без видимых слоёв; "
        "иначе unspecified.\n\n"
        "Флаги задачи (flags) — неявные приоритеты; strength от 0 до 1 (1 — ключевое требование, 0.5 — "
        "второстепенное). Ставь флаг только при явной опоре на текст и приводи цитату. Возможные флаги:\n%s\n\n"
        "Назначения детали (applications):\n%s\n\n"
        "Числовые требования (numbers), null если в тексте нет:\n%s\n"
        "Рабочая температура — длительная температура эксплуатации; кратковременный нагрев — отдельно.\n\n"
        "Доступные материалы: %s."
    ) % (TOOL_NAME, flags, apps, nums, mats)


def parse_with_llm(text, llm):
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": text},
    ]
    options = stage_options(getattr(llm, "model", None), 1)
    args, meta = llm.call_tool(messages, TOOL_NAME, "Результат разбора задания на печать", llm_schema(),
                               extra_body=options)
    return _from_llm_args(args), meta


def _from_llm_args(args):
    result = empty_parse()
    result["source"] = "llm"
    known_flags = set(flag_ids())
    known_apps = set(application_ids())
    known_mats = set(material_ids())
    c = args.get("constraints") or {}
    mat = c.get("material")
    result["constraints"]["material"] = mat if mat in known_mats else None
    result["constraints"]["supports"] = c.get("supports") if c.get("supports") in SUPPORTS else "unspecified"
    result["constraints"]["mode"] = c.get("mode") if c.get("mode") in MODES else "unspecified"
    result["constraints"]["quality"] = c.get("quality") if c.get("quality") in QUALITY else "unspecified"
    lh = c.get("layer_height_mm")
    result["constraints"]["layer_height_mm"] = float(lh) if isinstance(lh, (int, float)) and lh > 0 else None
    for item in args.get("flags") or []:
        if not isinstance(item, dict) or item.get("id") not in known_flags:
            continue
        try:
            strength = max(0.0, min(1.0, float(item.get("strength", 0.7))))
        except (TypeError, ValueError):
            strength = 0.7
        if strength <= 0:
            continue
        prev = result["flags"].get(item["id"])
        if prev is None or strength > prev["strength"]:
            result["flags"][item["id"]] = OrderedDict([
                ("strength", round(strength, 2)), ("evidence", (item.get("quote") or "")[:160]), ("source", "llm")])
    for app in args.get("applications") or []:
        if app in known_apps:
            result["applications"][app] = OrderedDict([("strength", 1.0), ("evidence", ""), ("source", "llm")])
    for key, value in (args.get("numbers") or {}).items():
        if key in NUMBER_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
            result["numbers"][key] = OrderedDict([("value", float(value)), ("evidence", ""), ("source", "llm")])
    result["part_name"] = args.get("part_name")
    result["summary"] = args.get("summary")
    return result


# ----------------------------------------------------------------------------- merge

LEXICON_ONLY_STRENGTH = 0.5


def merge(llm_result, lex_result):
    if llm_result is None:
        return lex_result
    merged = empty_parse()
    merged["source"] = "llm+lexicon"
    merged["part_name"] = llm_result["part_name"]
    merged["summary"] = llm_result["summary"]
    notes = merged["discrepancies"]

    lc, xc, mc = llm_result["constraints"], lex_result["constraints"], merged["constraints"]
    for key, empty in (("material", None), ("supports", "unspecified"), ("mode", "unspecified"),
                       ("quality", "unspecified"), ("layer_height_mm", None)):
        if lc[key] != empty:
            mc[key] = lc[key]
            if xc[key] not in (empty, lc[key]):
                notes.append("%s: LLM=%s, словарь=%s; принято значение LLM" % (key, lc[key], xc[key]))
        elif xc[key] != empty:
            mc[key] = xc[key]
            notes.append("%s: LLM не указала, словарь нашёл %s" % (key, xc[key]))

    for flag, item in llm_result["flags"].items():
        merged["flags"][flag] = OrderedDict(item)
        if flag in lex_result["flags"]:
            merged["flags"][flag]["source"] = "llm+lexicon"
    for flag, item in lex_result["flags"].items():
        if flag not in merged["flags"]:
            entry = OrderedDict(item)
            entry["strength"] = LEXICON_ONLY_STRENGTH
            merged["flags"][flag] = entry
            notes.append("флаг %s найден только словарём" % flag)

    for app, item in llm_result["applications"].items():
        merged["applications"][app] = OrderedDict(item)
    for app, item in lex_result["applications"].items():
        if app not in merged["applications"]:
            entry = OrderedDict(item)
            entry["strength"] = LEXICON_ONLY_STRENGTH
            merged["applications"][app] = entry

    for key in NUMBER_KEYS:
        lv, xv = llm_result["numbers"].get(key), lex_result["numbers"].get(key)
        if lv is not None:
            entry = OrderedDict(lv)
            if xv is not None:
                entry["evidence"] = xv["evidence"]
                if abs(lv["value"] - xv["value"]) > 1e-6:
                    notes.append("%s: LLM=%g, текст=%g; принято значение LLM" % (key, lv["value"], xv["value"]))
            merged["numbers"][key] = entry
        elif xv is not None:
            merged["numbers"][key] = OrderedDict(xv)
    _derive_flags_from_numbers(merged)
    resolve_contradictions(merged)
    return merged


CONTRADICTORY_FLAGS = (("flexible", "rigid"),)


def resolve_contradictions(parse):
    """Drop the weaker of two mutually exclusive flags (lexicon-confirmed wording wins)."""
    flags = parse["flags"]
    for a, b in CONTRADICTORY_FLAGS:
        if a in flags and b in flags:
            def rank(name):
                item = flags[name]
                return (item.get("source") != "llm", float(item["strength"]))
            keep, drop = (a, b) if rank(a) >= rank(b) else (b, a)
            del flags[drop]
            parse["discrepancies"].append("противоречивые флаги %s и %s: оставлен %s" % (a, b, keep))


def parse_task(text, llm=None):
    """Stage 1 entry point. Returns (parse, meta)."""
    lex_result = parse_with_lexicon(text)
    meta = OrderedDict([("llm", None), ("llm_error", None)])
    if llm is None:
        return lex_result, meta
    try:
        llm_result, llm_meta = parse_with_llm(text, llm)
        meta["llm"] = llm_meta
    except Exception as e:  # network, schema, ... -> the lexicon result still works
        meta["llm_error"] = str(e)
        lex_result["discrepancies"].append("LLM недоступна: %s" % e)
        return lex_result, meta
    return merge(llm_result, lex_result), meta


def flag_strength(parse, flag):
    item = parse["flags"].get(flag)
    return float(item["strength"]) if item else 0.0
