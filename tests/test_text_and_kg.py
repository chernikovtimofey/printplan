from collections import OrderedDict

import pytest

from printplan import text_parse as T
from printplan.llm_client import ScriptedLLM, LLMError


def flags(text):
    return T.parse_with_lexicon(text)["flags"]


# ----------------------------------------------------------------------------- stage 1: lexicon

def test_case_example():
    p = T.parse_with_lexicon("Кронштейн под нагрузку, PETG, без поддержек")
    c = p["constraints"]
    assert c["material"] == "enduse_petg"
    assert c["supports"] == "forbidden"
    assert "load" in p["flags"]
    assert "bracket_load" in p["applications"]


def test_support_marks_are_appearance_not_ban():
    p = T.parse_with_lexicon("Детали не должны содержать видимых слоёв и следов от поддержек.")
    assert p["constraints"]["supports"] == "unspecified"
    assert "appearance" in p["flags"]
    assert p["constraints"]["quality"] == "fine"


def test_negated_rigid_means_flexible():
    f = flags("Применение жестких материалов не допускается: пыльник должен сохранять эластичность.")
    assert "flexible" in f
    assert "rigid" not in f


def test_not_soft_means_rigid():
    f = flags("Деталь не должна быть мягкой, зуб не должен подгибаться.")
    assert "rigid" in f
    assert "flexible" not in f


def test_replaced_part_material_is_not_a_constraint():
    p = T.parse_with_lexicon("Текущие серийные колеса из литого полиуретана выходят из строя.")
    assert p["constraints"]["material"] is None


def test_explicit_mode_and_layer_height():
    p = T.parse_with_lexicon("Напечатай втулку в 5D, высота слоя 0.2 мм, побыстрее")
    assert p["constraints"]["mode"] == "5d_any"
    assert p["constraints"]["layer_height_mm"] == 0.2
    assert p["constraints"]["quality"] == "draft"


def test_numbers_from_spec_sentences():
    text = ("Температурный режим среды: +75°C...+85°C (постоянно).\n"
            "Температура размягчения по Вика (метод B, 50 Н): не ниже +130°C.\n"
            "Плотность: Не более 1.2 г/см³.\n"
            "Модуль упругости при растяжении: не менее 4500 МПа.\n"
            "Предел прочности при растяжении: не менее 50 МПа.\n"
            "Относительное удлинение при разрыве: не менее 2.5%.\n"
            "Водопоглощение за 24 часа погружения должно составлять не более 0.05%.\n"
            "Температурный режим: Строго +20°C ± 2°C.\n"
            "Требуется точность отверстия (допуск ±0.1 мм).\n"
            "Температура капания клея до 160°C, кратковременный контакт.")
    n = T.extract_numbers(text)
    assert n["temp_max_c"]["value"] == 85
    assert n["heat_resistance_min_c"]["value"] == 130
    assert n["density_max_g_cm3"]["value"] == 1.2
    assert n["modulus_min_mpa"]["value"] == 4500
    assert n["tensile_min_mpa"]["value"] == 50
    assert n["elongation_min_pct"]["value"] == 2.5
    assert n["water_absorption_max_pct"]["value"] == 0.05
    assert n["tolerance_mm"]["value"] == 0.1
    assert n["short_term_temp_c"]["value"] == 160
    assert n["temp_min_c"]["value"] == 75  # "± 2°C" is a tolerance, not a temperature


def test_numbers_imply_flags():
    p = T.parse_with_lexicon("Рабочая температура до +90°C. Плотность не более 1.1 г/см³.")
    assert "heat" in p["flags"] and "lightweight" in p["flags"]


def test_merge_prefers_llm_and_keeps_lexicon_extras():
    args = {
        "part_name": "кронштейн", "summary": "Кронштейн под нагрузку",
        "constraints": {"material": "enduse_petg", "supports": "unspecified", "mode": "unspecified",
                        "quality": "unspecified", "layer_height_mm": None},
        "flags": [{"id": "load", "strength": 0.9, "quote": "под нагрузку"}, {"id": "bogus", "strength": 1}],
        "applications": ["bracket_load", "nope"],
        "numbers": {"temp_max_c": 60},
    }
    llm = ScriptedLLM([args])
    p, meta = T.parse_task("Кронштейн под нагрузку, PETG, без поддержек, на улице", llm)
    assert p["source"] == "llm+lexicon"
    assert p["constraints"]["material"] == "enduse_petg"
    assert p["constraints"]["supports"] == "forbidden"          # LLM missed it, lexicon found it
    assert p["flags"]["load"]["strength"] == 0.9
    assert p["flags"]["outdoor_uv"]["strength"] == T.LEXICON_ONLY_STRENGTH
    assert "bogus" not in p["flags"]
    assert p["applications"]["bracket_load"]["source"] == "llm"
    assert p["applications"]["outdoor"]["strength"] == T.LEXICON_ONLY_STRENGTH
    assert "nope" not in p["applications"]
    assert p["numbers"]["temp_max_c"]["value"] == 60
    assert any("supports" in d for d in p["discrepancies"])


def test_llm_failure_falls_back_to_lexicon():
    p, meta = T.parse_task("Кронштейн, PETG", ScriptedLLM([LLMError("down")]))
    assert p["source"] == "lexicon"
    assert meta["llm_error"]
    assert p["constraints"]["material"] == "enduse_petg"


def test_llm_schema_enums_match_kg():
    schema = T.llm_schema()
    flag_enum = schema["properties"]["flags"]["items"]["properties"]["id"]["enum"]
    assert set(flag_enum) == set(T.flag_ids())


# ----------------------------------------------------------------------------- stage 5: knowledge graph

def make_parse(flags=None, numbers=None, apps=None, material=None):
    p = T.empty_parse()
    for k, v in (flags or {}).items():
        p["flags"][k] = OrderedDict([("strength", v), ("evidence", ""), ("source", "test")])
    for k, v in (numbers or {}).items():
        p["numbers"][k] = OrderedDict([("value", v), ("evidence", ""), ("source", "test")])
    for a in apps or []:
        p["applications"][a] = OrderedDict([("strength", 1.0), ("evidence", ""), ("source", "test")])
    p["constraints"]["material"] = material
    return p


def top(kg, parse, mode="classic"):
    return kg.rank(parse, mode=mode)["ranking"][0]["id"]


def test_no_flags_prefers_easiest_material(kg):
    assert top(kg, make_parse()) == "proto_pla"


def test_heat_selects_heat_resistant(kg):
    p = make_parse({"heat": 1.0, "rigid": 0.8}, {"temp_max_c": 100})
    r = kg.rank(p)
    assert r["ranking"][0]["id"] in ("enduse_pc", "fiberpart_pc_g20", "fiberpart_pa12_g12", "fiberpart_pp_g30")
    assert all(x["id"] != "proto_pla" for x in r["ranking"])
    assert any(e["id"] == "proto_pla" for e in r["excluded"])


def test_flexible_selects_elastomer(kg):
    assert top(kg, make_parse({"flexible": 1.0}, apps=["seal_flexible"])).startswith("sealant_")


def test_oil_resistant_seal_is_tpv(kg):
    assert top(kg, make_parse({"flexible": 1.0, "chem_oil_fuel": 1.0}, {"temp_max_c": 75},
                              apps=["seal_flexible", "automotive"])) == "sealant_tpv80"


def test_density_and_dielectric_are_hard_filters(kg):
    r = kg.rank(make_parse({"dielectric": 1.0, "lightweight": 0.8}, {"density_max_g_cm3": 1.2}))
    excluded = dict((e["id"], e["reasons"]) for e in r["excluded"])
    assert "fiberpart_tpu_c5" in excluded        # carbon filler
    assert "enduse_petg" in excluded             # 1.30 g/cm3 in the slicer profile


def test_impeller_spec_ranks_pp_gf(kg):
    p = make_parse({"heat": 1.0, "chem_alkali": 1.0, "chem_glycol": 1.0, "moisture": 1.0, "rigid": 0.8,
                    "dielectric": 1.0, "wear": 0.8},
                   {"temp_max_c": 85, "heat_resistance_min_c": 130, "density_max_g_cm3": 1.2,
                    "modulus_min_mpa": 4500, "water_absorption_max_pct": 0.05},
                   apps=["impeller_fan"])
    assert top(kg, p, "cylindrical") == "fiberpart_pp_g30"


def test_user_material_conflict_is_reported(kg):
    p = make_parse({"heat": 1.0}, {"temp_max_c": 90}, material="proto_pla")
    r = kg.rank(p)
    assert r["user_material"]["id"] == "proto_pla"
    assert any(c["material"] == "proto_pla" for c in r["conflicts"])


def test_soft_material_penalised_in_5d(kg):
    p = make_parse({"flexible": 1.0}, apps=["seal_flexible"])
    classic = dict((x["id"], x["score"]) for x in kg.rank(p, "classic")["ranking"])
    fived = dict((x["id"], x["score"]) for x in kg.rank(p, "cylindrical")["ranking"])
    sealant = [k for k in classic if k.startswith("sealant_") and k in fived]
    assert sealant and all(fived[k] < classic[k] for k in sealant)


def test_chains_are_traceable(kg):
    r = kg.rank(make_parse({"chem_glycol": 1.0}))
    assert any("гликол" in chain for chain in r["ranking"][0]["chains"])


def test_kg_report(kg):
    text = kg.report()
    assert "fiberpart_pp_g30" in text and "density (slicer): 1.18" in text


def test_contradictory_flags_resolved():
    p = T.empty_parse()
    p["flags"]["flexible"] = OrderedDict([("strength", 1.0), ("evidence", ""), ("source", "llm+lexicon")])
    p["flags"]["rigid"] = OrderedDict([("strength", 0.8), ("evidence", ""), ("source", "llm")])
    T.resolve_contradictions(p)
    assert "rigid" not in p["flags"] and "flexible" in p["flags"]
    assert p["discrepancies"]


def test_llm_only_flag_does_not_exclude(kg):
    p = make_parse({"rigid": 0.9}, apps=["seal_flexible"])
    p["flags"]["rigid"]["source"] = "llm"
    r = kg.rank(p)
    assert not any(e["id"].startswith("sealant_") for e in r["excluded"])
    p["flags"]["rigid"]["source"] = "llm+lexicon"
    r = kg.rank(p)
    assert any(e["id"].startswith("sealant_") for e in r["excluded"])
