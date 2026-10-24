import copy
import json
import os
import subprocess
import sys
from collections import OrderedDict

import numpy as np
import pytest

import meshes
from printplan import actions, fallback, io_utils, mesh as M, orientation as O, pipeline, rules, symmetry as S
from printplan import text_parse as T, validation
from printplan.llm_client import LLMError, ScriptedLLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evidence_for(tris, text, machine, kg, catalog):
    parse = T.parse_with_lexicon(text)
    mesh = M.analyze(tris, machine)
    rot = S.analyze(mesh)
    rank3d = O.rank(mesh, machine, supports_forbidden=parse["constraints"]["supports"] == "forbidden")
    E_KG = kg.rank(parse)
    E_G, rctx = rules.build(parse, mesh, rot, rank3d, machine, kg, catalog)
    return parse, mesh, rot, rank3d, E_G, rctx, E_KG


# ----------------------------------------------------------------------------- stage 4: mode choice

def test_bushing_goes_5d_spiral(machine, kg, catalog):
    _, _, _, _, E_G, _, _ = evidence_for(meshes.tube(20, 8, 15), "Втулка ролика", machine, kg, catalog)
    mc = E_G["mode_candidate"]
    assert mc["candidate"] == "cylindrical"
    assert mc["spiral_base_diameter_mm"] == pytest.approx(16.0, abs=0.1)
    theta = E_G["modes"]["cylindrical"]["theta0"]
    assert theta["non_printing_base_diameter"]["value"] == mc["spiral_base_diameter_mm"]


def test_solid_roller_goes_5d_full(machine, kg, catalog):
    _, _, _, _, E_G, _, _ = evidence_for(meshes.cylinder(15, 40), "Ролик", machine, kg, catalog)
    assert E_G["mode_candidate"]["candidate"] == "cylindrical_full"


def test_flexible_requirement_keeps_revolution_in_3d(machine, kg, catalog):
    text = "Эластичная манжета, должна быть гибкой и мягкой"
    _, _, _, _, E_G, _, _ = evidence_for(meshes.tube(20, 8, 15), text, machine, kg, catalog)
    mc = E_G["mode_candidate"]
    assert mc["candidate"] == "classic"
    assert any(e["term"] == "flexible_material" for e in mc["evidence"])


def test_flat_gear_and_bracket_stay_3d(machine, kg, catalog):
    for tris in (meshes.gear(teeth=24), meshes.l_bracket(), meshes.box((30, 20, 10))):
        _, _, _, _, E_G, _, _ = evidence_for(tris, "Деталь под нагрузку", machine, kg, catalog)
        assert E_G["mode_candidate"]["candidate"] == "classic"


def test_explicit_mode_wins(machine, kg, catalog):
    _, _, _, _, E_G, _, _ = evidence_for(meshes.tube(20, 8, 15), "Напечатай в 3D", machine, kg, catalog)
    mc = E_G["mode_candidate"]
    assert mc["candidate"] == "classic" and mc["source"] == "user"


def test_5d_orientation_and_theta0_use_cylindrical_keys(machine, kg, catalog):
    _, _, _, _, E_G, rctx, _ = evidence_for(meshes.cylinder(15, 40), "Вал под нагрузку на кручение, скручивание",
                                            machine, kg, catalog)
    item = E_G["modes"]["cylindrical_full"]
    assert item["orientation"]["id"] in ("axis+", "axis-")
    keys = set(item["theta0"].keys())
    assert "support_enable_cylindrical" in keys
    assert "speed_wall_0" not in keys


def test_supports_forbidden_theta_and_risk(machine, kg, catalog):
    _, _, _, _, E_G, _, _ = evidence_for(meshes.sphere(), "Шар без поддержек", machine, kg, catalog)
    item = E_G["modes"]["classic"]
    assert item["theta0"]["support_enable"]["value"] is False
    assert any(r["id"] == "supports_forbidden_unavoidable" for r in item["risks"])


def test_body_of_revolution_prefers_axis_vertical_in_3d(machine, kg, catalog):
    tris = meshes.rotate(meshes.cone(15, 30), [[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # axis along Y
    _, _, _, _, E_G, _, _ = evidence_for(tris, "Эластичный колпачок, гибкий", machine, kg, catalog)
    up = E_G["modes"]["classic"]["orientation"]["up_in_model"]
    assert abs(up[1]) > 0.99


def test_flag_corrections_in_theta0(machine, kg, catalog):
    _, _, _, _, E_G, _, _ = evidence_for(meshes.box((40, 30, 10)), "Корпус должен плотно сидеть в пазу, точность ±0.1 мм",
                                         machine, kg, catalog)
    theta = E_G["modes"]["classic"]["theta0"]
    assert theta["speed_wall_0"]["value"] == 20
    assert theta["xy_offset_layer_0"]["value"] == -0.1
    assert E_G["modes"]["classic"]["quality_type"] == "normal"


def test_mesh_risks(machine, kg, catalog):
    _, _, _, _, E_G, _, _ = evidence_for(meshes.box()[:-2], "Деталь", machine, kg, catalog)
    item = E_G["modes"]["classic"]
    assert any(r["id"] == "mesh_holes" for r in item["risks"])
    assert item["theta0"]["meshfix_extensive_stitching"]["value"] is True


# ----------------------------------------------------------------------------- stage 7: validation

def _setup(machine, kg, catalog, ctx, tris=None, text="Кронштейн под нагрузку"):
    parse, mesh, rot, rank3d, E_G, rctx, E_KG = evidence_for(tris if tris is not None else meshes.l_bracket(),
                                                              text, machine, kg, catalog)
    allowed = pipeline.allowed_materials(E_KG, parse, ctx, kg)
    from printplan import reasoning
    params = reasoning.tool_parameters(E_G, ctx, allowed)
    plan = fallback.build_plan(parse, E_G, E_KG, ctx, allowed)
    return parse, E_G, E_KG, allowed, params, plan


def test_fallback_plan_is_valid(machine, kg, catalog, ctx):
    parse, E_G, E_KG, allowed, params, plan = _setup(machine, kg, catalog, ctx)
    errors, _ = validation.validate_plan(plan, parse, E_G, E_KG, ctx, allowed, params)
    assert errors == []
    assert plan["explanation"]


@pytest.mark.parametrize("mutate,fragment", [
    (lambda p: p.pop("material"), "material"),
    (lambda p: p.update(mode="cylindrical"), "cylindrical"),
    (lambda p: p.update(quality_type="ultra"), "ultra"),
    (lambda p: p.update(material="unobtanium"), "unobtanium"),
    (lambda p: p["orientation"].update(candidate_id="axis+"), "axis+"),
    (lambda p: p["settings"].append({"key": "no_such_key", "value": 1, "reason": "x"}), "no_such_key"),
    (lambda p: p["settings"].append({"key": "infill_sparse_density", "value": -5, "reason": "x"}), "infill_sparse_density"),
    (lambda p: p["settings"].append({"key": "machine_width", "value": 500, "reason": "x"}), "machine_width"),
    (lambda p: p["settings"].append({"key": "speed_wall_0_cylindrical", "value": 20, "reason": "x"}), "5D"),
    (lambda p: p.update(explanation=" "), "объяснение"),
])
def test_validation_errors(machine, kg, catalog, ctx, mutate, fragment):
    parse, E_G, E_KG, allowed, params, plan = _setup(machine, kg, catalog, ctx)
    bad = copy.deepcopy(plan)
    mutate(bad)
    errors, _ = validation.validate_plan(validation.normalize_plan(bad, ctx), parse, E_G, E_KG, ctx, allowed, params)
    assert any(fragment.lower() in e.lower() for e in errors), errors


def test_validation_constraints(machine, kg, catalog, ctx):
    parse, E_G, E_KG, allowed, params, plan = _setup(
        machine, kg, catalog, ctx, meshes.t_overhang(), "Кронштейн, PETG, без поддержек")
    bad = copy.deepcopy(plan)
    bad["settings"].append({"key": "support_enable", "value": True, "reason": "x"})
    bad["material"] = next(m for m in allowed if m != "enduse_petg")
    errors, _ = validation.validate_plan(validation.normalize_plan(bad, ctx), parse, E_G, E_KG, ctx, allowed, params)
    assert any("запретил поддержки" in e for e in errors)
    assert any("задал материал" in e for e in errors)


def test_validation_requires_supports_for_overhangs(machine, kg, catalog, ctx):
    parse, E_G, E_KG, allowed, params, plan = _setup(machine, kg, catalog, ctx, meshes.t_overhang(), "Деталь")
    bad = copy.deepcopy(plan)
    bad["orientation"]["candidate_id"] = "+y"   # standing T: overhanging arms
    bad["deviations"] = [{"what": "orientation", "reason": "так красивее"}]
    bad["settings"] = [s for s in bad["settings"] if s["key"] != "support_enable"]
    bad["settings"].append({"key": "support_enable", "value": False, "reason": "x"})
    errors, _ = validation.validate_plan(validation.normalize_plan(bad, ctx), parse, E_G, E_KG, ctx, allowed, params)
    assert any("поддержки выключены" in e for e in errors)


def test_deviation_needs_justification(machine, kg, catalog, ctx):
    parse, E_G, E_KG, allowed, params, plan = _setup(machine, kg, catalog, ctx)
    other = next(m for m in allowed if m != plan["material"])
    bad = copy.deepcopy(plan)
    bad["material"] = other
    errors, _ = validation.validate_plan(bad, parse, E_G, E_KG, ctx, allowed, params)
    assert any("deviations" in e for e in errors)
    bad["deviations"] = [{"what": "material", "reason": "по данным графа ближе к задаче"}]
    errors, _ = validation.validate_plan(bad, parse, E_G, E_KG, ctx, allowed, params)
    assert not any("deviations" in e for e in errors)


def test_value_coercion(ctx):
    plan = validation.normalize_plan({"settings": [
        {"key": "support_enable", "value": "true", "reason": ""},
        {"key": "wall_line_count", "value": 4.0, "reason": ""},
        {"key": "printing_mode", "value": "classic", "reason": ""},
    ]}, ctx)
    values = dict((s["key"], s["value"]) for s in plan["settings"])
    assert values == {"support_enable": True, "wall_line_count": 4}


# ----------------------------------------------------------------------------- pipeline

def _good_answer(plan):
    return lambda messages: copy.deepcopy(plan)


def test_pipeline_without_llm_returns_fallback(machine):
    events = []
    result = pipeline.run("Втулка", triangles=meshes.tube(20, 8, 15), on_event=events.append)
    assert result["source"] == "fallback"
    assert result["plan"]["mode"] == "cylindrical"
    assert result["validation"]["errors"] == []
    assert [a["tool"] for a in result["actions"]][-2:] == ["place_model", "slice"]
    assert any(e["stage"] == "4" for e in events)


def test_pipeline_retry_then_success(machine, kg, catalog, ctx):
    good = pipeline.run("Кронштейн под нагрузку", triangles=meshes.l_bracket())["plan"]
    bad = copy.deepcopy(good)
    bad["material"] = "unobtanium"
    stage1 = {"part_name": "кронштейн", "summary": "", "constraints": {"material": "none", "supports": "unspecified",
              "mode": "unspecified", "quality": "unspecified"}, "flags": [{"id": "load", "strength": 0.9}],
              "applications": ["bracket_load"], "numbers": {}}
    llm = ScriptedLLM([stage1, bad, _good_answer(good)])
    result = pipeline.run("Кронштейн под нагрузку", triangles=meshes.l_bracket(), llm=llm)
    assert result["source"] == "llm"
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["errors"]
    retry_prompt = llm.calls[2]["messages"][-1]["content"]
    assert "unobtanium" in retry_prompt


def test_pipeline_falls_back_after_repeated_failures():
    stage1 = LLMError("stage 1 down")
    bad = {"verdict": "printable"}
    llm = ScriptedLLM([stage1, bad, bad, bad])
    result = pipeline.run("Кронштейн", triangles=meshes.l_bracket(), llm=llm)
    assert result["source"] == "fallback"
    assert len(result["attempts"]) == 3
    assert result["validation"]["errors"] == []


def test_pipeline_llm_exception_falls_back():
    llm = ScriptedLLM([LLMError("down"), LLMError("down")])
    result = pipeline.run("Кронштейн", triangles=meshes.l_bracket(), llm=llm)
    assert result["source"] == "fallback"
    assert "error" in result["attempts"][0]


# ----------------------------------------------------------------------------- actions

def test_stl_to_scene_rotation():
    # A model rotated so that its +Y points up in the STL frame...
    R = O.Orientation("+y", [0, 1, 0], "axis_up").rotation
    scene = actions.stl_to_scene_rotation(R)
    # ...has its scene-frame direction for model +Y (scene -Z after the reader swap) pointing to scene +Y.
    model_y_in_scene = actions.stl_to_scene_point([0, 1, 0])
    assert np.allclose(scene.dot(model_y_in_scene), [0, 1, 0])
    assert np.allclose(actions.stl_to_scene_point([1, 2, 3]), [1, 3, -2])


def test_settings_split_by_stack(ctx):
    g, e = actions.split_settings(OrderedDict([("support_enable", True), ("wall_line_count", 3)]), ctx)
    assert "support_enable" in g and "wall_line_count" in e


# ----------------------------------------------------------------------------- CLI

def test_cli_plan_smoke(tmpdir):
    stl = str(tmpdir.join("part.stl"))
    io_utils.write_stl(stl, meshes.tube(20, 8, 15))
    out = str(tmpdir.join("plan.json"))
    env = dict(os.environ)
    env.pop("STE_LLM_API_KEY", None)
    proc = subprocess.run([sys.executable, "-m", "printplan", "plan", stl, "Втулка ролика", "--json", out],
                          cwd=os.path.join(ROOT, "plugin", "AIAssistant"), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0, proc.stderr.decode("utf-8")
    assert "Режим: cylindrical" in proc.stdout.decode("utf-8")
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    assert data["plan"]["mode"] == "cylindrical"


def test_hard_contraindication_removes_5d_from_choices(machine, kg, catalog):
    text = "Эластичная манжета, должна быть гибкой и мягкой"
    _, _, _, _, E_G, _, _ = evidence_for(meshes.tube(20, 8, 15), text, machine, kg, catalog)
    assert list(E_G["modes"].keys()) == ["classic"]


def test_llm_only_flags_count_half_in_mode_choice(machine, kg, catalog):
    parse = T.parse_with_lexicon("Ролик")
    parse["flags"]["torsion"] = OrderedDict([("strength", 0.9), ("evidence", ""), ("source", "llm")])
    mesh = M.analyze(meshes.box((30, 20, 10)), machine)
    rot = S.analyze(mesh)
    mc = rules.choose_mode(parse, mesh, rot, O.rank(mesh, machine), machine)
    assert not any(e["term"] == "torsion" for e in mc["evidence"])
    cyl = M.analyze(meshes.cylinder(10, 60), machine)
    parse["flags"]["torsion"]["source"] = "llm+lexicon"
    mc = rules.choose_mode(parse, cyl, S.analyze(cyl), O.rank(cyl, machine), machine)
    assert any(e["term"] == "torsion" and e["weight"] == pytest.approx(1.8) for e in mc["evidence"])
