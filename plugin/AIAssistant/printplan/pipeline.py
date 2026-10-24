"""Orchestration of stages 1–7.

    result = run(task_text, triangles=..., llm=LLMClient.from_env())

Stages 1 (text) and 2 (mesh) run in parallel. Every stage reports a trace
event through `on_event(event_dict)` so the chat UI can show what the
assistant is doing. Without an LLM (or when it fails) the deterministic
fallback plan is returned, so the user always gets a working result.
"""
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from . import (actions, fallback, mesh as mesh_mod, orientation, reasoning, rules, symmetry, text_parse,
               validation)
from .io_utils import read_json_resource, read_stl, read_task
from .knowledge_graph import KnowledgeGraph
from .slicer_context import CatalogSlicerContext

MAX_RETRIES = 2


class Trace(object):
    def __init__(self, on_event=None):
        self.events = []
        self.on_event = on_event
        self.started = time.time()

    def emit(self, stage, status, summary, data=None):
        event = OrderedDict([("t", round(time.time() - self.started, 2)), ("stage", stage), ("status", status),
                             ("summary", summary)])
        if data is not None:
            event["data"] = data
        self.events.append(event)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass
        return event


def allowed_materials(E_KG, parse, ctx, kg):
    """Materials the plan may use: everything in the graph that passed the hard filters
    and exists in the slicer, plus the user's material."""
    excluded = set(e["id"] for e in E_KG["excluded"])
    slicer = set(ctx.material_ids())
    ids = [m for m in kg.materials if m not in excluded and m in slicer]
    user = parse["constraints"].get("material")
    if user and user not in ids and user in slicer:
        ids.insert(0, user)
    return ids


def run(task, triangles=None, stl_path=None, llm=None, ctx=None, images=None, on_event=None,
        max_retries=MAX_RETRIES):
    trace = Trace(on_event)
    ctx = ctx or CatalogSlicerContext()
    machine = ctx.machine()
    catalog = getattr(ctx, "catalog", None) or read_json_resource("slicer_catalog.json")
    kg = KnowledgeGraph(catalog=catalog)
    task_text = read_task(task) or ""
    if triangles is None:
        if stl_path is None:
            raise ValueError("Pass triangles or stl_path")
        triangles = read_stl(stl_path)

    # Stages 1 and 2 in parallel.
    trace.emit("1-2", "start", "Разбор текста и анализ сетки")
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_text = pool.submit(text_parse.parse_task, task_text, llm)
        f_mesh = pool.submit(mesh_mod.analyze, triangles, machine)
        mesh = f_mesh.result()
        trace.emit("2", "done", "Сетка: %d треугольников, %s мм, герметична: %s" % (
            mesh.Q_topo["n_triangles"], mesh.F_mesh["extents_mm"], "да" if mesh.Q_topo["watertight"] else "нет"),
            OrderedDict([("F_mesh", mesh.F_mesh), ("Q_topo", mesh.Q_topo)]))
        rot = symmetry.analyze(mesh)
        parse, parse_meta = f_text.result()
    constraints = ["%s=%s" % (k, v) for k, v in parse["constraints"].items() if v not in (None, "unspecified")]
    trace.emit("1", "done", "Явные ограничения: %s; флаги: %s" % (
        ", ".join(constraints) or "нет",
        ", ".join("%s=%.1f" % (k, v["strength"]) for k, v in parse["flags"].items()) or "нет"),
        OrderedDict([("parse", parse), ("meta", parse_meta)]))

    # Stage 3.
    supports_forbidden = parse["constraints"].get("supports") == "forbidden"
    rank3d = orientation.rank(mesh, machine, supports_forbidden=supports_forbidden)
    trace.emit("3", "done", "Лучшая ориентация 3D: %s (S_o = %.3f), мин. R_support = %.3f" % (
        rank3d["best"].id, rank3d["best"].score, rank3d["min_R_support"]))

    # Stages 4 and 5.
    E_KG = kg.rank(parse, mode="classic")
    hint = parse["constraints"].get("material") or (E_KG["ranking"][0]["id"] if E_KG["ranking"] else None)
    E_G, rctx = rules.build(parse, mesh, rot, rank3d, machine, kg, catalog, material_hint=hint)
    mc = E_G["mode_candidate"]
    trace.emit("4", "done", "Кандидат режима: %s (оценка %.1f, уверенность %s)" % (
        mc["candidate"], mc["score"], mc["confidence"]), mc)
    if mc["candidate"] != "classic":
        E_KG = kg.rank(parse, mode=mc["candidate"])
    allowed = allowed_materials(E_KG, parse, ctx, kg)
    trace.emit("5", "done", "Материалы: %s" % ", ".join(
        "%s %.2f" % (r["id"], r["score"]) for r in E_KG["ranking"][:3]),
        OrderedDict([("ranking", [r["id"] for r in E_KG["ranking"]]), ("conflicts", E_KG["conflicts"])]))

    # Stages 6 and 7.
    parameters = reasoning.tool_parameters(E_G, ctx, allowed)
    attempts = []
    plan, source = None, "fallback"
    fallback_reason = "языковая модель не подключена"
    if llm is not None:
        evidence = reasoning.build_evidence(task_text, parse, E_G, E_KG, ctx, allowed)
        messages = reasoning.build_messages(evidence, images)
        for attempt in range(max_retries + 1):
            trace.emit("6", "start", "Рассуждение LLM, попытка %d" % (attempt + 1))
            try:
                args, meta = reasoning.ask(llm, messages, parameters, on_progress=_throttled_progress(trace))
            except Exception as e:
                attempts.append(OrderedDict([("attempt", attempt + 1), ("error", str(e))]))
                trace.emit("6", "error", "LLM недоступна: %s" % e)
                fallback_reason = "ошибка обращения к языковой модели"
                break
            candidate = validation.normalize_plan(args, ctx)
            errors, warnings = validation.validate_plan(candidate, parse, E_G, E_KG, ctx, allowed, parameters)
            attempts.append(OrderedDict([("attempt", attempt + 1), ("meta", meta), ("errors", errors),
                                         ("warnings", warnings)]))
            trace.emit("7", "done" if not errors else "retry",
                       "Проверка: %s" % ("ошибок нет" if not errors else "%d ошибок" % len(errors)),
                       OrderedDict([("errors", errors), ("warnings", warnings)]))
            if not errors:
                plan, source = candidate, "llm"
                break
            messages = reasoning.retry_messages(messages, args, errors)
            fallback_reason = "ответы модели не прошли проверку"
    if plan is None:
        plan = fallback.build_plan(parse, E_G, E_KG, ctx, allowed, fallback_reason)
        trace.emit("7", "fallback", "Детерминированный план: %s" % fallback_reason)

    errors, warnings = validation.validate_plan(plan, parse, E_G, E_KG, ctx, allowed, parameters)
    settings = validation.final_settings(plan, E_G)
    chosen = resolve_orientation(plan, rank3d, rctx)
    axis = rot.selected if plan["mode"] != "classic" else None
    steps = actions.build_actions(plan, settings, chosen, ctx, axis)
    trace.emit("done", "done", "План: %s, %s, %s, ориентация %s" % (
        plan["mode"], plan["quality_type"], plan["material"], plan["orientation"]["candidate_id"]))

    result = OrderedDict()
    result["plan"] = plan
    result["source"] = source
    result["validation"] = OrderedDict([("errors", errors), ("warnings", warnings)])
    result["settings"] = settings
    result["orientation"] = OrderedDict([
        ("id", chosen.id), ("rotation_stl", actions._matrix(chosen.rotation)),
        ("rotation_scene", actions._matrix(actions.stl_to_scene_rotation(chosen.rotation))),
        ("translation_stl", actions._vector(chosen.translation)),
    ])
    result["actions"] = steps
    result["evidence"] = OrderedDict([("parse", parse), ("E_G", E_G), ("E_KG", E_KG)])
    result["attempts"] = attempts
    result["trace"] = trace.events
    return result


def _throttled_progress(trace, every_s=5.0):
    """Report LLM generation progress to the trace at most every `every_s` seconds."""
    state = {"next": every_s}

    def report(progress):
        if progress["elapsed_s"] >= state["next"]:
            state["next"] = progress["elapsed_s"] + every_s
            trace.emit("6", "progress", "Модель думает: %.0f с, рассуждение %d симв., ответ %d симв." % (
                progress["elapsed_s"], progress["reasoning_chars"], progress["answer_chars"]))
    return report


def resolve_orientation(plan, rank3d, rctx):
    """Orientation object for the plan's candidate_id in its mode."""
    mode = plan["mode"]
    oid = plan["orientation"]["candidate_id"]
    item = rctx["per_mode"].get(mode)
    if item is not None:
        if item["orientation"].id == oid:
            return item["orientation"]
        for o in item["table5d"]:
            if o.id == oid:
                return o
    if oid in rank3d["by_id"]:
        return rank3d["by_id"][oid]
    return item["orientation"] if item is not None else rank3d["best"]
