"""Tools of the assistant: one table for the chat agent and the MCP server.

Each tool has a name, a JSON schema of its arguments and a handler. Every call
goes through `ToolRegistry.call`, which writes "tool_call" / "tool_result"
events, so the chat shows what the assistant changes — whether the call came
from the agent, from an external MCP client or from `plan_print` applying a
plan step by step.
"""
import base64
import os
import re
import threading
import time
import traceback
from collections import OrderedDict

import numpy as np

from ..printplan import io_utils, pipeline
from ..printplan.slicer_context import check_value, coerce_value
from . import photo3d
from .slicer_base import SlicerError

MODES = ["classic", "cylindrical", "cylindrical_full"]
MODE_NAMES = {"classic": "3D", "cylindrical": "5D Spiral", "cylindrical_full": "5D Spiral Full"}


def jsonable(value):
    """Recursively convert numpy values to plain JSON types."""
    if isinstance(value, dict):
        return OrderedDict((str(k), jsonable(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def format_duration(seconds):
    if seconds is None:
        return None
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    return ("%d ч %02d мин" % (hours, minutes)) if hours else ("%d мин" % max(minutes, 1))


class SessionState(object):
    """What the tools remember between calls within one chat session."""

    def __init__(self):
        self.images = []          # data URLs of photos attached in the current turn
        self.photos = []          # last attached photos {"name", "data"}: kept for model_from_photos until new ones
        self.last_plan = None     # last pipeline result
        self.last_object_id = None
        self.lock = threading.RLock()


class ToolContext(object):
    def __init__(self, slicer, settings, events, registry=None):
        self.slicer = slicer
        self.settings = settings
        self.events = events
        self.registry = registry
        self.state = SessionState()

    def call(self, name, args=None, source="plan"):
        """Call another tool (used by plan_print so each step is visible in the chat)."""
        return self.registry.call(name, args or {}, self, source=source)


class Tool(object):
    def __init__(self, name, title, description, parameters, handler, public=True):
        self.name = name
        self.title = title
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.public = public


class ToolError(Exception):
    pass


class ToolRegistry(object):
    def __init__(self):
        self._tools = OrderedDict()
        self._counter = 0
        self._lock = threading.Lock()

    def add(self, tool):
        self._tools[tool.name] = tool

    def get(self, name):
        return self._tools.get(name)

    def public_tools(self):
        return [t for t in self._tools.values() if t.public]

    def openai_tools(self):
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}} for t in self.public_tools()]

    def mcp_tools(self):
        return [OrderedDict([("name", t.name), ("description", t.description), ("inputSchema", t.parameters)])
                for t in self.public_tools()]

    def _next_id(self):
        with self._lock:
            self._counter += 1
            return "t%d" % self._counter

    def call(self, name, args, ctx, source="agent", call_id=None):
        """Run a tool; returns {"ok": bool, "result" | "error": ...}. Never raises."""
        tool = self._tools.get(name)
        call_id = call_id or self._next_id()
        args = args if isinstance(args, dict) else {}
        title = tool.title if tool else name
        ctx.events.emit("tool_call", call_id=call_id, name=name, title=title, args=jsonable(args), source=source)
        started = time.time()
        if tool is None:
            outcome = OrderedDict([("ok", False), ("error", "Нет такого инструмента: %s" % name)])
        else:
            try:
                result = tool.handler(ctx, args)
                outcome = OrderedDict([("ok", True), ("result", jsonable(result))])
            except (SlicerError, ToolError) as e:
                outcome = OrderedDict([("ok", False), ("error", str(e))])
            except Exception as e:  # unexpected: report, keep the chat alive
                outcome = OrderedDict([("ok", False), ("error", "%s: %s" % (type(e).__name__, e)),
                                       ("trace", traceback.format_exc()[-2000:])])
        ctx.events.emit("tool_result", call_id=call_id, name=name, title=title, ok=outcome["ok"],
                        result=outcome.get("result"), error=outcome.get("error"), source=source,
                        elapsed_s=round(time.time() - started, 2))
        return outcome


# ----------------------------------------------------------------------------- argument helpers

def _schema(properties=None, required=None):
    return OrderedDict([("type", "object"), ("properties", OrderedDict(properties or [])),
                        ("required", list(required or []))])


def _object_id(ctx, args):
    oid = args.get("object_id")
    objects = ctx.slicer.list_objects()
    if oid is not None:
        try:
            oid = int(oid)
        except (TypeError, ValueError):
            raise ToolError("object_id должен быть числом из list_objects")
        if not any(o["id"] == oid for o in objects):
            raise ToolError("Нет объекта с id %s. На столе: %s" % (oid, ", ".join(
                "%s (%s)" % (o["id"], o["name"]) for o in objects) or "пусто"))
        return oid
    if not objects:
        raise ToolError("На столе нет моделей — сначала загрузите модель")
    if len(objects) == 1:
        return objects[0]["id"]
    last = ctx.state.last_object_id
    if last is not None and any(o["id"] == last for o in objects):
        return last
    raise ToolError("На столе несколько моделей, укажите object_id: %s" % ", ".join(
        "%s (%s)" % (o["id"], o["name"]) for o in objects))


def _output_path(ctx, args, ext):
    path = args.get("path")
    if path:
        folder = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        return path
    objects = ctx.slicer.list_objects()
    stem = os.path.splitext(objects[0]["name"])[0] if objects else "project"
    stem = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.U).strip() or "project"
    return os.path.join(ctx.settings.output_dir(), stem + ext)


# ----------------------------------------------------------------------------- handlers

def _get_state(ctx, args):
    return ctx.slicer.state()


def _list_quality(ctx, args):
    return ctx.slicer.list_quality_profiles()


def _list_materials(ctx, args):
    return ctx.slicer.list_materials()


def _get_settings(ctx, args):
    keys = args.get("keys") or []
    if not isinstance(keys, list) or not keys:
        raise ToolError("Передайте список ключей keys")
    return ctx.slicer.get_settings([str(k) for k in keys])


def _set_settings(ctx, args):
    values = args.get("settings")
    if not isinstance(values, dict) or not values:
        raise ToolError("Передайте словарь settings: {ключ: значение}")
    context = ctx.slicer.slicer_context()
    accepted, rejected, warnings = OrderedDict(), OrderedDict(), OrderedDict()
    for key, value in values.items():
        if key == "printing_mode":
            rejected[key] = "режим меняется инструментом set_printing_mode"
            continue
        spec = context.setting(key)
        value = coerce_value(value, spec)
        issues = check_value(key, value, spec)
        errors = [i["text"] for i in issues if i["level"] == "error"]
        if errors:
            rejected[key] = "; ".join(errors)
            continue
        if issues:
            warnings[key] = "; ".join(i["text"] for i in issues)
        accepted[key] = value
    applied = ctx.slicer.set_settings(accepted) if accepted else OrderedDict()
    slicer_errors = OrderedDict((k, v["validation"]) for k, v in applied.items()
                                if v.get("validation") in ("MinimumError", "MaximumError", "Exception"))
    out = OrderedDict([("applied", applied)])
    if rejected:
        out["rejected"] = rejected
    if warnings:
        out["warnings"] = warnings
    if slicer_errors:
        out["slicer_validation_errors"] = slicer_errors
    return out


def _reset_settings(ctx, args):
    ctx.slicer.reset_settings()
    return "Пользовательские изменения настроек сброшены к значениям профиля"


def _set_quality(ctx, args):
    return OrderedDict([("quality_type", ctx.slicer.set_quality(str(args.get("quality_type"))))])


def _set_material(ctx, args):
    material_id = str(args.get("material_id") or "")
    if not material_id:
        raise ToolError("Укажите material_id (см. list_materials)")
    return OrderedDict([("material", ctx.slicer.set_material(material_id, int(args.get("extruder", 0) or 0)))])


def _set_mode(ctx, args):
    mode = args.get("mode")
    if mode not in MODES:
        raise ToolError("mode должен быть одним из: %s" % ", ".join(MODES))
    return OrderedDict([("printing_mode", ctx.slicer.set_printing_mode(mode)), ("name", MODE_NAMES[mode])])


def _save_profile(ctx, args):
    name = str(args.get("name") or "").strip()
    if not name:
        raise ToolError("Укажите имя профиля")
    return OrderedDict([("profile", ctx.slicer.save_profile(name))])


def _load_model(ctx, args):
    path = str(args.get("path") or "")
    if not path:
        raise ToolError("Укажите путь к файлу модели")
    objects = ctx.slicer.load_model(path)
    if objects:
        ctx.state.last_object_id = objects[-1]["id"]
    return OrderedDict([("loaded", objects)])


PHOTO_MODEL_WARNING = ("Модель построена нейросетью по фото: форма приблизительная, точны пропорции и заданный "
                       "наибольший размер; отверстия, резьба и тонкие стенки могли сгладиться или исчезнуть. "
                       "Поэтому выбор режима и материала менее надёжен, чем по настоящей модели.")


def _photos_for_generation(ctx, args):
    paths = args.get("photo_paths") or []
    if paths:
        photos = []
        for path in paths:
            if not os.path.isfile(path):
                raise ToolError("Файл фото не найден: %s" % path)
            with open(path, "rb") as f:
                photos.append({"name": os.path.basename(path), "data": f.read()})
        return photos
    with ctx.state.lock:
        return list(ctx.state.photos)


def _model_from_photos(ctx, args):
    try:
        size = float(args.get("size_mm"))
    except (TypeError, ValueError):
        size = 0.0
    if not size > 0:
        raise ToolError("Нужен наибольший размер детали в мм (size_mm): масштаб по фото не определить — "
                        "спросите пользователя")
    photos = _photos_for_generation(ctx, args)
    if not photos:
        raise ToolError("Нет фото детали: прикрепите к сообщению одно или несколько фото")
    views = [str(v) for v in (args.get("views") or [])]
    bad = [v for v in views if v not in photo3d.VIEWS]
    if bad or len(set(views)) != len(views):
        raise ToolError("views — разные ракурсы из %s, по одному на фото" % ", ".join(photo3d.VIEWS))
    used = photos[:len(views) or photo3d.MAX_PHOTOS]
    views = views[:len(used)] or list(photo3d.VIEWS[:len(used)])
    try:
        triangles, info = photo3d.Photo3DClient(ctx.settings.photo3d_url()).generate([p["data"] for p in used], views)
    except photo3d.Photo3DError as e:
        raise ToolError(str(e))
    triangles = photo3d.scale_to_size(triangles, size)
    stem = re.sub(r"[^\w\-. ]+", "_", os.path.splitext(used[0]["name"])[0], flags=re.U).strip() or "photo"
    path = os.path.join(ctx.settings.output_dir(), stem + "_по_фото.stl")
    io_utils.write_stl(path, triangles)
    # Like an attached model, the generated one replaces the models on the bed.
    for name, step_args in (("clear_scene", {}), ("load_model", {"path": path})):
        if name == "clear_scene" and not ctx.slicer.list_objects():
            continue
        outcome = ctx.call(name, step_args)
        if not outcome["ok"]:
            raise ToolError(outcome["error"])
    with ctx.state.lock:  # the photos also go to the LLM step of plan_print
        ctx.state.images = [_data_url(p) for p in photos]
    return OrderedDict([
        ("loaded", outcome["result"]["loaded"]), ("path", path), ("size_mm", size),
        ("photos", [p["name"] for p in used]), ("views", views), ("triangles", int(len(triangles))),
        ("generation_s", info.get("seconds")), ("generator", info.get("model")), ("warning", PHOTO_MODEL_WARNING),
    ])


def _data_url(photo):
    mime = "image/png" if photo["name"].lower().endswith(".png") else "image/jpeg"
    return "data:%s;base64,%s" % (mime, base64.b64encode(photo["data"]).decode("ascii"))


def _list_objects(ctx, args):
    return ctx.slicer.list_objects()


def _object_info(ctx, args):
    oid = _object_id(ctx, args)
    return next(o for o in ctx.slicer.list_objects() if o["id"] == oid)


def _move_object(ctx, args):
    oid = _object_id(ctx, args)
    return ctx.slicer.move_object(oid, args.get("x"), args.get("y"), args.get("z"))


def _rotate_object(ctx, args):
    oid = _object_id(ctx, args)
    return ctx.slicer.rotate_object(oid, args.get("axis", "z"), float(args.get("angle_deg", 0)))


def _scale_object(ctx, args):
    oid = _object_id(ctx, args)
    factors = args.get("factors") or args.get("factor")
    if factors is None:
        raise ToolError("Укажите factor (одно число) или factors [x, y, z]")
    return ctx.slicer.scale_object(oid, factors)


def _arrange_all(ctx, args):
    return ctx.slicer.arrange_all()


def _delete_object(ctx, args):
    oid = _object_id(ctx, args)
    ctx.slicer.delete_object(oid)
    return "Объект удалён"


def _clear_scene(ctx, args):
    ctx.slicer.clear_scene()
    return "Стол очищен"


def _apply_orientation(ctx, args):
    oid = _object_id(ctx, args)
    return ctx.slicer.transform_object(oid, args.get("rotation"), args.get("translation"))


def _slice(ctx, args):
    last = {"value": -1.0}

    def progress(amount):
        if amount - last["value"] >= 0.05 or amount >= 1.0:
            last["value"] = amount
            ctx.events.emit("slice_progress", progress=round(float(amount), 3))

    ctx.events.emit("slice_progress", progress=0.0)
    result = OrderedDict(ctx.slicer.slice(progress=progress))
    result["print_time"] = format_duration(result.get("print_time_s"))
    return result


def _slice_status(ctx, args):
    return ctx.slicer.slice_status()


def _export_gcode(ctx, args):
    path = _output_path(ctx, args, ".gcode")
    return OrderedDict([("path", ctx.slicer.export_gcode(path))])


def _save_project(ctx, args):
    path = _output_path(ctx, args, ".3mf")
    return OrderedDict([("path", ctx.slicer.save_project(path))])


# ----------------------------------------------------------------------------- plan_print

def plan_summary(result, applied=None, slice_result=None, object_info=None):
    """What the agent and the chat page need from a pipeline result."""
    plan = result["plan"]
    evidence = result["evidence"]
    mc = evidence["E_G"]["mode_candidate"]
    kg = evidence["E_KG"]
    out = OrderedDict()
    if object_info:
        out["object"] = OrderedDict([("id", object_info["id"]), ("name", object_info["name"])])
    out["source"] = result["source"]
    out["verdict"] = plan["verdict"]
    out["mode"] = plan["mode"]
    out["mode_name"] = MODE_NAMES.get(plan["mode"], plan["mode"])
    out["quality_type"] = plan["quality_type"]
    out["material"] = plan["material"]
    out["orientation"] = OrderedDict([("id", plan["orientation"]["candidate_id"]),
                                      ("reason", plan["orientation"].get("reason", ""))])
    out["settings"] = [OrderedDict([("key", k), ("value", v["value"]), ("reason", v.get("reason", "")),
                                    ("source", v.get("source"))]) for k, v in result["settings"].items()]
    out["risks"] = plan.get("risks", [])
    out["design_suggestions"] = plan.get("design_suggestions", [])
    out["deviations"] = plan.get("deviations", [])
    out["questions_to_user"] = plan.get("questions_to_user", [])
    out["explanation"] = plan.get("explanation", "")
    out["mode_decision"] = OrderedDict([
        ("score", mc["score"]), ("threshold", mc["threshold"]), ("confidence", mc["confidence"]),
        ("evidence", ["%+.1f %s" % (e["weight"], e["text"]) for e in mc["evidence"]]),
    ])
    out["materials_ranking"] = [OrderedDict([("id", r["id"]), ("score", r["score"]),
                                             ("chain", (r.get("chains") or [""])[0])]) for r in kg["ranking"][:5]]
    out["material_conflicts"] = [c["text"] for c in kg.get("conflicts", [])]
    out["validation_warnings"] = result["validation"]["warnings"]
    if applied is not None:
        out["applied"] = applied
    if slice_result is not None:
        out["slice"] = slice_result
    return out


def _plan_print(ctx, args):
    task = str(args.get("task") or "").strip()
    if not task:
        raise ToolError("Передайте текст задания task")
    oid = _object_id(ctx, args)
    info = next(o for o in ctx.slicer.list_objects() if o["id"] == oid)
    triangles = ctx.slicer.get_triangles(oid)
    context = ctx.slicer.slicer_context()

    def on_event(event):
        ctx.events.emit("pipeline", stage=event["stage"], status=event["status"], summary=event["summary"])

    with ctx.state.lock:
        images = list(ctx.state.images) or None
    result = pipeline.run(task, triangles=triangles, llm=ctx.settings.llm(), ctx=context, images=images,
                          on_event=on_event)
    with ctx.state.lock:
        ctx.state.last_plan = result
        ctx.state.last_object_id = oid

    applied, slice_result = None, None
    if args.get("apply", True):
        applied = apply_plan(ctx, result, oid)
        if args.get("slice", True) and applied["ok"]:
            outcome = ctx.call("slice", {})
            slice_result = outcome.get("result") if outcome["ok"] else OrderedDict([("error", outcome["error"])])
    return plan_summary(result, applied, slice_result, info)


def apply_plan(ctx, result, object_id):
    """Execute a pipeline result in the slicer, one visible tool call per step."""
    plan = result["plan"]
    steps = []

    def step(name, args):
        outcome = ctx.call(name, args)
        steps.append(OrderedDict([("tool", name), ("ok", outcome["ok"]), ("error", outcome.get("error"))]))
        return outcome

    # Clearing user changes first means switching profile and material never asks
    # «discard or keep?», and nothing from an earlier plan leaks into this one.
    step("reset_settings", {})
    step("set_quality_profile", {"quality_type": plan["quality_type"]})
    step("set_material", {"material_id": plan["material"]})
    step("set_printing_mode", {"mode": plan["mode"]})
    values = OrderedDict((k, v["value"]) for k, v in result["settings"].items() if k != "printing_mode")
    if values:
        step("set_settings", {"settings": values})
    orientation = result["orientation"]
    step("apply_orientation", {"object_id": object_id, "rotation": orientation["rotation_stl"],
                               "translation": orientation["translation_stl"], "orientation_id": orientation["id"]})
    if plan["mode"] == "classic" and len(ctx.slicer.list_objects()) > 1:
        step("arrange_all", {})
    return OrderedDict([("ok", all(s["ok"] for s in steps)), ("steps", steps)])


# ----------------------------------------------------------------------------- registry

def build_registry():
    r = ToolRegistry()
    obj = ("object_id", {"type": "integer", "description": "id объекта из list_objects; можно не указывать, если модель одна"})
    r.add(Tool("plan_print", "План печати",
               "Главный инструмент. Анализирует модель и текст задания (геометрия, ось вращения, граф материалов, "
               "языковая модель), выбирает режим 3D/5D, профиль, материал, настройки и ориентацию, применяет их в "
               "слайсере и нарезает модель. Возвращает план с объяснением, рисками и вопросами. Передавайте в task "
               "полный текст задания пользователя.",
               _schema([("task", {"type": "string", "description": "Полный текст задания на печать"}), obj,
                        ("apply", {"type": "boolean", "description": "Применить план в слайсере (по умолчанию true)"}),
                        ("slice", {"type": "boolean", "description": "Нарезать после применения (по умолчанию true)"})],
                       ["task"]), _plan_print))
    r.add(Tool("get_state", "Состояние слайсера",
               "Принтер, режим печати, профиль качества, материал, изменённые настройки, модели на столе, "
               "состояние слайсинга.", _schema(), _get_state))
    r.add(Tool("list_quality_profiles", "Профили качества", "Список профилей качества (quality_type, высота слоя).",
               _schema(), _list_quality))
    r.add(Tool("list_materials", "Материалы", "Материалы, доступные для экструдера принтера.", _schema(),
               _list_materials))
    r.add(Tool("get_settings", "Чтение настроек", "Значения настроек слайсера по ключам (например layer_height).",
               _schema([("keys", {"type": "array", "items": {"type": "string"}})], ["keys"]), _get_settings))
    r.add(Tool("set_settings", "Изменение настроек",
               "Задать настройки слайсера: {ключ: значение}. Значения проверяются по пределам слайсера; "
               "ключи 5D-части имеют суффикс _cylindrical. Режим печати меняйте через set_printing_mode.",
               _schema([("settings", {"type": "object", "description": "Например {\"infill_sparse_density\": 30}"})],
                       ["settings"]), _set_settings))
    r.add(Tool("reset_settings", "Сброс настроек", "Сбросить все изменённые настройки к значениям профиля.",
               _schema(), _reset_settings))
    r.add(Tool("set_quality_profile", "Профиль качества",
               "Выбрать профиль качества по quality_type (high 0.06, normal 0.1, fast 0.15, draft 0.2, "
               "verydraft 0.3 мм ...).", _schema([("quality_type", {"type": "string"})], ["quality_type"]),
               _set_quality))
    r.add(Tool("set_material", "Материал", "Выбрать материал по идентификатору (см. list_materials).",
               _schema([("material_id", {"type": "string"}), ("extruder", {"type": "integer"})], ["material_id"]),
               _set_material))
    r.add(Tool("set_printing_mode", "Режим печати",
               "Переключить режим печати: classic (3D), cylindrical (5D Spiral), cylindrical_full (5D Spiral Full).",
               _schema([("mode", {"type": "string", "enum": MODES})], ["mode"]), _set_mode))
    r.add(Tool("save_profile", "Сохранение профиля", "Сохранить текущие изменения настроек как пользовательский профиль.",
               _schema([("name", {"type": "string"})], ["name"]), _save_profile))
    r.add(Tool("load_model", "Загрузка модели", "Загрузить файл модели (STL, OBJ, 3MF) на стол.",
               _schema([("path", {"type": "string"})], ["path"]), _load_model))
    r.add(Tool("model_from_photos", "Модель по фото",
               "Построить 3D-модель детали по фото (1–4 ракурса), когда файла модели нет, и поставить её на стол "
               "вместо прежних моделей. size_mm — наибольший габарит детали в мм, обязателен: по фото масштаб не "
               "определить. views — ракурсы фото по порядку (front, left, back, right; по умолчанию в этом порядке). "
               "Берутся фото, прикреплённые в чате, или photo_paths — пути к файлам. Генерация занимает до "
               "нескольких минут, форма приблизительная.",
               _schema([("size_mm", {"type": "number"}),
                        ("views", {"type": "array", "items": {"type": "string", "enum": list(photo3d.VIEWS)}}),
                        ("photo_paths", {"type": "array", "items": {"type": "string"}})], ["size_mm"]),
               _model_from_photos))
    r.add(Tool("list_objects", "Модели на столе", "Список моделей на столе: id, имя, габариты (мм).", _schema(),
               _list_objects))
    r.add(Tool("get_object_info", "Габариты модели", "Габариты и положение модели (мм, ось Z вверх).",
               _schema([obj]), _object_info))
    r.add(Tool("move_object", "Перемещение", "Переместить модель: центр в (x, y) от центра стола, низ на высоту z.",
               _schema([obj, ("x", {"type": "number"}), ("y", {"type": "number"}), ("z", {"type": "number"})]),
               _move_object))
    r.add(Tool("rotate_object", "Поворот", "Повернуть модель вокруг оси x, y или z на угол в градусах; "
               "после поворота модель ставится на стол.",
               _schema([obj, ("axis", {"type": "string", "enum": ["x", "y", "z"]}),
                        ("angle_deg", {"type": "number"})], ["axis", "angle_deg"]), _rotate_object))
    r.add(Tool("scale_object", "Масштаб", "Масштабировать модель: factor (одно число) или factors [x, y, z].",
               _schema([obj, ("factor", {"type": "number"}), ("factors", {"type": "array", "items": {"type": "number"}})]),
               _scale_object))
    r.add(Tool("arrange_all", "Расстановка", "Автоматически расставить модели на столе (только для режима 3D).",
               _schema(), _arrange_all))
    r.add(Tool("delete_object", "Удаление модели", "Удалить модель со стола.", _schema([obj]), _delete_object))
    r.add(Tool("clear_scene", "Очистка стола", "Удалить все модели со стола.", _schema(), _clear_scene))
    r.add(Tool("slice", "Слайсинг", "Нарезать модели с текущими настройками и дождаться результата: время печати и "
               "расход материала.", _schema(), _slice))
    r.add(Tool("get_slice_status", "Статус слайсинга", "Состояние и результат последнего слайсинга.", _schema(),
               _slice_status))
    r.add(Tool("export_gcode", "Экспорт G-code", "Сохранить G-code последнего слайсинга в файл (по умолчанию в папку "
               "вывода плагина).", _schema([("path", {"type": "string"})]), _export_gcode))
    r.add(Tool("save_project", "Сохранение проекта", "Сохранить проект 3MF (модель + все настройки).",
               _schema([("path", {"type": "string"})]), _save_project))
    r.add(Tool("apply_orientation", "Ориентация модели", "Поворот и размещение модели по плану (служебный).",
               _schema([obj, ("rotation", {"type": "array"}), ("translation", {"type": "array"}),
                        ("orientation_id", {"type": "string"})], ["rotation"]), _apply_orientation, public=False))
    return r
