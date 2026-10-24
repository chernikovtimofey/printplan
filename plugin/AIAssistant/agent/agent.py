"""Chat agent: one conversation with the LLM, which acts through the tools.

A turn: attachments are saved (models are loaded onto the bed right away, a
.docx/.txt task is read into the message, photos go to the model as images),
then the LLM is asked repeatedly — it either calls tools or answers. Without
an API key, or when the LLM fails, the turn falls back to a deterministic
`plan_print`, so the user still gets a configured project.
"""
import base64
import json
import os
import re
import tempfile
import threading
import time
import traceback
from collections import OrderedDict

from ..printplan.io_utils import read_task
from ..printplan.llm_client import stage_options
from .tools import format_duration

MAX_STEPS = 10
MAX_TOOL_RESULT_CHARS = 9000
MODEL_EXTENSIONS = (".stl", ".obj", ".3mf")
ACTIONS = OrderedDict([("export_gcode", "G-code"), ("save_project", "проект 3MF")])  # chat buttons
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
TEXT_EXTENSIONS = (".docx", ".txt", ".md")

SYSTEM_PROMPT = """Ты — ИИ-ассистент слайсера STE Slicer для принтера Stereotech Hybrid 530 V5.2.4 (режимы 3D, 5D Spiral и 5D Spiral Full). Ты помогаешь подготовить модель к печати и меняешь слайсер только через инструменты.

Как работать:
1. Если на столе есть модель и пользователь описал задачу печати (назначение детали, условия работы, требования, материал и т. п.), вызови plan_print и передай в task полный текст задания. Если задание пришло в нескольких сообщениях или во вложенном файле, объедини всё в task. plan_print сам выбирает режим, профиль, материал, настройки и ориентацию, применяет их и нарезает модель.
2. Если модели нет — попроси прикрепить STL, OBJ или 3MF. Если задачи нет — коротко спроси, для чего деталь и какие к ней требования.
3. После plan_print кратко объясни решение по данным результата: режим и почему, ориентация, материал, профиль и ключевые настройки, главные риски, время печати и расход материала. Передай пользователю questions_to_user и material_conflicts, если они есть. Карточка плана со всеми настройками уже показана в чате, не перечисляй их все.
4. Точечные изменения («заполнение 30 %», «поверни на бок», «смени материал») делай соответствующими инструментами, затем вызови slice и сообщи новое время печати и расход.
5. Сохраняй G-code (export_gcode) и проект (save_project), когда пользователь просит, и называй путь к файлу.
6. Не придумывай факты. Числа, свойства материалов и настройки называй только те, что есть в результатах инструментов; материалы называй их названиями из результата (например, Enduse PC). Если инструмент вернул ошибку, объясни её и предложи, что сделать.

Отвечай по-русски, кратко и по делу, без таблиц."""


class BusyError(Exception):
    pass


def _safe_name(name):
    base = os.path.basename(name or "file")
    base = re.sub(r"[^\w\-. ]+", "_", base, flags=re.U).strip()
    return base or "file"


class Session(object):
    def __init__(self, registry, context, settings, events, upload_dir=None):
        self.registry = registry
        self.context = context
        self.settings = settings
        self.events = events
        self.history = []
        self._upload_dir = upload_dir
        self._lock = threading.Lock()
        self._busy = False
        self._thread = None
        self._notes = []  # what the user did with buttons, told to the LLM with the next message

    @property
    def busy(self):
        return self._busy

    def reset(self):
        if self._busy:
            raise BusyError("Ассистент ещё отвечает")
        self.history = []
        self._notes = []
        with self.context.state.lock:
            self.context.state.images = []
            self.context.state.last_plan = None
        self.events.emit("reset")

    def run_action(self, name):
        """A button in the chat (save G-code or project): runs the tool directly, without the LLM."""
        if name not in ACTIONS:
            raise ValueError("Неизвестное действие: %s" % name)
        with self._lock:
            if self._busy:
                raise BusyError("Ассистент ещё отвечает")
            self._busy = True
        self.events.emit("turn_start")
        try:
            outcome = self.registry.call(name, {}, self.context, source="user")
            if outcome["ok"]:
                self._notes.append("Пользователь кнопкой в чате сохранил %s: %s" % (ACTIONS[name], outcome["result"]["path"]))
            return outcome
        finally:
            with self._lock:
                self._busy = False
            self.events.emit("turn_done")

    def start_turn(self, text, attachments=None):
        """Run a turn in a background thread; raises BusyError while another one runs."""
        with self._lock:
            if self._busy:
                raise BusyError("Ассистент ещё отвечает на предыдущее сообщение")
            self._busy = True
        self._thread = threading.Thread(target=self._run, args=(text, attachments or []), name="ai-assistant-turn")
        self._thread.daemon = True
        self._thread.start()

    def run_turn(self, text, attachments=None):
        """Synchronous turn (tests)."""
        with self._lock:
            if self._busy:
                raise BusyError("Ассистент ещё отвечает")
            self._busy = True
        self._run(text, attachments or [])

    # ------------------------------------------------------------------ turn
    def _run(self, text, attachments):
        self.events.emit("turn_start")
        try:
            self._turn(text or "", attachments)
        except Exception as e:
            self.events.emit("error", text="Внутренняя ошибка ассистента: %s" % e, trace=traceback.format_exc()[-3000:])
        finally:
            self._strip_images_from_history()
            with self._lock:
                self._busy = False
            self.events.emit("turn_done")

    def _turn(self, text, attachments):
        files = self._save_attachments(attachments)
        self.events.emit("user", text=text, attachments=[OrderedDict([("name", f["name"]), ("kind", f["kind"])])
                                                         for f in files])
        notes, self._notes = self._notes, []
        for f in files:
            if f["kind"] == "error":
                notes.append("Вложение %s не принято: %s" % (f["name"], f["error"]))
        task_texts = []
        for f in files:
            if f["kind"] == "text":
                try:
                    content = read_task(f["path"]).strip()
                except Exception as e:
                    notes.append("Не удалось прочитать %s: %s" % (f["name"], e))
                    continue
                task_texts.append("Задание из файла %s:\n%s" % (f["name"], content))
        loaded = []
        models = [f for f in files if f["kind"] == "model"]
        if models and self.context.slicer.list_objects():
            # The chat works on one part at a time: a newly attached model replaces the previous ones.
            outcome = self.registry.call("clear_scene", {}, self.context, source="agent")
            if outcome["ok"]:
                notes.append("Прежние модели убраны со стола")
            else:
                notes.append("Не удалось убрать прежние модели со стола: %s" % outcome["error"])
        for f in models:
            outcome = self.registry.call("load_model", {"path": f["path"]}, self.context, source="agent")
            if outcome["ok"]:
                loaded.extend(outcome["result"]["loaded"])
            else:
                notes.append("Модель %s не загружена: %s" % (f["name"], outcome["error"]))
        images = [f["data_url"] for f in files if f["kind"] == "image"]
        with self.context.state.lock:
            self.context.state.images = images

        full_text = "\n\n".join([t for t in [text.strip()] + task_texts if t])
        if loaded:
            notes.append("На стол загружены модели: %s" % ", ".join(
                "id=%s «%s», габариты %s мм" % (o["id"], o["name"], o.get("size_mm")) for o in loaded))
        llm = self.settings.llm()
        if llm is None:
            self._offline_turn(full_text, reason="языковая модель не подключена — укажите ключ API в настройках")
            return
        content = full_text
        if notes:
            content = (content + "\n\n" if content else "") + "[" + "; ".join(notes) + "]"
        if images:
            parts = [{"type": "text", "text": content or "(изображение)"}]
            parts.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
            self.history.append({"role": "user", "content": parts})
        else:
            self.history.append({"role": "user", "content": content})
        self._turn_task = full_text
        try:
            self._agent_loop(llm)
        except Exception as e:
            self.events.emit("error", text="Языковая модель недоступна: %s" % e)
            if not self._planned_this_turn and full_text:
                self._offline_turn(full_text, reason="ошибка обращения к языковой модели")

    def _agent_loop(self, llm):
        self._planned_this_turn = False
        tools = self.registry.openai_tools()
        options = stage_options(getattr(llm, "model", None), "agent")
        for _ in range(MAX_STEPS):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
            self.events.emit("thinking")
            message, meta = llm.chat_tools(messages, tools, extra_body=options, on_progress=self._on_progress)
            calls = message.get("tool_calls") or []
            entry = {"role": "assistant", "content": message.get("content") or ""}
            if calls:
                entry["tool_calls"] = calls
            self.history.append(entry)
            if not calls:
                self.events.emit("assistant", text=(message.get("content") or "").strip())
                return
            if message.get("content"):
                self.events.emit("assistant_note", text=message["content"].strip())
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name")
                args = self._parse_args(function.get("arguments"))
                outcome = self.registry.call(name, args, self.context, source="agent", call_id=call.get("id"))
                if name == "plan_print":
                    self._planned_this_turn = True
                self.history.append({"role": "tool", "tool_call_id": call.get("id"),
                                     "content": self._for_llm(outcome)})
        self.events.emit("assistant", text="Остановился после %d шагов. Уточните, что сделать дальше." % MAX_STEPS)

    def _offline_turn(self, text, reason):
        """Deterministic behaviour without the LLM."""
        objects = self.context.slicer.list_objects()
        if not objects:
            self.events.emit("assistant", text="Прикрепите модель (STL, OBJ или 3MF) и опишите задачу печати. "
                                               "(%s)" % reason)
            return
        if not text.strip():
            self.events.emit("assistant", text="Модель на столе. Опишите, для чего деталь и какие к ней требования. "
                                               "(%s)" % reason)
            return
        outcome = self.registry.call("plan_print", {"task": text}, self.context, source="agent")
        self._planned_this_turn = True
        if not outcome["ok"]:
            self.events.emit("assistant", text="Не удалось построить план: %s" % outcome["error"])
            return
        self.events.emit("assistant", text=offline_reply(outcome["result"], reason))

    # ------------------------------------------------------------------ helpers
    _planned_this_turn = False
    _turn_task = ""

    def _on_progress(self, progress):
        delta = progress.get("text_delta")
        if delta:
            self.events.emit("assistant_delta", text=delta)

    @staticmethod
    def _parse_args(arguments):
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        try:
            value = json.loads(arguments)
            return value if isinstance(value, dict) else {}
        except ValueError:
            return {}

    @staticmethod
    def _for_llm(outcome):
        payload = outcome.get("result") if outcome.get("ok") else {"error": outcome.get("error")}
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = text[:MAX_TOOL_RESULT_CHARS] + "…(обрезано)"
        return text

    def _strip_images_from_history(self):
        """Photos are sent once; later turns keep only a placeholder (keeps requests small)."""
        for message in self.history:
            if isinstance(message.get("content"), list):
                texts = [p.get("text", "") for p in message["content"] if p.get("type") == "text"]
                message["content"] = " ".join(texts) + " [приложено фото]"

    def _save_attachments(self, attachments):
        files = []
        folder = None
        for item in attachments:
            name = _safe_name(item.get("name"))
            ext = os.path.splitext(name)[1].lower()
            if ext in MODEL_EXTENSIONS:
                kind = "model"
            elif ext in IMAGE_EXTENSIONS:
                kind = "image"
            elif ext in TEXT_EXTENSIONS:
                kind = "text"
            else:
                files.append({"name": name, "kind": "error", "error": "неподдерживаемый тип файла"})
                continue
            try:
                data = base64.b64decode(item.get("data") or "")
            except Exception:
                files.append({"name": name, "kind": "error", "error": "повреждённые данные"})
                continue
            if kind == "image":
                mime = "image/png" if ext == ".png" else "image/jpeg"
                files.append({"name": name, "kind": "image",
                              "data_url": "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))})
                continue
            if folder is None:
                folder = self._new_upload_folder()
            path = os.path.join(folder, name)
            with open(path, "wb") as f:
                f.write(data)
            files.append({"name": name, "kind": kind, "path": path})
        return files

    def _new_upload_folder(self):
        base = self._upload_dir or os.path.join(self.settings.output_dir(), "uploads")
        if not os.path.isdir(base):
            os.makedirs(base)
        return tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=base)


def offline_reply(summary, reason):
    """Text answer built from a plan summary (no LLM)."""
    lines = []
    lines.append("Режим: %s; профиль: %s; материал: %s; ориентация: %s." % (
        summary["mode_name"], summary["quality_type"], summary["material"], summary["orientation"]["id"]))
    sl = summary.get("slice") or {}
    if sl.get("error"):
        lines.append("Слайсинг не выполнен: %s" % sl["error"])
    elif sl:
        lengths = sl.get("material_length_m") or []
        weights = sl.get("material_weight_g") or []
        lines.append("Время печати: %s; материал: %s м%s." % (
            format_duration(sl.get("print_time_s")) or "—", lengths[0] if lengths else "—",
            (", %s г" % weights[0]) if weights else ""))
    explanation = summary.get("explanation") or ""
    if "по правилам" not in explanation:
        lines.append("План построен по правилам (%s)." % reason)
    if explanation:
        lines.append(explanation)
    for q in summary.get("questions_to_user") or []:
        lines.append("Вопрос: %s" % q)
    if "ключ API" in reason:
        lines.append("Чтобы подключить языковую модель, укажите ключ API в настройках (⚙).")
    return "\n\n".join(lines)
