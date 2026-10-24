"""Model from photos: the plugin side (client, tool, agent) against a stand-in service."""
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np
import pytest

import meshes
from AIAssistant.agent import photo3d
from AIAssistant.agent.agent import Session, size_from_text
from AIAssistant.agent.config import Settings
from AIAssistant.agent.events import EventLog
from AIAssistant.agent.slicer_fake import FakeSlicer
from AIAssistant.agent.tools import ToolContext, build_registry
from AIAssistant.printplan import io_utils
from AIAssistant.printplan.llm_client import ScriptedLLM


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


@pytest.fixture
def service(tmpdir):
    """Stand-in for services/photo3d: returns a 1 x 0.5 x 0.4 box (largest extent 1, Z up)."""
    path = str(tmpdir.join("unit.stl"))
    io_utils.write_stl(path, meshes.box((1.0, 0.5, 0.4)))
    with open(path, "rb") as f:
        stl = base64.b64encode(f.read()).decode("ascii")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._reply(200, {"ready": True})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
            requests.append(body)
            self._reply(200, {"stl": stl, "triangles": 12, "seconds": 0.1, "model": "stand-in"})

        def _reply(self, code, data):
            raw = json.dumps(data).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    srv = _Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.url = "http://127.0.0.1:%d" % srv.server_address[1]
    srv.requests = requests
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def env(tmpdir, service):
    settings = Settings({"output_dir": str(tmpdir), "photo3d_url": service.url}, use_env=False)
    registry = build_registry()
    events = EventLog()
    ctx = ToolContext(FakeSlicer(), settings, events, registry)
    return registry, ctx, events, settings


def _photo(name="wheel.jpg"):
    return {"name": name, "data": base64.b64encode(b"\xff\xd8 not really a jpeg").decode("ascii")}


def _session(env, llm):
    registry, ctx, events, settings = env
    agent_settings = Settings({"output_dir": settings.get("output_dir")}, use_env=False)
    agent_settings.llm = lambda: llm
    return Session(registry, ctx, agent_settings, events)


def _extent(ctx):
    pts = ctx.slicer.get_triangles(ctx.slicer.list_objects()[0]["id"]).reshape(-1, 3)
    return pts.max(axis=0) - pts.min(axis=0)


def test_scale_to_size():
    tris = photo3d.scale_to_size(meshes.box((1.0, 0.5, 0.4), origin=(3, 3, 3)), 120)
    pts = tris.reshape(-1, 3)
    assert np.allclose(pts.max(axis=0) - pts.min(axis=0), [120, 60, 48])
    assert pts[:, 2].min() == pytest.approx(0) and np.allclose((pts.max(axis=0) + pts.min(axis=0))[:2], 0)
    with pytest.raises(ValueError):
        photo3d.scale_to_size(meshes.box(), 0)


def test_size_from_text():
    assert size_from_text("Колесо, диаметр 135 мм, отверстие 20 мм") == 135
    assert size_from_text("длина 12,5 см") == 125
    assert size_from_text("допуск 0.1 мм, без размеров") is None


def test_tool_builds_scaled_model_and_replaces_the_bed(env, service):
    registry, ctx, events, _ = env
    ctx.slicer.add_triangles("old.stl", meshes.box())
    ctx.state.photos = [{"name": "wheel.jpg", "data": b"a"}, {"name": "side.png", "data": b"b"}]
    out = registry.call("model_from_photos", {"size_mm": 120}, ctx)
    assert out["ok"], out.get("error")
    assert [o["name"] for o in ctx.slicer.list_objects()] == ["wheel_по_фото.stl"]
    assert np.allclose(_extent(ctx), [120, 60, 48])
    assert os.path.isfile(out["result"]["path"]) and "приблизительная" in out["result"]["warning"]
    assert [i["view"] for i in service.requests[0]["images"]] == ["front", "left"]
    assert base64.b64decode(service.requests[0]["images"][1]["data"]) == b"b"
    steps = [(e["name"], e["source"]) for e in events.since(0) if e["kind"] == "tool_call"]
    assert steps == [("model_from_photos", "agent"), ("clear_scene", "plan"), ("load_model", "plan")]
    assert len(ctx.state.images) == 2 and ctx.state.images[1].startswith("data:image/png;base64,")


def test_tool_explicit_views_and_paths(env, service, tmpdir):
    registry, ctx, _, _ = env
    paths = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        paths.append(str(tmpdir.join(name)))
        with open(paths[-1], "wb") as f:
            f.write(name.encode("ascii"))
    out = registry.call("model_from_photos", {"size_mm": 50, "views": ["front", "back"], "photo_paths": paths}, ctx)
    assert out["ok"], out.get("error")
    assert [i["view"] for i in service.requests[0]["images"]] == ["front", "back"]  # third photo not used
    assert out["result"]["photos"] == ["a.jpg", "b.jpg"]


@pytest.mark.parametrize("args,state_photos,message", [
    ({}, True, "наибольший размер"),
    ({"size_mm": 100}, False, "Нет фото"),
    ({"size_mm": 100, "views": ["top"]}, True, "ракурсы"),
])
def test_tool_errors(env, args, state_photos, message):
    registry, ctx, _, _ = env
    if state_photos:
        ctx.state.photos = [{"name": "p.jpg", "data": b"x"}]
    out = registry.call("model_from_photos", args, ctx)
    assert not out["ok"] and message in out["error"]


def test_tool_reports_unreachable_service(env):
    registry, ctx, _, settings = env
    settings.set("photo3d_url", "http://127.0.0.1:9")
    ctx.state.photos = [{"name": "p.jpg", "data": b"x"}]
    out = registry.call("model_from_photos", {"size_mm": 100}, ctx)
    assert not out["ok"] and "недоступен" in out["error"] and "127.0.0.1:9" in out["error"]
    assert ctx.slicer.list_objects() == []


def test_offline_photo_with_size_builds_model_and_plans(env):
    registry, ctx, events, _ = env
    _session(env, None).run_turn("Колесо конвейера, диаметр 60 мм", [_photo()])
    names = [e["name"] for e in events.since(0) if e["kind"] == "tool_call" and e["source"] == "agent"]
    assert names == ["model_from_photos", "plan_print"]
    answer = [e["text"] for e in events.since(0) if e["kind"] == "assistant"][-1]
    assert "построена по фото" in answer and "60 мм" in answer and "Режим:" in answer


def test_offline_asks_for_the_size_then_builds(env):
    registry, ctx, events, _ = env
    session = _session(env, None)
    session.run_turn("Втулка, рабочая температура до +80 °C", [_photo()])
    answer = [e["text"] for e in events.since(0) if e["kind"] == "assistant"][-1]
    assert "наибольший размер" in answer and ctx.slicer.list_objects() == []
    session.run_turn("Наибольший размер 40 мм")
    calls = [e for e in events.since(0) if e["kind"] == "tool_call" and e["source"] == "agent"]
    assert [e["name"] for e in calls] == ["model_from_photos", "plan_print"]
    assert "+80" in calls[1]["args"]["task"] and "40 мм" in calls[1]["args"]["task"]  # both messages


def test_offline_photo_of_the_model_on_the_bed_is_extra_material(env):
    registry, ctx, events, _ = env
    ctx.slicer.add_triangles("part.stl", meshes.box((30, 20, 10)))
    _session(env, None).run_turn("Вот фото этой детали, нужна точная посадка", [_photo()])
    names = [e["name"] for e in events.since(0) if e["kind"] == "tool_call" and e["source"] == "agent"]
    assert names == ["plan_print"]


def test_agent_calls_model_from_photos(env):
    registry, ctx, events, _ = env
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [{"name": "model_from_photos", "arguments": {"size_mm": 80}}]},
        {"content": "", "tool_calls": [{"name": "plan_print", "arguments": {"task": "Ролик, диаметр 80 мм"}}]},
        {"content": "Модель построена по фото и приблизительная."},
    ])
    _session(env, llm).run_turn("Ролик, диаметр 80 мм", [_photo("roller.jpg")])
    names = [e["name"] for e in events.since(0) if e["kind"] == "tool_call" and e["source"] == "agent"]
    assert names == ["model_from_photos", "plan_print"]
    first_user = llm.calls[0]["messages"][1]["content"]  # photos are stripped from the history after the turn
    text = first_user if isinstance(first_user, str) else first_user[0]["text"]
    assert "Прикреплены фото без файла модели: roller.jpg; на столе нет моделей" in text
    assert max(_extent(ctx)) == pytest.approx(80)


def test_agent_refuses_a_size_the_user_did_not_give(env):
    registry, ctx, events, _ = env
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [{"name": "model_from_photos", "arguments": {"size_mm": 150}}]},
        {"content": "Какой наибольший размер детали в мм?"},
        {"content": "", "tool_calls": [{"name": "model_from_photos", "arguments": {"size_mm": 120}}]},
        {"content": "Модель построена по фото."},
    ])
    session = _session(env, llm)
    session.run_turn("Нужно напечатать это колесо, работает при +85 °C", [_photo()])
    assert not any(e["kind"] == "tool_call" for e in events.since(0))  # the guessed size was not used
    refusal = [m for m in session.history if m["role"] == "tool"][0]["content"]
    assert "150 мм не назван пользователем" in refusal
    assert ctx.slicer.list_objects() == []
    session.run_turn("120")  # a bare number as the answer is enough
    names = [e["name"] for e in events.since(0) if e["kind"] == "tool_call" and e["source"] == "agent"]
    assert names == ["model_from_photos"] and max(_extent(ctx)) == pytest.approx(120)
