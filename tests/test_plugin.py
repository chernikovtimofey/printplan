"""Plugin layer: tools, MCP server, chat agent and HTTP server on the fake slicer."""
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

import meshes
from AIAssistant.agent.agent import BusyError, Session
from AIAssistant.agent.config import Settings
from AIAssistant.agent.events import EventLog
from AIAssistant.agent.mcp import MCPServer
from AIAssistant.agent.server import AssistantServer
from AIAssistant.agent.slicer_fake import FakeSlicer
from AIAssistant.agent.tools import ToolContext, build_registry
from AIAssistant.printplan import io_utils
from AIAssistant.printplan.llm_client import LLMError, ScriptedLLM


@pytest.fixture
def env(tmpdir):
    settings = Settings({"output_dir": str(tmpdir)}, use_env=False)
    registry = build_registry()
    events = EventLog()
    ctx = ToolContext(FakeSlicer(), settings, events, registry)
    return registry, ctx, events, settings


def call(registry, ctx, name, **args):
    out = registry.call(name, args, ctx)
    assert out["ok"], out.get("error")
    return out["result"]


def size_xyz(ctx, oid):
    pts = ctx.slicer.get_triangles(oid).reshape(-1, 3)
    return pts.max(axis=0) - pts.min(axis=0), (pts.max(axis=0) + pts.min(axis=0)) / 2.0


# ----------------------------------------------------------------------------- tools

def test_tool_lists(env):
    registry = env[0]
    names = [t["function"]["name"] for t in registry.openai_tools()]
    for required in ("plan_print", "set_settings", "set_printing_mode", "set_quality_profile", "set_material",
                     "load_model", "rotate_object", "arrange_all", "slice", "export_gcode", "save_project"):
        assert required in names
    assert "apply_orientation" not in names            # internal step of plan_print
    assert [t["name"] for t in registry.mcp_tools()] == names


def test_set_settings_validation(env):
    registry, ctx, events, _ = env
    res = call(registry, ctx, "set_settings", settings={"wall_line_count": "3", "infill_sparse_density": 150,
                                                        "machine_width": 10, "no_such_key": 1, "printing_mode": "classic"})
    assert res["applied"]["wall_line_count"]["value"] == 3          # coerced from "3"
    assert "infill_sparse_density" in res["warnings"]
    assert set(res["rejected"]) == {"machine_width", "no_such_key", "printing_mode"}


def test_profiles_materials_and_mode(env):
    registry, ctx, events, _ = env
    assert call(registry, ctx, "get_state")["printing_mode"] == "cylindrical_full"   # V5.2.4 default
    call(registry, ctx, "set_printing_mode", mode="classic")
    call(registry, ctx, "set_quality_profile", quality_type="draft")
    call(registry, ctx, "set_material", material_id="enduse_petg")
    state = call(registry, ctx, "get_state")
    assert (state["printing_mode"], state["quality_type"], state["material"]) == ("classic", "draft", "enduse_petg")
    assert not registry.call("set_printing_mode", {"mode": "spherical"}, ctx)["ok"]
    assert not registry.call("set_material", {"material_id": "unobtanium"}, ctx)["ok"]


def test_object_tools(env, tmpdir):
    registry, ctx, events, _ = env
    path = str(tmpdir.join("box.stl"))
    io_utils.write_stl(path, meshes.box((10, 20, 30)))
    oid = call(registry, ctx, "load_model", path=path)["loaded"][0]["id"]
    size, center = size_xyz(ctx, oid)
    assert np.allclose(size, [10, 20, 30]) and np.allclose(center[:2], [0, 0])
    call(registry, ctx, "rotate_object", axis="x", angle_deg=90)
    size, center = size_xyz(ctx, oid)
    assert np.allclose(size, [10, 30, 20], atol=1e-6)
    assert ctx.slicer.get_triangles(oid)[..., 2].min() == pytest.approx(0, abs=1e-9)
    call(registry, ctx, "move_object", x=50, y=-20)
    size, center = size_xyz(ctx, oid)
    assert np.allclose(center[:2], [50, -20])
    call(registry, ctx, "scale_object", factor=2)
    assert np.allclose(size_xyz(ctx, oid)[0], [20, 60, 40], atol=1e-6)
    # two objects: object_id becomes mandatory unless the last loaded one is meant
    ctx.slicer.add_triangles("other", meshes.box())
    ctx.state.last_object_id = None
    assert "несколько" in registry.call("get_object_info", {}, ctx)["error"]


def test_plan_print_5d_bushing(env):
    registry, ctx, events, _ = env
    # The bushing is loaded lying on its side; the plan must stand its axis up through the bed centre.
    tube = meshes.rotate(meshes.tube(20, 8, 15), [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    oid = ctx.slicer.add_triangles("bushing.stl", tube)["id"]
    res = call(registry, ctx, "plan_print", task="Втулка ролика конвейера")
    assert res["mode"] == "cylindrical" and res["applied"]["ok"]
    assert res["slice"]["state"] == "done" and res["slice"]["print_time_s"] > 0
    state = ctx.slicer.state()
    assert state["printing_mode"] == "cylindrical"
    assert state["user_settings"]["non_printing_base_diameter"] == pytest.approx(16.0, abs=0.1)
    size, center = size_xyz(ctx, oid)
    assert size[2] == pytest.approx(15.0, abs=1e-6)            # axis vertical
    assert np.allclose(center[:2], [0, 0], atol=1e-6)            # axis through the bed centre
    steps = [e["name"] for e in events.since(0) if e["kind"] == "tool_call" and e["source"] == "plan"]
    assert steps[:4] == ["reset_settings", "set_quality_profile", "set_material", "set_printing_mode"]
    assert "apply_orientation" in steps and steps[-1] == "slice"
    assert any(e["kind"] == "pipeline" for e in events.since(0))


def test_plan_print_respects_support_ban(env):
    registry, ctx, events, _ = env
    ctx.slicer.add_triangles("t.stl", meshes.t_overhang())
    res = call(registry, ctx, "plan_print", task="Кронштейн, PETG, без поддержек", slice=False)
    assert res["mode"] == "classic" and res["material"] == "enduse_petg"
    assert ctx.slicer.state()["user_settings"]["support_enable"] is False
    assert "slice" not in res


def test_export_and_save(env, tmpdir):
    registry, ctx, events, settings = env
    ctx.slicer.add_triangles("part.stl", meshes.box())
    assert not registry.call("export_gcode", {}, ctx)["ok"]       # not sliced yet
    call(registry, ctx, "slice")
    gcode = call(registry, ctx, "export_gcode")["path"]
    project = call(registry, ctx, "save_project")["path"]
    assert gcode == os.path.join(str(tmpdir), "part.gcode") and os.path.isfile(gcode)
    assert project.endswith("part.3mf") and os.path.isfile(project)
    progress = [e["progress"] for e in events.since(0) if e["kind"] == "slice_progress"]
    assert progress[0] == 0.0 and progress[-1] == 1.0


def test_tool_errors_are_reported_not_raised(env):
    registry, ctx, events, _ = env
    out = registry.call("no_such_tool", {}, ctx)
    assert not out["ok"]
    out = registry.call("plan_print", {"task": "что-нибудь"}, ctx)        # empty bed
    assert not out["ok"] and "модел" in out["error"]
    kinds = [e["kind"] for e in events.since(0)]
    assert kinds.count("tool_call") == kinds.count("tool_result") == 2


# ----------------------------------------------------------------------------- MCP

def test_mcp_protocol(env):
    registry, ctx, events, _ = env
    mcp = MCPServer(registry, ctx)

    def rpc(method, params=None, msg_id=1):
        body = {"jsonrpc": "2.0", "method": method}
        if msg_id is not None:
            body["id"] = msg_id
        if params is not None:
            body["params"] = params
        text = mcp.handle_text(json.dumps(body))
        return json.loads(text) if text else None

    init = rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t"}})
    assert init["result"]["capabilities"]["tools"] is not None
    assert rpc("notifications/initialized", msg_id=None) is None
    tools = rpc("tools/list")["result"]["tools"]
    assert any(t["name"] == "plan_print" and "inputSchema" in t for t in tools)
    res = rpc("tools/call", {"name": "set_printing_mode", "arguments": {"mode": "classic"}})["result"]
    assert res["isError"] is False and json.loads(res["content"][0]["text"])["printing_mode"] == "classic"
    res = rpc("tools/call", {"name": "set_printing_mode", "arguments": {"mode": "bad"}})["result"]
    assert res["isError"] is True
    assert rpc("tools/call", {"name": "apply_orientation", "arguments": {}})["error"]["code"] == -32602
    assert rpc("no/such")["error"]["code"] == -32601
    assert json.loads(mcp.handle_text("{not json"))["error"]["code"] == -32700
    batch = json.loads(mcp.handle_text(json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"},
                                                   {"jsonrpc": "2.0", "method": "notifications/initialized"}])))
    assert len(batch) == 1
    assert any(e["kind"] == "tool_call" and e["source"] == "mcp" for e in events.since(0))


# ----------------------------------------------------------------------------- agent

def _stl_attachment(name, tris):
    buf = io.BytesIO()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_" + name)
    io_utils.write_stl(path, tris)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    os.remove(path)
    return {"name": name, "data": data}


def _session(env, llm):
    """The agent talks to `llm`; tools (plan_print) keep the env settings, i.e. no LLM."""
    registry, ctx, events, settings = env
    agent_settings = Settings({"output_dir": settings.get("output_dir")}, use_env=False)
    agent_settings.llm = lambda: llm
    return Session(registry, ctx, agent_settings, events)


def test_agent_tool_loop(env):
    registry, ctx, events, settings = env
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [{"name": "plan_print", "arguments": {"task": "Втулка ролика"}}]},
        {"content": "Выбран режим 5D Spiral: втулка — тело вращения."},
    ])
    session = _session(env, llm)
    session.run_turn("Втулка ролика", [_stl_attachment("bushing.stl", meshes.tube(20, 8, 15))])
    evs = events.since(0)
    kinds = [e["kind"] for e in evs]
    assert kinds[0] == "turn_start" and kinds[-1] == "turn_done"
    names = [e["name"] for e in evs if e["kind"] == "tool_call" and e["source"] == "agent"]
    assert names == ["load_model", "plan_print"]
    assert [e["text"] for e in evs if e["kind"] == "assistant"] == ["Выбран режим 5D Spiral: втулка — тело вращения."]
    # the LLM got the tool result back with the matching tool_call_id
    tool_msgs = [m for m in session.history if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["tool_call_id"] == session.history[1]["tool_calls"][0]["id"]
    assert '"mode":"cylindrical"' in tool_msgs[0]["content"]
    # the first request told the model which object was loaded
    first_user = llm.calls[0]["messages"][1]["content"]
    assert "bushing.stl" in first_user and "id=" in first_user


def test_agent_reads_docx_task(env, tmpdir):
    import zipfile
    path = str(tmpdir.join("task.docx"))
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", '<w:document xmlns:w="w"><w:body><w:p><w:r><w:t>Рабочая температура до +85°C'
                                         '</w:t></w:r></w:p></w:body></w:document>')
    with open(path, "rb") as f:
        att = {"name": "ТЗ.docx", "data": base64.b64encode(f.read()).decode("ascii")}
    llm = ScriptedLLM([{"content": "Понял задачу. Прикрепите модель."}])
    _session(env, llm).run_turn("Вот задание", [att])
    assert "Рабочая температура до +85°C" in llm.calls[0]["messages"][1]["content"]


def test_agent_without_llm_plans_deterministically(env):
    registry, ctx, events, settings = env
    session = _session(env, None)
    session.run_turn("Втулка ролика", [_stl_attachment("b.stl", meshes.tube(20, 8, 15))])
    answer = [e["text"] for e in events.since(0) if e["kind"] == "assistant"][-1]
    assert "по правилам" in answer and "5D Spiral" in answer
    assert ctx.slicer.state()["printing_mode"] == "cylindrical"


def test_agent_falls_back_when_llm_fails(env):
    registry, ctx, events, settings = env
    session = _session(env, ScriptedLLM([LLMError("timeout")]))
    session.run_turn("Кронштейн под нагрузку", [_stl_attachment("l.stl", meshes.l_bracket())])
    evs = events.since(0)
    assert any(e["kind"] == "error" for e in evs)
    assert any(e["kind"] == "tool_call" and e["name"] == "plan_print" for e in evs)
    assert evs[-1]["kind"] == "turn_done" and not session.busy


def test_new_model_replaces_the_previous_one(env):
    registry, ctx, events, settings = env
    session = _session(env, None)
    session.run_turn("Втулка ролика", [_stl_attachment("first.stl", meshes.tube(20, 8, 15))])
    session.run_turn("Теперь кронштейн", [_stl_attachment("second.stl", meshes.l_bracket())])
    assert [o["name"] for o in ctx.slicer.list_objects()] == ["second.stl"]
    calls = [e["name"] for e in events.since(0) if e["kind"] == "tool_call" and e["source"] == "agent"]
    assert calls.count("clear_scene") == 1
    assert calls.index("clear_scene") == calls.index("load_model", calls.index("load_model") + 1) - 1


def test_models_attached_together_are_all_loaded(env):
    registry, ctx, events, settings = env
    llm = ScriptedLLM([{"content": "Обе модели на столе."}])
    _session(env, llm).run_turn("Две детали", [_stl_attachment("a.stl", meshes.box()),
                                               _stl_attachment("b.stl", meshes.box((5, 5, 5)))])
    assert sorted(o["name"] for o in ctx.slicer.list_objects()) == ["a.stl", "b.stl"]
    assert not any(e["kind"] == "tool_call" and e["name"] == "clear_scene" for e in events.since(0))


def test_save_buttons_run_without_the_llm_and_tell_it_afterwards(env):
    registry, ctx, events, settings = env
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [{"name": "plan_print", "arguments": {"task": "Втулка ролика"}}]},
        {"content": "Готово."},
        {"content": "Файл лежит в папке вывода."},
    ])
    session = _session(env, llm)
    session.run_turn("Втулка ролика", [_stl_attachment("bushing.stl", meshes.tube(20, 8, 15))])
    calls_before = len(llm.calls)
    outcome = session.run_action("export_gcode")
    assert outcome["ok"] and os.path.isfile(outcome["result"]["path"])
    assert outcome["result"]["path"].endswith("bushing.gcode")
    assert session.run_action("save_project")["result"]["path"].endswith("bushing.3mf")
    assert len(llm.calls) == calls_before  # the buttons do not call the LLM
    clicks = [e for e in events.since(0) if e["kind"] == "tool_call" and e["source"] == "user"]
    assert [e["name"] for e in clicks] == ["export_gcode", "save_project"]
    session.run_turn("Где файл?")
    question = llm.calls[-1]["messages"][-1]["content"]
    assert "сохранил G-code" in question and "bushing.gcode" in question and "проект 3MF" in question
    session.run_turn("И ещё раз?")
    assert "сохранил" not in llm.calls[-1]["messages"][-1]["content"]  # told once


def test_save_buttons_reject_other_tools_and_busy_session(env):
    session = _session(env, None)
    with pytest.raises(ValueError):
        session.run_action("clear_scene")
    session._busy = True
    with pytest.raises(BusyError):
        session.run_action("export_gcode")


def test_agent_rejects_unknown_attachment(env):
    llm = ScriptedLLM([{"content": "Этот файл не поддерживается."}])
    _session(env, llm).run_turn("", [{"name": "virus.exe", "data": "AAAA"}])
    assert "не принято" in llm.calls[0]["messages"][1]["content"]


# ----------------------------------------------------------------------------- HTTP server

@pytest.fixture
def server(tmpdir):
    settings = Settings({"output_dir": str(tmpdir)}, use_env=False)
    srv = AssistantServer(FakeSlicer(), settings, port=0).start()
    yield srv
    srv.stop()


_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy 127.0.0.1


def http(server, path, body=None, ctype="application/json", method=None):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
    req = urllib.request.Request(server.url.rstrip("/") + path, data=data, method=method or ("POST" if data is not None else "GET"))
    if data is not None:
        req.add_header("Content-Type", ctype)
    try:
        with _NO_PROXY.open(req, timeout=30) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type")


def test_server_static_and_state(server):
    status, body, ctype = http(server, "/")
    assert status == 200 and b"AI Assistant" in body and ctype.startswith("text/html")
    assert http(server, "/app.js")[0] == 200 and http(server, "/style.css")[0] == 200
    assert http(server, "/../agent/server.py")[0] == 404
    status, body, _ = http(server, "/api/state")
    state = json.loads(body.decode("utf-8"))
    assert state["settings"]["has_api_key"] is False and state["mcp_url"].endswith("/mcp")


def test_server_settings_never_return_the_key(server):
    status, body, _ = http(server, "/api/settings", {"api_key": "sk-secret-123456", "model": "m1"})
    data = json.loads(body.decode("utf-8"))
    assert status == 200 and data["settings"]["has_api_key"] and data["settings"]["api_key_tail"] == "3456"
    assert b"sk-secret" not in http(server, "/api/state")[1]
    # cross-site forms cannot post JSON without a preflight; plain text is refused
    assert http(server, "/api/settings", b'{"model": "x"}', ctype="text/plain")[0] == 415


def test_server_chat_turn_and_mcp(server):
    att = _stl_attachment("bushing.stl", meshes.tube(20, 8, 15))
    status, body, _ = http(server, "/api/message", {"text": "Втулка ролика", "attachments": [att]})
    assert status == 202
    seq, kinds, deadline = 0, [], time.time() + 60
    while "turn_done" not in kinds and time.time() < deadline:
        data = json.loads(http(server, "/api/events?since=%d&wait=5" % seq)[1].decode("utf-8"))
        for ev in data["events"]:
            seq = ev["seq"]
            kinds.append(ev["kind"])
    assert "turn_done" in kinds and "assistant" in kinds and "tool_result" in kinds
    status, body, _ = http(server, "/mcp", {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                          "params": {"name": "get_state", "arguments": {}}})
    result = json.loads(body.decode("utf-8"))["result"]
    assert json.loads(result["content"][0]["text"])["printing_mode"] == "cylindrical"
    assert http(server, "/mcp", {"jsonrpc": "2.0", "method": "notifications/initialized"})[0] == 202
    # save buttons
    status, body, _ = http(server, "/api/action", {"tool": "save_project"})
    data = json.loads(body.decode("utf-8"))
    assert status == 200 and data["ok"] and data["result"]["path"].endswith(".3mf")
    assert http(server, "/api/action", {"tool": "delete_object"})[0] == 400


def test_server_save_button_before_slicing(server):
    status, body, _ = http(server, "/api/action", {"tool": "export_gcode"})
    data = json.loads(body.decode("utf-8"))
    assert status == 200 and not data["ok"] and "нарез" in data["error"]
