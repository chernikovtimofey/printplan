"""Local HTTP server: chat page, chat API, event stream and MCP endpoint.

    GET  /                  chat page (web/index.html and its assets)
    GET  /api/state         settings (without the key), tools, slicer state
    GET  /api/events        events after ?since=N, long-polling up to ?wait=S seconds
    POST /api/message       {"text": ..., "attachments": [{"name", "data" (base64)}]}
    POST /api/action        {"tool": "export_gcode" | "save_project"}: the save buttons, without the LLM
    POST /api/settings      {"base_url", "model", "api_key", "port", "output_dir"}
    POST /api/reset         start a new conversation
    POST /mcp               MCP JSON-RPC (initialize, tools/list, tools/call)

Binds to 127.0.0.1 only. POST requests must be application/json, so a web page
in the same browser cannot send them without a CORS preflight this server
never approves.
"""
import json
import mimetypes
import os
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from .agent import BusyError, Session
from .events import EventLog
from .mcp import MCPServer
from .tools import ToolContext, build_registry, jsonable

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
MAX_BODY = 400 * 1024 * 1024


class _HTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class AssistantServer(object):
    def __init__(self, slicer, settings, host="127.0.0.1", port=None, web_dir=WEB_DIR, log=None):
        self.slicer = slicer
        self.settings = settings
        self.events = EventLog()
        self.registry = build_registry()
        self.context = ToolContext(slicer, settings, self.events, self.registry)
        self.session = Session(self.registry, self.context, settings, self.events)
        self.mcp = MCPServer(self.registry, self.context)
        self.web_dir = web_dir
        self.log = log
        port = settings.port() if port is None else port
        try:
            self._httpd = _HTTPServer((host, port), _make_handler(self))
        except OSError:
            self._httpd = _HTTPServer((host, 0), _make_handler(self))  # preferred port busy
        self._thread = None

    @property
    def port(self):
        return self._httpd.server_address[1]

    @property
    def url(self):
        return "http://127.0.0.1:%d/" % self.port

    def start(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="ai-assistant-http")
        self._thread.daemon = True
        self._thread.start()
        return self

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()

    def state(self):
        out = OrderedDict()
        out["settings"] = self.settings.public()
        out["busy"] = self.session.busy
        out["last_seq"] = self.events.last_seq()
        out["slicer"] = self.slicer.name
        out["mcp_url"] = self.url + "mcp"
        out["tools"] = [OrderedDict([("name", t.name), ("title", t.title)]) for t in self.registry.public_tools()]
        return out


def _make_handler(server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "STE-AI-Assistant/0.1"

        def log_message(self, fmt, *args):
            if server.log is not None:
                server.log(fmt % args)

        # -------------------------------------------------------------- responses
        def _send(self, code, body, content_type="application/json; charset=utf-8"):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code, data):
            self._send(code, json.dumps(jsonable(data), ensure_ascii=False))

        def _read_json(self):
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                raise ValueError("Content-Type must be application/json")
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("Request body too large")
            raw = self.rfile.read(length) if length else b""
            return raw.decode("utf-8")

        # -------------------------------------------------------------- GET
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/state":
                return self._json(200, server.state())
            if url.path == "/api/events":
                query = parse_qs(url.query)
                since = int((query.get("since") or ["0"])[0])
                wait = min(float((query.get("wait") or ["0"])[0]), 30.0)
                events = server.events.since(since, timeout=wait)
                return self._json(200, {"events": events, "last_seq": server.events.last_seq()})
            if url.path == "/mcp":
                return self._json(405, {"error": "Use POST with JSON-RPC 2.0"})
            return self._static(url.path)

        do_HEAD = do_GET

        def _static(self, path):
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            full = os.path.normpath(os.path.join(server.web_dir, rel))
            if not full.startswith(os.path.normpath(server.web_dir) + os.sep) or not os.path.isfile(full):
                return self._send(404, "Not found", "text/plain; charset=utf-8")
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript",):
                ctype += "; charset=utf-8"
            with open(full, "rb") as f:
                self._send(200, f.read(), ctype)

        # -------------------------------------------------------------- POST
        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._read_json()
            except ValueError as e:
                return self._json(415 if "Content-Type" in str(e) else 413, {"error": str(e)})
            if path == "/mcp":
                text = server.mcp.handle_text(body)
                if not text:
                    return self._send(202, b"", "application/json")
                return self._send(200, text)
            try:
                data = json.loads(body) if body else {}
            except ValueError:
                return self._json(400, {"error": "Invalid JSON"})
            if path == "/api/message":
                try:
                    server.session.start_turn(data.get("text") or "", data.get("attachments") or [])
                except BusyError as e:
                    return self._json(409, {"error": str(e)})
                return self._json(202, {"ok": True})
            if path == "/api/action":
                try:
                    return self._json(200, server.session.run_action(data.get("tool") or ""))
                except BusyError as e:
                    return self._json(409, {"error": str(e)})
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
            if path == "/api/settings":
                try:
                    server.settings.update(data)
                except Exception as e:
                    return self._json(400, {"error": str(e)})
                server.events.emit("settings", settings=server.settings.public())
                return self._json(200, {"ok": True, "settings": server.settings.public()})
            if path == "/api/reset":
                try:
                    server.session.reset()
                except BusyError as e:
                    return self._json(409, {"error": str(e)})
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "Not found"})

    return Handler
