"""MCP server over the tool registry (JSON-RPC 2.0, the subset MCP clients need).

Implemented by hand because the official MCP SDK requires Python 3.10+:
`initialize`, `notifications/initialized`, `ping`, `tools/list`, `tools/call`.
The HTTP server exposes it at POST /mcp; the chat agent calls the same
registry directly.
"""
import json
from collections import OrderedDict

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = OrderedDict([("name", "ste-slicer-ai-assistant"), ("version", "0.1.0")])

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


def _error(msg_id, code, message):
    return OrderedDict([("jsonrpc", "2.0"), ("id", msg_id),
                        ("error", OrderedDict([("code", code), ("message", message)]))])


def _result(msg_id, result):
    return OrderedDict([("jsonrpc", "2.0"), ("id", msg_id), ("result", result)])


class MCPServer(object):
    def __init__(self, registry, context):
        self.registry = registry
        self.context = context

    def handle_text(self, text):
        """Handle a request body; returns the response text ('' for notifications only)."""
        try:
            message = json.loads(text)
        except ValueError:
            return json.dumps(_error(None, PARSE_ERROR, "Parse error"))
        if isinstance(message, list):
            responses = [r for r in (self.handle(m) for m in message) if r is not None]
            return json.dumps(responses, ensure_ascii=False) if responses else ""
        response = self.handle(message)
        return json.dumps(response, ensure_ascii=False) if response is not None else ""

    def handle(self, message):
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
            return _error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST,
                          "Invalid request")
        msg_id = message.get("id")
        method = message["method"]
        params = message.get("params") or {}
        is_notification = "id" not in message

        if method == "initialize":
            result = OrderedDict([
                ("protocolVersion", params.get("protocolVersion") or PROTOCOL_VERSION),
                ("capabilities", OrderedDict([("tools", OrderedDict([("listChanged", False)]))])),
                ("serverInfo", SERVER_INFO),
                ("instructions", "Инструменты слайсера STE Slicer (Stereotech Hybrid 530): настройки, профили, "
                                 "материалы, режим 3D/5D, модели, слайсинг. Главный инструмент — plan_print."),
            ])
            return _result(msg_id, result)
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return None if is_notification else _result(msg_id, OrderedDict())
        if method == "tools/list":
            return _result(msg_id, OrderedDict([("tools", self.registry.mcp_tools())]))
        if method == "tools/call":
            name = params.get("name")
            tool = self.registry.get(name)
            if tool is None or not tool.public:
                return _error(msg_id, INVALID_PARAMS, "Unknown tool: %s" % name)
            outcome = self.registry.call(name, params.get("arguments") or {}, self.context, source="mcp")
            payload = outcome.get("result") if outcome["ok"] else OrderedDict([("error", outcome["error"])])
            content = [OrderedDict([("type", "text"), ("text", json.dumps(payload, ensure_ascii=False))])]
            return _result(msg_id, OrderedDict([("content", content), ("isError", not outcome["ok"])]))
        if is_notification:
            return None
        return _error(msg_id, METHOD_NOT_FOUND, "Method not found: %s" % method)
