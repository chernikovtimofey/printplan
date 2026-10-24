"""OpenAI-compatible Chat Completions client (stdlib only, Python 3.5).

Configuration comes from the caller or from environment variables:

    STE_LLM_BASE_URL   default https://api.timeweb.ai/v1
    STE_LLM_MODEL      default dashscope/qwen3.8-flash
    STE_LLM_API_KEY    required, never stored in the sources
    STE_LLM_EXTRA_BODY optional JSON merged into every request body
    STE_LLM_STREAM     "0" disables streaming (on by default: no read timeouts on
                       long generations, progress callbacks for the UI)

`call_tool` is the structured-output entry point used by stages 1 and 6: the
model is forced to call one function whose parameters are the JSON schema of
the answer; if no tool call comes back, the same request is repeated with
`response_format: json_schema`.
"""
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from collections import OrderedDict

DEFAULT_BASE_URL = "https://api.timeweb.ai/v1"
DEFAULT_MODEL = "dashscope/qwen3.8-flash"
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)


class LLMError(Exception):
    pass


class LLMClient(object):
    def __init__(self, base_url=None, api_key=None, model=None, timeout=60.0, max_retries=2,
                 extra_body=None, temperature=0.2, stream=None, deadline=300.0):
        self.base_url = (base_url or os.environ.get("STE_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("STE_LLM_API_KEY")
        self.model = model or os.environ.get("STE_LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        if extra_body is None and os.environ.get("STE_LLM_EXTRA_BODY"):
            extra_body = json.loads(os.environ["STE_LLM_EXTRA_BODY"])
        self.extra_body = extra_body or {}
        if stream is None:
            stream = os.environ.get("STE_LLM_STREAM", "1") != "0"
        self.stream = stream
        self.deadline = deadline
        if not self.api_key:
            raise LLMError("No API key: set STE_LLM_API_KEY or pass api_key")

    @classmethod
    def from_env(cls):
        """Client configured from the environment, or None when no key is set."""
        if not os.environ.get("STE_LLM_API_KEY"):
            return None
        return cls()

    # ------------------------------------------------------------------ transport

    def chat(self, messages, tools=None, tool_choice=None, response_format=None, max_tokens=None,
             extra_body=None, on_progress=None):
        body = OrderedDict([("model", self.model), ("messages", messages), ("temperature", self.temperature)])
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        if response_format:
            body["response_format"] = response_format
        if max_tokens:
            body["max_tokens"] = max_tokens
        if self.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        body.update(extra_body or {})
        body.update(self.extra_body)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        url = self.base_url + "/chat/completions"
        last_error = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(2.0 * attempt, 6.0))
            request = urllib.request.Request(url, data=data, method="POST", headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            })
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    if self.stream:
                        payload = _read_stream(resp, started, self.deadline, on_progress)
                    else:
                        payload = json.loads(resp.read().decode("utf-8"))
                payload["_elapsed_s"] = round(time.time() - started, 2)
                return payload
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                last_error = LLMError("HTTP %d from LLM API: %s" % (e.code, detail))
                if e.code not in RETRY_STATUS:
                    raise last_error
            except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
                last_error = LLMError("LLM API unreachable: %s" % (e,))
            except LLMError as e:
                last_error = e
        raise last_error

    # ------------------------------------------------------------------ structured output

    def chat_tools(self, messages, tools, extra_body=None, on_progress=None):
        """One turn of a tool-using conversation (the model decides whether to call tools).

        Returns (assistant message dict, meta). The message has "content" and,
        when the model calls tools, "tool_calls" in the OpenAI format (each with
        an "id" to answer with a role="tool" message).
        """
        response = self.chat(messages, tools=tools or None, tool_choice="auto" if tools else None,
                             extra_body=extra_body, on_progress=on_progress)
        message = dict(_message(response))
        message["role"] = "assistant"
        calls = []
        for i, call in enumerate(message.get("tool_calls") or []):
            call = dict(call)
            call["id"] = call.get("id") or "call_%d" % i
            call["type"] = "function"
            calls.append(call)
        if calls:
            message["tool_calls"] = calls
        else:
            message.pop("tool_calls", None)
        meta = OrderedDict([("model", self.model), ("elapsed_s", response.get("_elapsed_s")),
                            ("usage", response.get("usage"))])
        return message, meta

    def call_tool(self, messages, name, description, parameters, extra_body=None, on_progress=None):
        """Force a call of function `name`; return (arguments dict, meta dict)."""
        tool = {"type": "function", "function": {"name": name, "description": description,
                                                 "parameters": parameters}}
        meta = OrderedDict([("model", self.model), ("method", "tool_call")])
        response = self.chat(messages, tools=[tool],
                             tool_choice={"type": "function", "function": {"name": name}},
                             extra_body=extra_body, on_progress=on_progress)
        meta["elapsed_s"] = response.get("_elapsed_s")
        meta["usage"] = response.get("usage")
        args = extract_tool_arguments(response, name)
        if args is not None:
            return args, meta
        # Fallback: JSON schema response format.
        meta["method"] = "json_schema"
        response = self.chat(messages, response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "schema": parameters},
        }, extra_body=extra_body, on_progress=on_progress)
        meta["elapsed_s"] = (meta["elapsed_s"] or 0) + (response.get("_elapsed_s") or 0)
        content = _message(response).get("content")
        args = parse_json_text(content)
        if args is None:
            raise LLMError("The model returned neither a tool call nor valid JSON")
        return args, meta


def _read_stream(resp, started, deadline, on_progress=None):
    """Assemble a server-sent-events completion into the non-streaming response shape."""
    content, reasoning = [], 0
    calls = OrderedDict()
    usage = None
    finish = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if chunk.get("error"):
            raise LLMError("LLM API error: %s" % (chunk["error"],))
        if chunk.get("usage"):
            usage = chunk["usage"]
        text_delta = ""
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason") or finish
            if delta.get("reasoning_content"):
                reasoning += len(delta["reasoning_content"])
            if delta.get("content"):
                content.append(delta["content"])
                text_delta += delta["content"]
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                call = calls.setdefault(idx, {"type": "function", "function": {"name": None, "arguments": ""}})
                if tc.get("id"):
                    call["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    call["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    call["function"]["arguments"] += fn["arguments"]
        if on_progress is not None:
            try:
                on_progress(OrderedDict([
                    ("elapsed_s", round(time.time() - started, 1)), ("reasoning_chars", reasoning),
                    ("answer_chars", sum(len(c) for c in content) +
                     sum(len(c["function"]["arguments"]) for c in calls.values())),
                    ("text_delta", text_delta)]))
            except Exception:
                pass
        if deadline and time.time() - started > deadline:
            raise LLMError("LLM answer took longer than %.0f s" % deadline)
    message = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = list(calls.values())
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage,
            "_reasoning_chars": reasoning}


def stage_options(model, stage):
    """Provider-specific request options per stage (1, 6 or "agent").

    Qwen models on DashScope-compatible APIs think before answering; the
    extraction of stage 1 and the chat agent's tool choice do not need it, and
    stage 6 gets a bounded budget (measured: 129 s unbounded vs ~40 s with the
    budget). Override with STE_LLM_STAGE1_EXTRA / STE_LLM_STAGE6_EXTRA /
    STE_LLM_AGENT_EXTRA (JSON); "{}" disables.
    """
    env = os.environ.get("STE_LLM_%s_EXTRA" % ("STAGE%s" % stage if stage in (1, 6) else str(stage).upper()))
    if env is not None:
        return json.loads(env)
    if model and "qwen" in model.lower():
        return {"thinking_budget": 1024} if stage == 6 else {"enable_thinking": False}
    return {}


def _message(response):
    try:
        return response["choices"][0]["message"] or {}
    except (KeyError, IndexError, TypeError):
        raise LLMError("Unexpected response shape: %s" % (json.dumps(response)[:300],))


def extract_tool_arguments(response, name=None):
    message = _message(response)
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if name and function.get("name") not in (name, None):
            continue
        args = function.get("arguments")
        if isinstance(args, dict):
            return args
        parsed = parse_json_text(args)
        if parsed is not None:
            return parsed
    # Some models put the JSON into the content instead of a tool call.
    return parse_json_text(message.get("content"))


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_text(text):
    """Parse JSON that may be wrapped in code fences or surrounded by prose."""
    if not text or not isinstance(text, str):
        return None
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE_RE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            value = json.loads(cand, object_pairs_hook=OrderedDict)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


class ScriptedLLM(object):
    """Test double: returns prepared argument dicts (or raises prepared errors) in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model = "scripted"

    def chat_tools(self, messages, tools, extra_body=None, on_progress=None):
        """Prepared item: {"content": str, "tool_calls": [{"name": ..., "arguments": {...}}]}."""
        self.calls.append(OrderedDict([("name", "chat_tools"), ("messages", list(messages)), ("tools", tools)]))
        if not self.responses:
            raise LLMError("ScriptedLLM: no responses left")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        message = {"role": "assistant", "content": item.get("content")}
        calls = []
        for i, call in enumerate(item.get("tool_calls") or []):
            calls.append({"id": "call_%d_%d" % (len(self.calls), i), "type": "function",
                          "function": {"name": call["name"],
                                       "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False)}})
        if calls:
            message["tool_calls"] = calls
        return message, OrderedDict([("model", "scripted"), ("elapsed_s", 0.0)])

    def call_tool(self, messages, name, description, parameters, extra_body=None, on_progress=None):
        self.calls.append(OrderedDict([("name", name), ("messages", messages), ("parameters", parameters)]))
        if not self.responses:
            raise LLMError("ScriptedLLM: no responses left")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        return item, OrderedDict([("model", "scripted"), ("method", "tool_call"), ("elapsed_s", 0.0)])
