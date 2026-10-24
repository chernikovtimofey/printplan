"""Command line interface.

    python -m printplan plan MODEL.stl TASK(.docx|.txt|"text") [--no-llm] [--json OUT.json]
    python -m printplan parse TASK [--no-llm]
    python -m printplan kg-report

The LLM is used when STE_LLM_API_KEY is set (see llm_client.py) unless --no-llm.
"""
import argparse
import io
import json
import sys
from collections import OrderedDict

from . import pipeline, text_parse
from .io_utils import read_json_resource, read_task
from .knowledge_graph import KnowledgeGraph
from .llm_client import LLMClient


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def dumps(data):
    return json.dumps(data, ensure_ascii=False, indent=1, default=_json_default)


def _print(text=""):
    sys.stdout.write(text + "\n")


def print_plan(result):
    plan = result["plan"]
    _print("Источник плана: %s" % result["source"])
    _print("Вердикт: %s" % plan["verdict"])
    _print("Режим: %s | профиль: %s | материал: %s | ориентация: %s" % (
        plan["mode"], plan["quality_type"], plan["material"], plan["orientation"]["candidate_id"]))
    _print("  почему так ориентирована: %s" % plan["orientation"].get("reason", ""))
    _print("Настройки:")
    for key, item in result["settings"].items():
        _print("  %-32s = %-10s  [%s] %s" % (key, item["value"], item["source"], item["reason"]))
    if plan.get("risks"):
        _print("Риски:")
        for r in plan["risks"]:
            _print("  [%s] %s (%s)" % (r.get("severity"), r.get("text"), r.get("evidence", "")))
    if plan.get("design_suggestions"):
        _print("Подсказки по конструкции:")
        for d in plan["design_suggestions"]:
            _print("  - %s" % d)
    if plan.get("deviations"):
        _print("Отступления от рекомендаций:")
        for d in plan["deviations"]:
            _print("  - %s: %s" % (d.get("what"), d.get("reason")))
    _print("Объяснение:\n  %s" % plan.get("explanation", ""))
    if plan.get("questions_to_user"):
        _print("Вопросы пользователю:")
        for q in plan["questions_to_user"]:
            _print("  - %s" % q)
    v = result["validation"]
    if v["errors"] or v["warnings"]:
        _print("Проверка: ошибки %s; предупреждения %s" % (v["errors"], v["warnings"]))
    _print("Действия в слайсере:")
    for a in result["actions"]:
        _print("  %d. %s %s" % (a["step"], a["tool"], json.dumps(a["args"], ensure_ascii=False, default=_json_default)[:160]))


def main(argv=None):
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(prog="printplan", description="Print planning pipeline for STE Slicer")
    sub = parser.add_subparsers(dest="command")
    p_plan = sub.add_parser("plan", help="build a print plan for a model and a task")
    p_plan.add_argument("model")
    p_plan.add_argument("task")
    p_plan.add_argument("--no-llm", action="store_true")
    p_plan.add_argument("--json", help="write the full result as JSON")
    p_plan.add_argument("--trace", action="store_true", help="print stage events")
    p_parse = sub.add_parser("parse", help="stage 1 only")
    p_parse.add_argument("task")
    p_parse.add_argument("--no-llm", action="store_true")
    sub.add_parser("kg-report", help="print the knowledge graph for review")
    args = parser.parse_args(argv)

    if args.command == "kg-report":
        _print(KnowledgeGraph(catalog=read_json_resource("slicer_catalog.json")).report())
        return 0
    if args.command == "parse":
        llm = None if args.no_llm else LLMClient.from_env()
        parse, meta = text_parse.parse_task(read_task(args.task), llm)
        _print(dumps(OrderedDict([("parse", parse), ("meta", meta)])))
        return 0
    if args.command == "plan":
        llm = None if args.no_llm else LLMClient.from_env()
        on_event = (lambda e: _print("[%6.2fs] stage %s %s: %s" % (e["t"], e["stage"], e["status"], e["summary"]))) \
            if args.trace else None
        result = pipeline.run(args.task, stl_path=args.model, llm=llm, on_event=on_event)
        print_plan(result)
        if args.json:
            with io.open(args.json, "w", encoding="utf-8") as f:
                f.write(dumps(result))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
