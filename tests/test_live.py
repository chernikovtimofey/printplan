"""Opt-in tests against the real LLM API:

    STE_LLM_API_KEY=... PRINTPLAN_LIVE=1 pytest -m live

With PRINTPLAN_EXAMPLES_DIR set, every labelled part in it is also planned live.
"""
import os
import sys

import pytest

import meshes
from conftest import EXAMPLES_DIR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

LIVE = os.environ.get("PRINTPLAN_LIVE") == "1" and os.environ.get("STE_LLM_API_KEY")
pytestmark = [pytest.mark.live, pytest.mark.skipif(not LIVE, reason="set PRINTPLAN_LIVE=1 and STE_LLM_API_KEY")]


def _client():
    from printplan.llm_client import LLMClient
    return LLMClient()


def test_stage1_live():
    from printplan import text_parse
    parse, meta = text_parse.parse_task("Кронштейн под нагрузку, PETG, без поддержек", _client())
    assert meta["llm_error"] is None
    assert parse["constraints"]["material"] == "enduse_petg"
    assert parse["constraints"]["supports"] == "forbidden"
    assert "load" in parse["flags"]


def test_case_example_live():
    from printplan import pipeline
    result = pipeline.run("Кронштейн под нагрузку, PETG, без поддержек", triangles=meshes.l_bracket(), llm=_client())
    plan = result["plan"]
    assert result["validation"]["errors"] == []
    assert plan["material"] == "enduse_petg"
    values = dict((k, v["value"]) for k, v in result["settings"].items())
    assert values.get("support_enable") is not True


def _labelled_cases():
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    import run_cases
    return [c for c in run_cases.find_cases(EXAMPLES_DIR) if c["expected"]]


LABELLED_CASES = _labelled_cases()


@pytest.mark.skipif(not os.path.isdir(EXAMPLES_DIR), reason="PRINTPLAN_EXAMPLES_DIR not set")
@pytest.mark.parametrize("case", LABELLED_CASES, ids=[c["name"][:30] for c in LABELLED_CASES])
def test_example_parts_live(case):
    from printplan import pipeline
    result = pipeline.run(case["task"], stl_path=case["stl"], llm=_client())
    assert result["validation"]["errors"] == []
    group = "3D" if result["plan"]["mode"] == "classic" else "5D"
    assert group == case["expected"]
