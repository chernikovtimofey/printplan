"""Regression on a directory of labelled parts (PRINTPLAN_EXAMPLES_DIR).

Every subfolder holds an STL and a task; a name ending in "- 3D" or "- 5D"
gives the expected mode, unlabelled parts are expected to stay 3D. The test
pins the deterministic stages (no LLM). Skipped when the variable is not set.
"""
import os
import sys

import pytest

from conftest import EXAMPLES_DIR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

pytestmark = [pytest.mark.examples,
              pytest.mark.skipif(not os.path.isdir(EXAMPLES_DIR), reason="PRINTPLAN_EXAMPLES_DIR not set")]


def _cases():
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    import run_cases
    return run_cases.find_cases(EXAMPLES_DIR)


CASES = _cases()


@pytest.mark.parametrize("case", CASES, ids=[c["name"][:30] for c in CASES])
def test_example_mode_and_plan(case):
    from printplan import pipeline
    result = pipeline.run(case["task"], stl_path=case["stl"])
    group = "3D" if result["plan"]["mode"] == "classic" else "5D"
    assert group == (case["expected"] or "3D")
    assert result["validation"]["errors"] == []
