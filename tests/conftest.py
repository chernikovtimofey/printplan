import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "plugin", "AIAssistant"))
# The plugin package itself (AIAssistant.agent uses relative imports into AIAssistant.printplan).
sys.path.insert(0, os.path.join(ROOT, "plugin"))
sys.path.insert(0, HERE)

import pytest  # noqa: E402

# Optional directory of labelled parts for tests/test_examples.py (see tools/run_cases.py).
EXAMPLES_DIR = os.environ.get("PRINTPLAN_EXAMPLES_DIR") or ""


@pytest.fixture(scope="session")
def catalog():
    from printplan.io_utils import read_json_resource
    return read_json_resource("slicer_catalog.json")


@pytest.fixture(scope="session")
def machine(catalog):
    return catalog["machine"]


@pytest.fixture(scope="session")
def ctx(catalog):
    from printplan.slicer_context import CatalogSlicerContext
    return CatalogSlicerContext(catalog)


@pytest.fixture(scope="session")
def kg(catalog):
    from printplan.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph(catalog=catalog)
