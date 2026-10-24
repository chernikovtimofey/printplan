#!/usr/bin/env python
"""Run the pipeline on a set of parts and print an expected-vs-predicted table.

The parts directory (--dir or PRINTPLAN_EXAMPLES_DIR) holds one subfolder per
part: an STL, a task (.docx/.txt/.md) and optionally a photo. A subfolder name
ending in "- 3D" or "- 5D" gives the expected mode. Full results go to
out/cases/<subfolder>/result.json.

    python tools/run_cases.py --dir PARTS_DIR [--no-llm] [--images] [--only SUBSTRING]
"""
from __future__ import print_function

import argparse
import base64
import glob
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "plugin", "AIAssistant"))

from printplan import pipeline  # noqa: E402
from printplan.__main__ import dumps  # noqa: E402
from printplan.llm_client import LLMClient  # noqa: E402

MODE_GROUP = {"classic": "3D", "cylindrical": "5D", "cylindrical_full": "5D"}


def find_cases(base):
    cases = []
    for folder in sorted(glob.glob(os.path.join(base, "*", ""))):
        files = os.listdir(folder)
        stl = [f for f in files if f.lower().endswith(".stl")]
        task = [f for f in files if f.lower().endswith((".docx", ".txt", ".md"))]
        photo = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not stl or not task:
            continue
        name = os.path.basename(os.path.dirname(folder))
        expected = None
        for tag in ("3D", "5D"):
            if name.rstrip().endswith("- " + tag):
                expected = tag
        cases.append({"name": name, "stl": os.path.join(folder, stl[0]), "task": os.path.join(folder, task[0]),
                      "photo": os.path.join(folder, photo[0]) if photo else None, "expected": expected})
    return cases


def data_url(path):
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return "data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode("ascii"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=os.environ.get("PRINTPLAN_EXAMPLES_DIR"),
                        help="directory with one subfolder per part (default: $PRINTPLAN_EXAMPLES_DIR)")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--images", action="store_true", help="attach the part photo to the stage-6 request")
    parser.add_argument("--only", help="run only folders containing this text")
    parser.add_argument("--out", default=os.path.join(ROOT, "out", "cases"))
    args = parser.parse_args(argv)
    if not args.dir or not os.path.isdir(args.dir):
        parser.error("pass --dir or set PRINTPLAN_EXAMPLES_DIR to a directory of parts")

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    llm = None if args.no_llm else LLMClient.from_env()
    if llm is None and not args.no_llm:
        print("STE_LLM_API_KEY is not set: running without the LLM")
    cases = [c for c in find_cases(args.dir) if not args.only or args.only in c["name"]]
    rows = []
    for case in cases:
        started = time.time()
        images = [data_url(case["photo"])] if args.images and case["photo"] else None
        result = pipeline.run(case["task"], stl_path=case["stl"], llm=llm, images=images)
        elapsed = time.time() - started
        plan = result["plan"]
        mc = result["evidence"]["E_G"]["mode_candidate"]
        group = MODE_GROUP[plan["mode"]]
        ok = None if case["expected"] is None else group == case["expected"]
        rows.append([case["name"][:44], case["expected"] or "?", plan["mode"], "%+.1f/%s" % (mc["score"], mc["confidence"]),
                     plan["orientation"]["candidate_id"], plan["quality_type"], plan["material"], plan["verdict"],
                     result["source"], str(max(len(result["attempts"]) - 1, 0)), "%.0fs" % elapsed,
                     "" if ok is None else ("ok" if ok else "MISS")])
        out_dir = os.path.join(args.out, case["name"])
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        with io.open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            f.write(dumps(result))
        print("done: %s (%.0f s)" % (case["name"], elapsed))

    header = ["part", "exp", "mode", "score/conf", "orient", "quality", "material", "verdict", "source", "retries",
              "time", "check"]
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    print()
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    checked = [r for r in rows if r[-1]]
    if checked:
        print("\nmode (3D/5D) matches folder label: %d / %d" % (sum(r[-1] == "ok" for r in checked), len(checked)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
