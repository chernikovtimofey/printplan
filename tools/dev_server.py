#!/usr/bin/env python
"""Run the AI Assistant chat without STE Slicer, on the in-memory fake slicer.

    python tools/dev_server.py [--port 17531] [--no-llm]

Opens nothing by itself: go to the printed URL. The LLM is used when
STE_LLM_API_KEY is set (or a key is entered on the settings page) unless
--no-llm. G-code and project files written here are marked as fake.
"""
from __future__ import print_function

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plugin"))

from AIAssistant.agent.config import Settings  # noqa: E402
from AIAssistant.agent.server import AssistantServer  # noqa: E402
from AIAssistant.agent.slicer_fake import FakeSlicer  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=17531)
    parser.add_argument("--no-llm", action="store_true", help="ignore STE_LLM_API_KEY")
    parser.add_argument("--slice-delay", type=float, default=1.5, help="seconds the fake slicing takes")
    args = parser.parse_args(argv)
    settings = Settings({"port": args.port}, use_env=not args.no_llm)
    server = AssistantServer(FakeSlicer(slice_delay=args.slice_delay), settings, port=args.port).start()
    print("AI Assistant (fake slicer): %s" % server.url)
    print("MCP endpoint: %smcp" % server.url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
