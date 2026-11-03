#!/usr/bin/env python3
import os
import sys

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.experiments.make_report_charts import main

if __name__ == "__main__":
    raise SystemExit(main())
