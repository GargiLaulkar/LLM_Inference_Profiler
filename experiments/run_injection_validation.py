"""Thin root wrapper for the M4 injection-validation runner."""

import os
import sys

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.experiments.run_injection_validation import main

if __name__ == "__main__":
    raise SystemExit(main())
