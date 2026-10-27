#!/usr/bin/env python3
"""Root CLI and module wrapper for Observer Overhead Measurement (E6)."""

import os
import sys

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.experiments.measure_overhead import evaluate_observer_overhead, main

if __name__ == "__main__":
    main()
