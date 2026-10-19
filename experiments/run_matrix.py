#!/usr/bin/env python3
"""Scenario Matrix Runner for LLM Inference Profiler."""

import sys
import os

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.experiments.run_matrix import main

if __name__ == "__main__":
    main()
