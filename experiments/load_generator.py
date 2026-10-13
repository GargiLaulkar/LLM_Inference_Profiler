#!/usr/bin/env python3
"""Open-loop Async Load Generator for LLM Inference Profiler."""

import sys
import os

# Delegate directly to llm_profiler implementation
pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.experiments.load_generator import main

if __name__ == "__main__":
    main()
