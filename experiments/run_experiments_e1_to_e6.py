#!/usr/bin/env python3
"""Root CLI and module wrapper for Experiments E1 to E6."""

import os
import sys

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.experiments.run_experiments_e1_to_e6 import ExperimentSuiteRunner, main

if __name__ == "__main__":
    main()
