"""Root wrapper for the LLM Profiler analyzer CLI."""

import os
import sys

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.analyzer.cli import main

if __name__ == "__main__":
    main()
