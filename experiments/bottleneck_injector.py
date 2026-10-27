#!/usr/bin/env python3
"""Root CLI and module wrapper for Bottleneck Injector in experiments."""

import os
import sys

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.analyzer.bottleneck_injector import BottleneckInjector, main

if __name__ == "__main__":
    main()
