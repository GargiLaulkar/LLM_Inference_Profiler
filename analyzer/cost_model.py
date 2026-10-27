#!/usr/bin/env python3
"""Root CLI and module wrapper for Inference Cost Model."""

import os
import sys

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.analyzer.cost_model import (
    CostModel,
    StandaloneCostExporter,
    calculate_csv_cost,
    calculate_request_cost,
    load_gpu_price_from_config,
    main,
)

if __name__ == "__main__":
    main()
