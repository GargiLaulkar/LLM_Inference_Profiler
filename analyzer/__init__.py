"""Root analyzer package wrapper for LLM Inference Profiler."""

import os
import sys

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.analyzer.cost_model import (
    CostModel,
    calculate_request_cost,
    calculate_csv_cost,
    load_gpu_price_from_config,
)

__all__ = [
    "CostModel",
    "calculate_request_cost",
    "calculate_csv_cost",
    "load_gpu_price_from_config",
]
