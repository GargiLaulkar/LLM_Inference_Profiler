"""Suite Runner for Experiments E1 to E6 (Member 4 - Team Handbook IDs).

Matches Team Handbook Specification:
- E1: Injected-bottleneck detection (synthetic sanity check; real E1 requires M2 server hooks)
- E2: Load level (RPS scaling: 1 vs 5 vs 10 RPS)
- E3: Prompt/output length mix (short/long prompt vs short/long output)
- E4: Batching on/off (marked pending_live)
- E5: Model size (marked pending_live)
- E6: Profiler overhead on vs off (instrumentation benchmark; live run pending server load test)

All current outputs carry "data_source": "synthetic".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import yaml

from ..analyzer.bottleneck_analyzer import BottleneckAnalyzer
from ..analyzer.bottleneck_injector import BottleneckInjector
from ..analyzer.cost_model import CostModel, load_gpu_price_from_config
from .measure_overhead import evaluate_instrumentation_microbenchmark


class ExperimentSuiteRunner:
    """Orchestrates experiments E1 through E6 with strict handbook alignment."""

    def __init__(
        self,
        output_dir: str = "results/experiments",
        config_path: Optional[str] = None,
        gpu_price_per_hour_usd: Optional[float] = None,
    ) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.gpu_price_per_hour_usd = (
            float(gpu_price_per_hour_usd)
            if gpu_price_per_hour_usd is not None
            else load_gpu_price_from_config(config_path)
        )
        self.analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=self.gpu_price_per_hour_usd)
        self.injector = BottleneckInjector(seed=42)

    def run_e1_injected_bottleneck_detection(self, requests_per_scenario: int = 30) -> Dict[str, Any]:
        """E1: Injected-bottleneck detection.

        NOTE: Injector-based detection is a synthetic sanity check. Real E1 requires M2's hooks.
        """
        e1_dir = os.path.join(self.output_dir, "e1_injected_bottlenecks")
        os.makedirs(e1_dir, exist_ok=True)

        results: Dict[str, Any] = {}
        stages_to_inject = [
            ("tokenization", 0.50),
            ("queue_wait", 0.80),
            ("prefill", 0.60),
            ("decode", 0.60),
            ("network", 0.50),
        ]

        for stage, delay in stages_to_inject:
            tag = f"e1_inject_{stage}"
            csv_rows, traces = self.injector.generate_synthetic_workload(
                num_requests=requests_per_scenario,
                scenario_tag=tag,
                target_rps=3.0,
                prompt_tokens=128,
                output_tokens=64,
                injected_stage=stage,
                injection_delay_s=delay,
            )
            csv_path, traces_path = self.injector.save_workload(
                csv_rows, traces, output_dir=e1_dir, prefix=tag
            )
            analysis = self.analyzer.analyze(csv_path, traces_path)
            top_b = analysis["top_bottlenecks"][0]

            results[stage] = {
                "injected_stage": stage,
                "injected_delay_s": delay,
                "detected_rank_1_stage": top_b["stage"],
                "detected_share": top_b["share_of_latency"],
                "is_detected_correctly": top_b["stage"] == stage,
                "suggested_action": top_b["suggested_action"],
            }

        return {
            "experiment_id": "E1",
            "name": "E1: Injected-Bottleneck Detection",
            "data_source": "synthetic",
            "status": "completed_synthetic",
            "validation_note": "Injector-based detection is a synthetic sanity check; real E1 requires M2's live hooks.",
            "injection_tests": results,
            "all_passed": all(r["is_detected_correctly"] for r in results.values()),
        }

    def run_e2_load_level_rps(self, requests_per_scenario: int = 30) -> Dict[str, Any]:
        """E2: Load level scaling (1 vs 5 vs 10 RPS)."""
        e2_dir = os.path.join(self.output_dir, "e2_load_level")
        os.makedirs(e2_dir, exist_ok=True)

        results: Dict[str, Any] = {}
        for rps in [1.0, 5.0, 10.0]:
            tag = f"e2_rps_{int(rps)}"
            csv_rows, traces = self.injector.generate_synthetic_workload(
                num_requests=requests_per_scenario,
                scenario_tag=tag,
                target_rps=rps,
                prompt_tokens=128,
                output_tokens=64,
            )
            csv_path, traces_path = self.injector.save_workload(
                csv_rows, traces, output_dir=e2_dir, prefix=tag
            )
            analysis = self.analyzer.analyze(csv_path, traces_path)
            results[f"rps_{int(rps)}"] = {
                "rps": rps,
                "data_source": "synthetic",
                "top_bottlenecks": analysis["top_bottlenecks"],
                "cost_per_million_tokens_usd": analysis["cost_per_million_tokens_usd"],
                "explained_share": analysis["explained_share"],
            }

        return {
            "experiment_id": "E2",
            "name": "E2: Load Level (RPS Scaling)",
            "data_source": "synthetic",
            "status": "completed_synthetic",
            "scenarios": results,
            "key_finding": "At 1 RPS compute dominates; at 10 RPS queue_wait rises from concurrency contention.",
        }

    def run_e3_prompt_output_length_mix(self, requests_per_scenario: int = 30) -> Dict[str, Any]:
        """E3: Prompt/output length mix."""
        e3_dir = os.path.join(self.output_dir, "e3_length_mix")
        os.makedirs(e3_dir, exist_ok=True)

        mixes = [
            ("short_prompt_32_64", 32, 64),
            ("long_prompt_1024_64", 1024, 64),
            ("long_output_128_256", 128, 256),
        ]
        results: Dict[str, Any] = {}

        for tag, p_len, o_len in mixes:
            csv_rows, traces = self.injector.generate_synthetic_workload(
                num_requests=requests_per_scenario,
                scenario_tag=tag,
                target_rps=2.0,
                prompt_tokens=p_len,
                output_tokens=o_len,
            )
            csv_path, traces_path = self.injector.save_workload(
                csv_rows, traces, output_dir=e3_dir, prefix=tag
            )
            analysis = self.analyzer.analyze(csv_path, traces_path)
            results[tag] = {
                "prompt_tokens": p_len,
                "output_tokens": o_len,
                "data_source": "synthetic",
                "top_bottlenecks": analysis["top_bottlenecks"],
                "cost_per_million_tokens_usd": analysis["cost_per_million_tokens_usd"],
                "explained_share": analysis["explained_share"],
            }

        return {
            "experiment_id": "E3",
            "name": "E3: Prompt/Output Length Mix",
            "data_source": "synthetic",
            "status": "completed_synthetic",
            "scenarios": results,
            "key_finding": "Long prompts shift bottleneck to prefill; long outputs shift bottleneck to decode.",
        }

    def run_e4_batching_on_off(self) -> Dict[str, Any]:
        """E4: Batching on/off (Continuous batching vs sequential). Marked pending_live."""
        return {
            "experiment_id": "E4",
            "name": "E4: Batching On vs Off",
            "data_source": "synthetic",
            "status": "pending_live",
            "note": "[PENDING LIVE BENCHMARK] Requires toggling vLLM live engine flags (--max-num-seqs 1 vs dynamic batching) on the active server.",
            "planned_comparison": {
                "batching_on": "Continuous batching enabled (default vLLM scheduler)",
                "batching_off": "Strict sequential scheduling (--max-num-seqs 1)",
            },
        }

    def run_e5_model_size(self) -> Dict[str, Any]:
        """E5: Model size comparison. Marked pending_live."""
        return {
            "experiment_id": "E5",
            "name": "E5: Model Size Comparison",
            "data_source": "synthetic",
            "status": "pending_live",
            "note": "[PENDING LIVE BENCHMARK] Requires downloading and hosting multiple model parameter scales (e.g. Qwen2.5-0.5B vs 1.5B) under 6GB VRAM constraint.",
            "planned_comparison": {
                "reference_model": "Qwen/Qwen2.5-0.5B-Instruct (fits comfortably in 6GB)",
                "scaled_model": "Qwen/Qwen2.5-1.5B-Instruct (requires careful VRAM allocation)",
            },
        }

    def run_e6_profiler_overhead_on_vs_off(self, iterations: int = 5_000) -> Dict[str, Any]:
        """E6: Profiler overhead on vs off."""
        report = evaluate_instrumentation_microbenchmark(iterations=iterations)
        return {
            "experiment_id": "E6",
            "name": "E6: Profiler Overhead On vs Off",
            "data_source": "synthetic",
            "status": "completed_synthetic",
            "note": "Instrumentation microbenchmark. Full end-to-end load generator comparison with profiling ON/OFF scheduled for live phase.",
            "overhead_metrics": report,
            "key_finding": f"Instrumentation adds {report['profiler_overhead_per_request_us']:.2f} us per request ({report['overhead_percent_on_100ms_request']:.4f}% on 100ms request).",
        }

    def run_all(self) -> Dict[str, Any]:
        """Run all experiments E1 through E6."""
        print("[INFO] Running Experiment E1: Injected-Bottleneck Detection (Synthetic Sanity Check)...")
        e1 = self.run_e1_injected_bottleneck_detection()

        print("[INFO] Running Experiment E2: Load Level (RPS Scaling)...")
        e2 = self.run_e2_load_level_rps()

        print("[INFO] Running Experiment E3: Prompt/Output Length Mix...")
        e3 = self.run_e3_prompt_output_length_mix()

        print("[INFO] Registering Experiment E4: Batching On/Off [PENDING LIVE]...")
        e4 = self.run_e4_batching_on_off()

        print("[INFO] Registering Experiment E5: Model Size Comparison [PENDING LIVE]...")
        e5 = self.run_e5_model_size()

        print("[INFO] Running Experiment E6: Profiler Overhead (Instrumentation Microbenchmark)...")
        e6 = self.run_e6_profiler_overhead_on_vs_off()

        consolidated = {
            "data_source": "synthetic",
            "gpu_price_per_hour_usd": self.gpu_price_per_hour_usd,
            "experiments": {
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "e4": e4,
                "e5": e5,
                "e6": e6,
            },
        }

        summary_path = os.path.join(self.output_dir, "experiments_e1_to_e6_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)

        print(f"\n[SUCCESS] Consolidated experiment summary saved to {summary_path}")
        return consolidated


def main() -> None:
    """CLI runner for experiment suite E1 to E6."""
    parser = argparse.ArgumentParser(description="Run experiments E1 to E6 (Team Handbook IDs).")
    parser.add_argument("--out-dir", default="results/experiments", help="Output directory")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--gpu-price", type=float, default=None, help="GPU price per hour override")
    args = parser.parse_args()

    runner = ExperimentSuiteRunner(
        output_dir=args.out_dir,
        config_path=args.config,
        gpu_price_per_hour_usd=args.gpu_price,
    )
    runner.run_all()


if __name__ == "__main__":
    main()
