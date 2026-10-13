#!/usr/bin/env python3
"""Open-loop Async Load Generator for LLM Inference Profiler.

Contract:
Columns: request_id,send_ts,end_ts,total_latency_s,prompt_tokens,output_tokens,status,target_rps,scenario_tag
"""

import argparse
import asyncio
import csv
import os
import sys
import time
import uuid
from typing import List, Dict, Any, Optional
import yaml
import aiohttp

# Shared Contract CSV Headers
CSV_HEADERS = [
    "request_id",
    "send_ts",
    "end_ts",
    "total_latency_s",
    "prompt_tokens",
    "output_tokens",
    "status",
    "target_rps",
    "scenario_tag",
]

# Neutral vocabulary for deterministic prompt generation
NEUTRAL_WORDS = [
    "network", "analysis", "system", "performance", "metric", "profiler",
    "latency", "throughput", "evaluation", "token", "queue", "inference",
    "server", "benchmark", "prefill", "decode", "memory", "resource",
    "execution", "hardware", "process", "pipeline", "request", "response"
]


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load config.yaml with path resolution fallback."""
    paths_to_check = [
        config_path,
        os.path.join("llm_profiler", "config.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
        "config.yaml",
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


class PromptCalibrator:
    """Generates calibrated prompts of desired token lengths."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            self.tokenizer = None

    def build_prompt(self, target_tokens: int) -> str:
        """Constructs a prompt approximating target_tokens."""
        if target_tokens <= 0:
            return "Test"

        if self.tokenizer is not None:
            # Calibrate using real tokenizer
            # Start with an estimate of words
            word_count = max(1, int(target_tokens * 0.8))
            words = []
            while len(words) < word_count:
                words.extend(NEUTRAL_WORDS)
            prompt = " ".join(words[:word_count])

            # Adjust iteratively to get close to target_tokens
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            curr_len = len(tokens)

            if curr_len > target_tokens:
                tokens = tokens[:target_tokens]
                return self.tokenizer.decode(tokens)
            elif curr_len < target_tokens:
                needed = target_tokens - curr_len
                extra_words = (NEUTRAL_WORDS * ((needed // len(NEUTRAL_WORDS)) + 2))[:needed]
                prompt = prompt + " " + " ".join(extra_words)
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)[:target_tokens]
                return self.tokenizer.decode(tokens)
            return prompt
        else:
            # Fallback heuristic: 1 word ~ 1.3 tokens
            word_count = max(1, int(target_tokens / 1.3))
            words = []
            while len(words) < word_count:
                words.extend(NEUTRAL_WORDS)
            return " ".join(words[:word_count])

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return max(1, int(len(text.split()) * 1.3))


def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Computes mean, p50, p95, p99 from a list of latencies."""
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

    sorted_lats = sorted(latencies)
    n = len(sorted_lats)
    mean_val = sum(sorted_lats) / n

    def get_p(p: float) -> float:
        # Nearest rank method
        idx = int((p / 100.0) * n)
        idx = min(idx, n - 1)
        return sorted_lats[idx]

    return {
        "mean": mean_val,
        "p50": get_p(50),
        "p95": get_p(95),
        "p99": get_p(99),
    }


async def send_single_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    request_id: str,
    target_rps: float,
    scenario_tag: str,
    estimated_prompt_tokens: int,
) -> Dict[str, Any]:
    """Dispatches a single completion request with timing and error isolation."""
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
    }

    send_ts = time.time()
    t_start = time.perf_counter()
    status = "ok"
    prompt_tokens = estimated_prompt_tokens
    output_tokens = payload.get("max_tokens", 0)

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            t_end = time.perf_counter()
            end_ts = time.time()
            total_latency = t_end - t_start

            if response.status == 200:
                data = await response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)
                status = "ok"
            else:
                resp_text = await response.text()
                status = f"error:http_{response.status}"
    except asyncio.TimeoutError:
        t_end = time.perf_counter()
        end_ts = time.time()
        total_latency = t_end - t_start
        status = "error:timeout"
    except aiohttp.ClientError as ce:
        t_end = time.perf_counter()
        end_ts = time.time()
        total_latency = t_end - t_start
        status = f"error:{type(ce).__name__}"
    except Exception as e:
        t_end = time.perf_counter()
        end_ts = time.time()
        total_latency = t_end - t_start
        status = f"error:{type(e).__name__}"

    return {
        "request_id": request_id,
        "send_ts": send_ts,
        "end_ts": end_ts,
        "total_latency_s": total_latency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "status": status,
        "target_rps": target_rps,
        "scenario_tag": scenario_tag,
    }


async def run_load_generator(
    server_url: str,
    model_name: str,
    rps: float,
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
    scenario_tag: str,
    output_csv: str,
) -> Dict[str, Any]:
    """Executes open-loop load generation schedule."""
    calibrator = PromptCalibrator(model_name)
    calibrated_prompt = calibrator.build_prompt(prompt_tokens)
    est_tokens = calibrator.count_tokens(calibrated_prompt)

    payload = {
        "model": model_name,
        "prompt": calibrated_prompt,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": False,
    }

    completion_endpoint = server_url.rstrip("/")
    if not completion_endpoint.endswith("/completions"):
        if completion_endpoint.endswith("/v1"):
            completion_endpoint = f"{completion_endpoint}/completions"
        else:
            completion_endpoint = f"{completion_endpoint}/v1/completions"

    # Concurrency limit 512, timeout 120s
    connector = aiohttp.TCPConnector(limit=512)
    timeout = aiohttp.ClientTimeout(total=120)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    results: List[Dict[str, Any]] = []
    wall_start = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        interval = 1.0 / rps if rps > 0 else 0.0

        for i in range(num_requests):
            req_id = str(uuid.uuid4())
            target_dispatch = wall_start + (i * interval)
            delay = target_dispatch - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

            # Fire request without blocking schedule (open-loop)
            task = asyncio.create_task(
                send_single_request(
                    session=session,
                    url=completion_endpoint,
                    payload=payload,
                    request_id=req_id,
                    target_rps=rps,
                    scenario_tag=scenario_tag,
                    estimated_prompt_tokens=est_tokens,
                )
            )
            tasks.append(task)

        # Wait for all in-flight requests to complete
        results = await asyncio.gather(*tasks)

    wall_duration = time.perf_counter() - wall_start

    # Write CSV with exact contract headers and order
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Metrics summary calculation
    successful = [r for r in results if r["status"] == "ok"]
    success_count = len(successful)
    total_count = len(results)
    success_rate = (success_count / total_count * 100.0) if total_count > 0 else 0.0

    successful_lats = [r["total_latency_s"] for r in successful]
    pcts = calculate_percentiles(successful_lats)

    achieved_throughput = (success_count / wall_duration) if wall_duration > 0 else 0.0
    total_out_tokens = sum(r["output_tokens"] for r in successful)
    tokens_per_sec = (total_out_tokens / wall_duration) if wall_duration > 0 else 0.0

    summary = {
        "count": total_count,
        "success_count": success_count,
        "success_rate_pct": success_rate,
        "mean_latency_s": pcts["mean"],
        "p50_latency_s": pcts["p50"],
        "p95_latency_s": pcts["p95"],
        "p99_latency_s": pcts["p99"],
        "achieved_throughput_rps": achieved_throughput,
        "total_output_tokens_per_sec": tokens_per_sec,
        "wall_duration_s": wall_duration,
        "output_csv": output_csv,
    }

    # Print summary report
    print("\n" + "=" * 60)
    print(f"LOAD GENERATION SUMMARY: {scenario_tag}")
    print("=" * 60)
    print(f"Total Requests:       {summary['count']}")
    print(f"Success Rate:         {summary['success_rate_pct']:.2f}% ({summary['success_count']}/{summary['count']})")
    print(f"Mean Latency:         {summary['mean_latency_s']:.4f} s")
    print(f"P50 Latency:          {summary['p50_latency_s']:.4f} s")
    print(f"P95 Latency:          {summary['p95_latency_s']:.4f} s")
    print(f"P99 Latency:          {summary['p99_latency_s']:.4f} s")
    print(f"Achieved Throughput:  {summary['achieved_throughput_rps']:.2f} req/s")
    print(f"Output Token Rate:    {summary['total_output_tokens_per_sec']:.2f} tok/s")
    print(f"Wall Time:            {summary['wall_duration_s']:.2f} s")
    print(f"Results CSV:          {output_csv}")
    print("=" * 60 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Open-loop LLM Serving Stack Load Generator")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--rps", type=float, default=None, help="Target requests per second")
    parser.add_argument("--num-requests", type=int, default=None, help="Total requests to generate")
    parser.add_argument("--prompt-tokens", type=int, default=None, help="Target input prompt token length")
    parser.add_argument("--output-tokens", type=int, default=None, help="Target completion token length")
    parser.add_argument("--scenario-tag", type=str, default="benchmark", help="Tag for scenario identification")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Print calibrated prompt and token count without sending")

    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name = cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
    server_url = cfg.get("server_url", "http://127.0.0.1:8000/v1")
    results_dir = cfg.get("results_dir", "results")

    # Dry-run handling
    if args.dry_run:
        prompt_len = args.prompt_tokens or 128
        calibrator = PromptCalibrator(model_name)
        prompt = calibrator.build_prompt(prompt_len)
        token_count = calibrator.count_tokens(prompt)
        print("=" * 60)
        print(f"LOAD GENERATOR DRY RUN (Target: {prompt_len} tokens)")
        print("=" * 60)
        print(f"Model:                {model_name}")
        print(f"Calibrated Tokens:    {token_count}")
        print(f"Generated Prompt:\n{prompt}")
        print("=" * 60)
        sys.exit(0)

    # Defaults from config first scenario or defaults
    first_scenario = {}
    if cfg.get("scenarios") and isinstance(cfg["scenarios"], list) and len(cfg["scenarios"]) > 0:
        first_scenario = cfg["scenarios"][0]

    rps = args.rps if args.rps is not None else float(first_scenario.get("rps", 1.0))
    num_requests = args.num_requests if args.num_requests is not None else int(first_scenario.get("num_requests", 10))
    prompt_tokens = args.prompt_tokens if args.prompt_tokens is not None else int(first_scenario.get("prompt_tokens", 128))
    output_tokens = args.output_tokens if args.output_tokens is not None else int(first_scenario.get("output_tokens", 64))
    scenario_tag = args.scenario_tag or first_scenario.get("name", "benchmark")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = args.out or os.path.join(results_dir, f"{scenario_tag}_{timestamp}.csv")

    asyncio.run(
        run_load_generator(
            server_url=server_url,
            model_name=model_name,
            rps=rps,
            num_requests=num_requests,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            scenario_tag=scenario_tag,
            output_csv=output_csv,
        )
    )


if __name__ == "__main__":
    main()
