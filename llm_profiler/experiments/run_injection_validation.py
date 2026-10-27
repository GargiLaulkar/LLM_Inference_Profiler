"""Live and synthetic validation harness for M2 latency-injection hooks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..analyzer.bottleneck_analyzer import BottleneckAnalyzer
from ..analyzer.readers import CONTRACT_CSV_COLUMNS
from ..analyzer.bottleneck_injector import BottleneckInjector
from .load_generator import load_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "results"
DEFAULT_CONFIG = ROOT / "config.yaml"
TRACE_PATH_ENV = "LLMPROF_TRACE_PATH"
CSV_FIELDS = [
    "scenario",
    "hook_name",
    "hook_status",
    "repeat",
    "expected_stage",
    "top_stage",
    "top3_stages",
    "status",
    "data_source",
    "false_positive",
    "message",
    "request_csv",
    "trace_jsonl",
]

INJECTIONS: Dict[str, Dict[str, Any]] = {
    "slow_tokenizer": {
        "env_var": "LLMPROF_SLOW_TOKENIZER_MS",
        "env_value": "500",
        "synthetic_delay_ms": 500,
        "expected_stage": "tokenization",
    },
    "no_batching": {
        "env_var": "LLMPROF_NO_BATCHING",
        "env_value": "1",
        "synthetic_delay_ms": 800,
        "expected_stage": "queue_wait",
    },
    "cpu_preprocessing": {
        "env_var": "LLMPROF_CPU_PREPROCESS_MS",
        "env_value": "400",
        "synthetic_delay_ms": 400,
        "expected_stage": "tokenization",
    },
}


def detect_available_hooks(repo_root: Path = ROOT) -> Dict[str, bool]:
    """Find hook references in implementation sources, excluding docs and tests."""
    excluded = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "docs", "tests", "results"}
    supported_suffixes = {".py", ".ps1", ".sh", ".yaml", ".yml"}
    matches = {name: False for name in INJECTIONS}

    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in supported_suffixes:
            continue
        if excluded.intersection(path.relative_to(repo_root).parts):
            continue
        if path.name == Path(__file__).name:
            continue
        source = path.read_text(encoding="utf-8")
        for name, spec in INJECTIONS.items():
            if spec["env_var"] in source:
                matches[name] = True
    return matches


def _write_synthetic_trial(
    csv_path: Path,
    trace_path: Path,
    *,
    num_requests: int,
    rps: float,
    prompt_tokens: int,
    output_tokens: int,
    scenario_tag: str,
    injection: Optional[Dict[str, Any]],
    seed: int,
) -> None:
    """Write explicitly synthetic contract data for the mocked load-generator call."""
    injector = BottleneckInjector(seed=seed)
    injected_stage = injection["expected_stage"] if injection else None
    delay_s = float(injection["synthetic_delay_ms"]) / 1000.0 if injection else 0.0
    rows, traces = injector.generate_synthetic_workload(
        num_requests=num_requests,
        scenario_tag=scenario_tag,
        target_rps=rps,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        injected_stage=injected_stage,
        injection_delay_s=delay_s,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CONTRACT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with trace_path.open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace) + "\n")


def _mocked_subprocess_run(
    command: Sequence[str],
    *,
    synthetic_trial: Dict[str, Any],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Mock M1's CLI subprocess while materializing a synthetic CSV and trace."""
    required_args = (
        "--rps",
        "--num-requests",
        "--prompt-tokens",
        "--output-tokens",
        "--scenario-tag",
        "--out",
    )
    if any(arg not in command for arg in required_args):
        raise ValueError("Mock load-generator call is missing a required CLI argument")
    _write_synthetic_trial(**synthetic_trial)
    return subprocess.CompletedProcess(args=list(command), returncode=0, stdout="", stderr="")


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _command_runner(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, check=False)


def _server_health_url(server_url: str) -> str:
    base = server_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/health"


def _wait_for_server(
    process: Any,
    health_url: str,
    startup_timeout_s: float,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    deadline = time.monotonic() + startup_timeout_s
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server process exited before becoming healthy (exit {process.returncode})")
        try:
            with urlopen(health_url, timeout=1.0) as response:
                if 200 <= int(response.getcode()) < 400:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(f"Server did not become healthy at {health_url}: {last_error}")


def _start_server(
    command: Sequence[str],
    *,
    hook_env: Optional[Dict[str, str]],
    trace_path: Path,
    health_url: str,
    repo_root: Path,
    startup_timeout_s: float,
    popen: Callable[..., Any] = subprocess.Popen,
) -> Any:
    environment = os.environ.copy()
    for spec in INJECTIONS.values():
        environment.pop(spec["env_var"], None)
    environment.update(hook_env or {})
    environment[TRACE_PATH_ENV] = str(trace_path)
    process = popen(
        list(command),
        cwd=str(repo_root),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_server(process, health_url, startup_timeout_s)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    return process


def _stop_server(process: Any) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _load_generator_command(
    *,
    config_path: Path,
    csv_path: Path,
    scenario_tag: str,
    rps: float,
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "llm_profiler.experiments.load_generator",
        "--config",
        str(config_path),
        "--rps",
        str(rps),
        "--num-requests",
        str(num_requests),
        "--prompt-tokens",
        str(prompt_tokens),
        "--output-tokens",
        str(output_tokens),
        "--scenario-tag",
        scenario_tag,
        "--out",
        str(csv_path),
    ]


def _execute_trial(
    *,
    scenario: str,
    repeat_index: int,
    hook_status: str,
    expected_stage: str,
    injection: Optional[Dict[str, Any]],
    data_source: str,
    config_path: Path,
    config: Dict[str, Any],
    output_dir: Path,
    repo_root: Path,
    rps: float,
    num_requests: int,
    prompt_tokens: int,
    output_tokens: int,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
    server_command: Optional[Sequence[str]],
    startup_timeout_s: float,
    batching_enabled: Optional[bool],
) -> Dict[str, Any]:
    tag = f"e1_{scenario}_repeat_{repeat_index:02d}"
    csv_path = output_dir / f"{tag}.csv"
    trace_path = output_dir / f"{tag}_traces.jsonl"

    if data_source == "synthetic":
        synthetic_trial = {
            "csv_path": csv_path,
            "trace_path": trace_path,
            "num_requests": num_requests,
            "rps": rps,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "scenario_tag": tag,
            "injection": injection,
            "seed": repeat_index,
        }
        command = _load_generator_command(
            config_path=config_path,
            csv_path=csv_path,
            scenario_tag=tag,
            rps=rps,
            num_requests=num_requests,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        completed = command_runner(command, synthetic_trial=synthetic_trial)
    else:
        if not server_command:
            raise ValueError("--server-command is required to run an available hook live")
        if trace_path.exists():
            trace_path.unlink()
        server_url = str(config.get("server_url", "http://127.0.0.1:8000/v1"))
        hook_env = {injection["env_var"]: injection["env_value"]} if injection else {}
        process = _start_server(
            server_command,
            hook_env=hook_env,
            trace_path=trace_path,
            health_url=_server_health_url(server_url),
            repo_root=repo_root,
            startup_timeout_s=startup_timeout_s,
        )
        try:
            command = _load_generator_command(
                config_path=config_path,
                csv_path=csv_path,
                scenario_tag=tag,
                rps=rps,
                num_requests=num_requests,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            completed = command_runner(command, cwd=repo_root)
        finally:
            _stop_server(process)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Load generator failed for {tag} (exit {completed.returncode}): {completed.stderr}"
        )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Load generator did not create its request CSV: {csv_path}")
    if not trace_path.is_file():
        raise FileNotFoundError(
            f"No per-request trace JSONL was created at {trace_path}; "
            f"the server must honor {TRACE_PATH_ENV}"
        )

    result = BottleneckAnalyzer(config_path=config_path).analyze(
        csv_path=csv_path,
        traces_path=trace_path,
        max_bottlenecks=3,
        batching_enabled=batching_enabled,
    )
    top_stages = [item["stage"] for item in result["top_bottlenecks"]]
    top_stage = top_stages[0] if top_stages else ""
    passed = top_stage == expected_stage
    return {
        "scenario": scenario,
        "hook_name": injection["env_var"] if injection else "",
        "hook_status": hook_status,
        "repeat": str(repeat_index),
        "expected_stage": expected_stage,
        "top_stage": top_stage,
        "top3_stages": json.dumps(top_stages),
        "status": "pass" if passed else "fail",
        "data_source": data_source,
        "false_positive": "",
        "message": "",
        "request_csv": str(csv_path),
        "trace_jsonl": str(trace_path),
    }


def _make_summary(rows: Iterable[Dict[str, Any]], hooks: Dict[str, bool], data_source: str) -> Dict[str, Any]:
    materialized = list(rows)
    evaluated = [
        row for row in materialized
        if row.get("scenario") not in {"control", "pending"}
        and row.get("status") in {"pass", "fail"}
    ]
    correct = sum(row["status"] == "pass" for row in evaluated)
    top3_hits = 0
    for row in evaluated:
        try:
            top3_hits += row["expected_stage"] in json.loads(row["top3_stages"])
        except (KeyError, json.JSONDecodeError, TypeError):
            continue

    control_rows = [row for row in materialized if row.get("scenario") == "control"]
    control = control_rows[-1] if control_rows else None
    false_positive: Optional[bool] = None
    if control:
        try:
            false_positive = control["top_stage"] in {
                spec["expected_stage"] for spec in INJECTIONS.values()
            }
        except KeyError:
            false_positive = None

    pending = [name for name, available in hooks.items() if not available]
    failed = any(row.get("status") == "fail" for row in materialized)
    if failed:
        status = "fail"
    elif pending:
        status = "pending_live"
    else:
        status = "pass"

    return {
        "experiment_id": "E1",
        "status": status,
        "data_source": data_source,
        "pending_live_injections": pending,
        "detection_accuracy": correct / len(evaluated) if evaluated else None,
        "top3_hit_rate": top3_hits / len(evaluated) if evaluated else None,
        "false_positive_check": {
            "status": (
                "not_run" if control is None
                else "false_positive" if false_positive
                else "pass"
            ),
            "false_positive": false_positive,
            "control_top_stage": control.get("top_stage") if control else None,
        },
        "evaluated_runs": len(evaluated),
        "passed_runs": correct,
        "failed_runs": len(evaluated) - correct,
    }


def run_validation(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    repeat: int = 3,
    num_requests: int = 5,
    rps: Optional[float] = None,
    prompt_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    dry_run: bool = False,
    server_command: Optional[Sequence[str]] = None,
    startup_timeout_s: float = 180.0,
    resume: bool = True,
    repo_root: Path = ROOT,
    available_hooks: Optional[Dict[str, bool]] = None,
    command_runner: Optional[Callable[..., subprocess.CompletedProcess[str]]] = None,
) -> Dict[str, Any]:
    """Run control and repeated injections, persisting every completed trial."""
    if repeat < 3:
        raise ValueError("repeat must be at least three")
    if num_requests < 1:
        raise ValueError("num_requests must be at least one")

    config = load_config(str(config_path))
    scenarios = config.get("scenarios") or [{}]
    baseline = scenarios[0] if isinstance(scenarios, list) and scenarios else {}
    resolved_rps = float(rps if rps is not None else baseline.get("rps", 1.0))
    resolved_prompt_tokens = int(
        prompt_tokens if prompt_tokens is not None else baseline.get("prompt_tokens", 128)
    )
    resolved_output_tokens = int(
        output_tokens if output_tokens is not None else baseline.get("output_tokens", 64)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_path = output_dir / "e1_detection.csv"
    summary_path = output_dir / "e1_summary.json"
    rows = _load_rows(detection_path) if resume else []
    hooks = available_hooks if available_hooks is not None else detect_available_hooks(repo_root)
    source = "synthetic" if dry_run else "live"
    runner = command_runner or (_mocked_subprocess_run if dry_run else _command_runner)

    def completed(scenario: str, repeat_index: int) -> bool:
        return any(
            row.get("scenario") == scenario
            and row.get("repeat") == str(repeat_index)
            and row.get("data_source") == source
            and row.get("status") not in {"pending", ""}
            for row in rows
        )

    def save(row: Dict[str, Any]) -> None:
        rows.append(row)
        _write_rows(detection_path, rows)

    # Live control is run only when at least one real hook can be exercised.
    should_run_control = dry_run or any(hooks.values())
    if should_run_control and not completed("control", 1):
        try:
            control_row = _execute_trial(
                scenario="control",
                repeat_index=1,
                hook_status="control",
                expected_stage="measured",
                injection=None,
                data_source=source,
                config_path=config_path,
                config=config,
                output_dir=output_dir,
                repo_root=repo_root,
                rps=resolved_rps,
                num_requests=num_requests,
                prompt_tokens=resolved_prompt_tokens,
                output_tokens=resolved_output_tokens,
                command_runner=runner,
                server_command=server_command,
                startup_timeout_s=startup_timeout_s,
                batching_enabled=None,
            )
            control_row["status"] = "pass"
            save(control_row)
        except Exception as exc:
            if not dry_run:
                raise
            raise RuntimeError(f"Synthetic control dry-run failed: {exc}") from exc

    for name, injection in INJECTIONS.items():
        available = bool(hooks.get(name, False))
        if not dry_run and not available:
            if not any(row.get("scenario") == "pending" and row.get("hook_name") == injection["env_var"] for row in rows):
                save({
                    "scenario": "pending",
                    "hook_name": injection["env_var"],
                    "hook_status": "pending",
                    "repeat": "",
                    "expected_stage": injection["expected_stage"],
                    "top_stage": "",
                    "top3_stages": "[]",
                    "status": "pending",
                    "data_source": "live",
                    "false_positive": "",
                    "message": "Hook implementation not found in repository sources.",
                    "request_csv": "",
                    "trace_jsonl": "",
                })
            continue

        scenario = name
        for repeat_index in range(1, repeat + 1):
            if completed(scenario, repeat_index):
                continue
            row = _execute_trial(
                scenario=scenario,
                repeat_index=repeat_index,
                hook_status="available" if available else "synthetic",
                expected_stage=injection["expected_stage"],
                injection=injection,
                data_source=source,
                config_path=config_path,
                config=config,
                output_dir=output_dir,
                repo_root=repo_root,
                rps=resolved_rps,
                num_requests=num_requests,
                prompt_tokens=resolved_prompt_tokens,
                output_tokens=resolved_output_tokens,
                command_runner=runner,
                server_command=server_command,
                startup_timeout_s=startup_timeout_s,
                batching_enabled=False if name == "no_batching" else True,
            )
            save(row)

    summary = _make_summary(rows, hooks, source)
    if should_run_control:
        control = next((row for row in reversed(rows) if row.get("scenario") == "control"), None)
        if control and control.get("top_stage"):
            control["false_positive"] = str(
                control["top_stage"] in {spec["expected_stage"] for spec in INJECTIONS.values()}
            ).lower()
            control["status"] = "fail" if control["false_positive"] == "true" else "pass"
            _write_rows(detection_path, rows)
            summary = _make_summary(rows, hooks, source)
    summary["repeat_requested"] = repeat
    summary["request_count_per_trial"] = num_requests
    summary["detection_csv"] = str(detection_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _server_command_arg(value: str) -> List[str]:
    try:
        command = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("server command must be a JSON string array") from exc
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise argparse.ArgumentTypeError("server command must be a non-empty JSON string array")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate live M2 latency injections; --dry-run uses only synthetic data."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeat", type=int, default=3, help="Runs per injection (minimum 3)")
    parser.add_argument("--num-requests", type=int, default=5, help="Small requests per run")
    parser.add_argument("--rps", type=float, default=None)
    parser.add_argument("--prompt-tokens", type=int, default=None)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--server-command",
        type=_server_command_arg,
        default=None,
        help="Live server command as a JSON string array; never invoked by --dry-run.",
    )
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_validation(
        config_path=args.config,
        output_dir=args.output_dir,
        repeat=args.repeat,
        num_requests=args.num_requests,
        rps=args.rps,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        dry_run=args.dry_run,
        server_command=args.server_command,
        startup_timeout_s=args.startup_timeout,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] in {"pass", "pending_live"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
