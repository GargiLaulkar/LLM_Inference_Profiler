#!/usr/bin/env python3
"""M3 GPU & Deep-Dive Telemetry Hello-World and Environment Verification.

Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks
Role: M3 (GPU and Deep-Dive Lead)

Purpose:
    Standalone GPU discovery and diagnostic tool using NVML (pynvml / nvidia-ml-py)
    and standard system utilities. Verifies NVIDIA drivers, CUDA versions, VRAM,
    instantaneous power draw, core utilization, thermal stats, and developer tooling
    (nvidia-smi, nsys, docker, torch).

Design Constraints:
    - Zero intra-repo dependencies: only standard library and pynvml.
    - Can be copied standalone directly into Kaggle, Google Colab, or any Linux/Windows host.
    - Graceful non-crashing execution on non-NVIDIA hosts (e.g. Intel integrated graphics).

Usage:
    # Basic check (single snapshot + 5-sample live table if GPU available):
    python llm_profiler/collectors/gpu_hello.py

    # Custom sample count and interval (e.g. 3 samples, 0.5s interval):
    python llm_profiler/collectors/gpu_hello.py --samples 3 --interval 0.5

    # Machine-readable JSON output:
    python llm_profiler/collectors/gpu_hello.py --json

    # Save Markdown report to file:
    python llm_profiler/collectors/gpu_hello.py --out docs/m3/env_check_output.md

Prometheus Shared Contract References (for subsequent M3 exporter):
    # Metric: llmprof_gpu_utilization_percent{gpu="<index>"}
    # Metric: llmprof_gpu_memory_used_bytes{gpu="<index>"}
    # Metric: llmprof_gpu_power_watts{gpu="<index>"}
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import pynvml  # Provided by package 'nvidia-ml-py'
except ImportError:
    pynvml = None


def check_tooling_and_env() -> Dict[str, Any]:
    """Inspect system environment, compiler/profiler toolchains, and optional libraries.

    Returns:
        Dictionary mapping tool names to status details (OK, MISSING, SKIPPED, ERROR).
    """
    report: Dict[str, Any] = {
        "os": {
            "status": "OK",
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "python": {
            "status": "OK",
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
    }

    # 1. nvidia-smi CLI verification
    smi_path = shutil.which("nvidia-smi")
    if smi_path:
        try:
            res = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            first_line = ""
            for line in res.stdout.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("+") and not stripped.startswith("|"):
                    first_line = stripped
                    break
            if not first_line and res.stdout.strip():
                first_line = res.stdout.strip().splitlines()[0]

            report["nvidia-smi"] = {
                "status": "OK" if res.returncode == 0 else "ERROR",
                "path": smi_path,
                "first_line": first_line or "nvidia-smi executed successfully",
                "returncode": res.returncode,
            }
        except Exception as exc:
            report["nvidia-smi"] = {
                "status": "ERROR",
                "path": smi_path,
                "reason": str(exc),
            }
    else:
        report["nvidia-smi"] = {
            "status": "MISSING",
            "reason": "nvidia-smi binary not found on system PATH",
        }

    # 2. nsys (NVIDIA Nsight Systems) CLI verification
    nsys_path = shutil.which("nsys")
    if nsys_path:
        try:
            res = subprocess.run(
                ["nsys", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            version_str = res.stdout.strip() or res.stderr.strip()
            first_line = version_str.splitlines()[0] if version_str else "nsys detected"
            report["nsys"] = {
                "status": "OK" if res.returncode == 0 else "ERROR",
                "path": nsys_path,
                "version": first_line,
                "returncode": res.returncode,
            }
        except Exception as exc:
            report["nsys"] = {
                "status": "ERROR",
                "path": nsys_path,
                "reason": str(exc),
            }
    else:
        report["nsys"] = {
            "status": "MISSING",
            "reason": "nsys (Nsight Systems) binary not found on system PATH",
        }

    # 3. Linux kernel perf_event_paranoid check
    if platform.system() == "Linux":
        paranoid_path = "/proc/sys/kernel/perf_event_paranoid"
        if os.path.exists(paranoid_path):
            try:
                with open(paranoid_path, "r", encoding="utf-8") as f:
                    val = f.read().strip()
                report["perf_event_paranoid"] = {
                    "status": "OK",
                    "value": val,
                    "guidance": (
                        "Value <= 2 permits CPU/GPU uncore counters; "
                        "value <= 1 permits kernel profiling."
                    ),
                }
            except Exception as exc:
                report["perf_event_paranoid"] = {
                    "status": "ERROR",
                    "reason": str(exc),
                }
        else:
            report["perf_event_paranoid"] = {
                "status": "MISSING",
                "reason": f"{paranoid_path} not found on this Linux system",
            }
    else:
        report["perf_event_paranoid"] = {
            "status": "SKIPPED",
            "reason": f"Non-Linux OS ({platform.system()})",
        }

    # 4. PyTorch optional installation and CUDA support check
    try:
        import torch  # type: ignore

        cuda_avail = bool(torch.cuda.is_available())
        report["torch"] = {
            "status": "OK",
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": cuda_avail,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if cuda_avail else 0,
        }
    except ImportError:
        report["torch"] = {
            "status": "MISSING",
            "reason": "torch package not installed in current Python environment",
        }
    except Exception as exc:
        report["torch"] = {
            "status": "ERROR",
            "reason": str(exc),
        }

    # 5. Docker CLI verification
    docker_path = shutil.which("docker")
    if docker_path:
        try:
            res = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            v_line = res.stdout.strip().splitlines()[0] if res.stdout.strip() else "docker detected"
            report["docker"] = {
                "status": "OK" if res.returncode == 0 else "ERROR",
                "path": docker_path,
                "version": v_line,
                "returncode": res.returncode,
            }
        except Exception as exc:
            report["docker"] = {
                "status": "ERROR",
                "path": docker_path,
                "reason": str(exc),
            }
    else:
        report["docker"] = {
            "status": "MISSING",
            "reason": "docker binary not found on system PATH",
        }

    return report


def _decode_string(val: Any) -> str:
    """Helper to convert bytes or string attributes safely to UTF-8 str."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _format_cuda_driver_version(ver: Any) -> str:
    """Format numeric CUDA driver version (e.g. 12040 -> '12.4')."""
    if isinstance(ver, int):
        major = ver // 1000
        minor = (ver % 1000) // 10
        return f"{major}.{minor}"
    return str(ver)


def collect_gpu_telemetry() -> Tuple[bool, Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Initialize NVML, inspect all NVIDIA GPUs, and gather hardware metrics.

    Each metric call is individually wrapped in try/except to gracefully handle
    unsupported features (such as power draw on virtualized or legacy GPUs).

    Returns:
        Tuple of:
            - nvml_available (bool)
            - error_message (Optional[str])
            - list of device dictionaries
            - driver_info dictionary
    """
    if pynvml is None:
        return (
            False,
            "pynvml module not installed (install via 'pip install nvidia-ml-py')",
            [],
            {},
        )

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        return (
            False,
            f"NVML initialization failed: {type(exc).__name__} - {exc}",
            [],
            {},
        )

    devices: List[Dict[str, Any]] = []
    driver_info: Dict[str, Any] = {}

    try:
        # System-level driver info
        try:
            driver_info["driver_version"] = _decode_string(pynvml.nvmlSystemGetDriverVersion())
        except Exception as exc:
            driver_info["driver_version"] = f"unsupported ({exc})"

        try:
            raw_cuda = pynvml.nvmlSystemGetCudaDriverVersion()
            driver_info["cuda_driver_version"] = _format_cuda_driver_version(raw_cuda)
        except Exception as exc:
            driver_info["cuda_driver_version"] = f"unsupported ({exc})"

        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception as exc:
            return (
                True,
                f"Failed to query device count: {exc}",
                [],
                driver_info,
            )

        driver_info["device_count"] = device_count

        for idx in range(device_count):
            dev_metrics: Dict[str, Any] = {"index": idx}
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            except Exception as exc:
                dev_metrics["error"] = f"Failed to acquire handle: {exc}"
                devices.append(dev_metrics)
                continue

            # Device Name
            try:
                raw_name = pynvml.nvmlDeviceGetName(handle)
                dev_metrics["name"] = _decode_string(raw_name)
            except Exception:
                dev_metrics["name"] = "unsupported"

            # Memory details (total, used, free in MiB and bytes)
            # Prometheus contract: llmprof_gpu_memory_used_bytes{gpu}
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                dev_metrics["memory_total_bytes"] = int(mem.total)
                dev_metrics["memory_used_bytes"] = int(mem.used)
                dev_metrics["memory_free_bytes"] = int(mem.free)
                dev_metrics["memory_total_mib"] = round(mem.total / (1024.0 * 1024.0), 2)
                dev_metrics["memory_used_mib"] = round(mem.used / (1024.0 * 1024.0), 2)
                dev_metrics["memory_free_mib"] = round(mem.free / (1024.0 * 1024.0), 2)
            except Exception:
                dev_metrics["memory_total_mib"] = "unsupported"
                dev_metrics["memory_used_mib"] = "unsupported"
                dev_metrics["memory_free_mib"] = "unsupported"
                dev_metrics["memory_used_bytes"] = "unsupported"

            # Core and Memory Utilization percentages
            # Prometheus contract: llmprof_gpu_utilization_percent{gpu}
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                dev_metrics["gpu_utilization_percent"] = float(rates.gpu)
                dev_metrics["memory_utilization_percent"] = float(rates.memory)
            except Exception:
                dev_metrics["gpu_utilization_percent"] = "unsupported"
                dev_metrics["memory_utilization_percent"] = "unsupported"

            # Power usage in watts (NVML returns milliwatts)
            # Prometheus contract: llmprof_gpu_power_watts{gpu}
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                dev_metrics["power_draw_watts"] = round(power_mw / 1000.0, 2)
            except Exception:
                dev_metrics["power_draw_watts"] = "unsupported"

            # Power Management Limit in watts (NVML returns milliwatts)
            try:
                limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                dev_metrics["power_limit_watts"] = round(limit_mw / 1000.0, 2)
            except Exception:
                try:
                    limit_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
                    dev_metrics["power_limit_watts"] = round(limit_mw / 1000.0, 2)
                except Exception:
                    dev_metrics["power_limit_watts"] = "unsupported"

            # Temperature in Celsius
            try:
                temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                dev_metrics["temperature_celsius"] = int(temp_c)
            except Exception:
                dev_metrics["temperature_celsius"] = "unsupported"

            devices.append(dev_metrics)

        return (True, None, devices, driver_info)

    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def sample_live_metrics(samples: int = 5, interval: float = 1.0) -> List[Dict[str, Any]]:
    """Sample GPU utilization, memory usage, and power draw repeatedly.

    Returns a history of snapshots so transient changes under load can be monitored.
    """
    if pynvml is None or samples <= 0:
        return []

    try:
        pynvml.nvmlInit()
    except Exception:
        return []

    history: List[Dict[str, Any]] = []

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for sample_i in range(1, samples + 1):
            timestamp = time.time()
            sample_record: Dict[str, Any] = {
                "sample_index": sample_i,
                "timestamp": timestamp,
                "gpus": [],
            }

            for idx in range(device_count):
                gpu_entry: Dict[str, Any] = {"index": idx}
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

                    # Utilization
                    # Prometheus contract: llmprof_gpu_utilization_percent{gpu}
                    try:
                        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_entry["gpu_utilization_percent"] = float(rates.gpu)
                        gpu_entry["mem_utilization_percent"] = float(rates.memory)
                    except Exception:
                        gpu_entry["gpu_utilization_percent"] = "unsupported"
                        gpu_entry["mem_utilization_percent"] = "unsupported"

                    # Memory (MiB & bytes)
                    # Prometheus contract: llmprof_gpu_memory_used_bytes{gpu}
                    try:
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_entry["memory_used_mib"] = round(mem.used / (1024.0 * 1024.0), 2)
                        gpu_entry["memory_used_bytes"] = int(mem.used)
                    except Exception:
                        gpu_entry["memory_used_mib"] = "unsupported"
                        gpu_entry["memory_used_bytes"] = "unsupported"

                    # Power (watts)
                    # Prometheus contract: llmprof_gpu_power_watts{gpu}
                    try:
                        pmw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        gpu_entry["power_watts"] = round(pmw / 1000.0, 2)
                    except Exception:
                        gpu_entry["power_watts"] = "unsupported"

                except Exception as exc:
                    gpu_entry["error"] = str(exc)

                sample_record["gpus"].append(gpu_entry)

            history.append(sample_record)

            if sample_i < samples and interval > 0.0:
                time.sleep(interval)

        return history

    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Format complete diagnostic results as a clean Markdown report."""
    lines: List[str] = [
        "# M3 GPU Environment & Telemetry Verification Report",
        "",
        f"**Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"**Operating System**: {data['system']['os']['platform']}",
        f"**Python Executable**: `{data['system']['python']['executable']}` ({data['system']['python']['version']})",
        "",
        "---",
        "",
        "## 1. Tooling & Platform Verification",
        "",
        "| Component / Tool | Status | Details / Version |",
        "|---|---|---|",
    ]

    sys_info = data["system"]
    smi = sys_info.get("nvidia-smi", {})
    lines.append(f"| `nvidia-smi` | **{smi.get('status')}** | {smi.get('first_line') or smi.get('reason')} |")

    nsys = sys_info.get("nsys", {})
    lines.append(f"| `nsys` (Nsight Systems) | **{nsys.get('status')}** | {nsys.get('version') or nsys.get('reason')} |")

    perf = sys_info.get("perf_event_paranoid", {})
    perf_desc = f"value={perf.get('value')} ({perf.get('guidance', '')})" if perf.get("status") == "OK" else perf.get("reason", "")
    lines.append(f"| `perf_event_paranoid` | **{perf.get('status')}** | {perf_desc} |")

    torch_info = sys_info.get("torch", {})
    if torch_info.get("status") == "OK":
        t_desc = f"v{torch_info.get('version')}, cuda_available={torch_info.get('cuda_available')}, torch_cuda={torch_info.get('torch_cuda_version')}"
    else:
        t_desc = torch_info.get("reason", "")
    lines.append(f"| PyTorch | **{torch_info.get('status')}** | {t_desc} |")

    docker_info = sys_info.get("docker", {})
    lines.append(f"| Docker | **{docker_info.get('status')}** | {docker_info.get('version') or docker_info.get('reason')} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. NVIDIA NVML Telemetry Status",
        "",
    ])

    nvml = data.get("nvml", {})
    if not nvml.get("available"):
        lines.extend([
            "> [!NOTE]",
            f"> **NVML Offline**: {nvml.get('error', 'NVML library not initialized')}.",
            ">",
            "> This is normal and expected on developer laptops without an NVIDIA GPU (e.g. Intel HD graphics).",
            "> Run this script in Google Colab (T4 GPU runtime) or Kaggle to record live hardware metrics.",
            "",
        ])
    else:
        drv = nvml.get("driver_info", {})
        lines.extend([
            f"- **Driver Version**: `{drv.get('driver_version')}`",
            f"- **CUDA Driver Version**: `{drv.get('cuda_driver_version')}`",
            f"- **Total GPUs Detected**: `{drv.get('device_count')}`",
            "",
            "### GPU Specifications & Metrics Snapshot",
            "",
            "| GPU | Name | VRAM Total | VRAM Used | VRAM Free | Core Util | Mem Util | Power Draw | Power Limit | Temp |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ])

        for dev in nvml.get("devices", []):
            lines.append(
                f"| `{dev.get('index')}` | {dev.get('name')} | "
                f"{dev.get('memory_total_mib')} MiB | {dev.get('memory_used_mib')} MiB | {dev.get('memory_free_mib')} MiB | "
                f"{dev.get('gpu_utilization_percent')}% | {dev.get('memory_utilization_percent')}% | "
                f"{dev.get('power_draw_watts')} W | {dev.get('power_limit_watts')} W | {dev.get('temperature_celsius')} °C |"
            )
        lines.append("")

    # Sampling section
    samples = data.get("samples", [])
    if samples:
        lines.extend([
            "---",
            "",
            "## 3. Live Sampling Observations",
            "",
            "| Sample # | GPU | Core Util (%) | VRAM Used (MiB) | Power Draw (W) |",
            "|---|---|---|---|---|",
        ])
        for s in samples:
            for g in s.get("gpus", []):
                lines.append(
                    f"| {s.get('sample_index')} | `{g.get('index')}` | "
                    f"{g.get('gpu_utilization_percent')} | {g.get('memory_used_mib')} | {g.get('power_watts')} |"
                )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Shared Contract Prometheus Mapping",
        "",
        "The metrics collected above match the M3 Prometheus telemetry contracts:",
        "- `llmprof_gpu_utilization_percent{gpu=\"<index>\"}`",
        "- `llmprof_gpu_memory_used_bytes{gpu=\"<index>\"}`",
        "- `llmprof_gpu_power_watts{gpu=\"<index>\"}`",
    ])

    return "\n".join(lines)


def run_diagnostics(samples: int = 5, interval: float = 1.0) -> Dict[str, Any]:
    """Execute complete environment and NVML diagnostic workflow."""
    system_report = check_tooling_and_env()
    nvml_ok, nvml_err, devices, driver_info = collect_gpu_telemetry()

    sampling_history: List[Dict[str, Any]] = []
    if nvml_ok and samples > 0:
        sampling_history = sample_live_metrics(samples=samples, interval=interval)

    return {
        "timestamp": time.time(),
        "system": system_report,
        "nvml": {
            "available": nvml_ok,
            "error": nvml_err,
            "driver_info": driver_info,
            "devices": devices,
        },
        "samples": sampling_history,
    }


def print_console_report(data: Dict[str, Any], samples: int, interval: float) -> None:
    """Print human-readable diagnostic output to standard output."""
    sys_info = data["system"]
    nvml_info = data["nvml"]

    print("=" * 72)
    print("  PROJECT 6: M3 GPU & ENVIRONMENT DIAGNOSTIC CHECK (pynvml hello)")
    print("=" * 72)

    # 1. System Info
    print(f"\n[HOST ENVIRONMENT]")
    print(f"  OS Platform      : {sys_info['os']['platform']}")
    print(f"  Python Version   : {sys_info['python']['version']} ({sys_info['python']['executable']})")

    print(f"\n[TOOLING CHECKS]")
    for tool_name in ["nvidia-smi", "nsys", "perf_event_paranoid", "torch", "docker"]:
        tool_data = sys_info.get(tool_name, {})
        status = tool_data.get("status", "UNKNOWN")
        if status == "OK":
            desc = (
                tool_data.get("version")
                or tool_data.get("first_line")
                or f"value={tool_data.get('value')}"
            )
            print(f"  {tool_name:<20}: [  OK  ] {desc}")
        elif status == "SKIPPED":
            print(f"  {tool_name:<20}: [ SKIP ] {tool_data.get('reason')}")
        else:
            print(f"  {tool_name:<20}: [{status:^6}] {tool_data.get('reason', 'N/A')}")

    # 2. NVML Details
    print(f"\n[NVML TELEMETRY]")
    if not nvml_info.get("available"):
        print("  NVML Available   : NO")
        print(f"  Status / Reason  : {nvml_info.get('error')}")
        print("  -> Clean execution preserved. No NVIDIA GPU hardware detected on this host.")
        print("  -> Ready to be run on Kaggle/Colab with NVIDIA T4 runtime.")
    else:
        drv = nvml_info.get("driver_info", {})
        print("  NVML Available   : YES")
        print(f"  Driver Version   : {drv.get('driver_version')}")
        print(f"  CUDA Driver Ver  : {drv.get('cuda_driver_version')}")
        print(f"  Detected GPUs    : {drv.get('device_count')}")

        print("\n  Detailed GPU Metrics:")
        print("  " + "-" * 68)
        for dev in nvml_info.get("devices", []):
            idx = dev.get("index")
            name = dev.get("name")
            mem_tot = dev.get("memory_total_mib")
            mem_usd = dev.get("memory_used_mib")
            mem_fre = dev.get("memory_free_mib")
            core_u = dev.get("gpu_utilization_percent")
            mem_u = dev.get("memory_utilization_percent")
            p_draw = dev.get("power_draw_watts")
            p_lim = dev.get("power_limit_watts")
            temp = dev.get("temperature_celsius")

            print(f"  GPU [{idx}] {name}")
            print(f"    VRAM           : {mem_usd} MiB used / {mem_tot} MiB total ({mem_fre} MiB free)")
            print(f"    Utilization    : Core {core_u}% | Memory Controller {mem_u}%")
            print(f"    Power          : Draw {p_draw} W | Limit {p_lim} W")
            print(f"    Temperature    : {temp} °C")
            print("  " + "-" * 68)

    # 3. Sampling
    samples_list = data.get("samples", [])
    if samples_list:
        print(f"\n[LIVE SAMPLING: {len(samples_list)} samples @ {interval}s interval]")
        print("  " + "-" * 68)
        print(f"  {'Sample':<8} {'GPU':<6} {'Core Util (%)':<16} {'VRAM Used (MiB)':<18} {'Power (W)':<12}")
        print("  " + "-" * 68)
        for s in samples_list:
            s_idx = s.get("sample_index")
            for g in s.get("gpus", []):
                print(
                    f"  {s_idx:<8} {g.get('index'):<6} "
                    f"{str(g.get('gpu_utilization_percent')):<16} "
                    f"{str(g.get('memory_used_mib')):<18} "
                    f"{str(g.get('power_watts')):<12}"
                )
        print("  " + "-" * 68)

    print("\n" + "=" * 72)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="M3 GPU & Deep-Dive Environment Verification and NVML Hello-World"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of telemetry samples to read (default: 5)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval in seconds between samples (default: 1.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable report",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional file path to write Markdown summary report",
    )

    args = parser.parse_args(argv)

    results = run_diagnostics(samples=args.samples, interval=args.interval)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_console_report(results, samples=args.samples, interval=args.interval)

    if args.out:
        md_content = generate_markdown_report(results)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md_content)
        if not args.json:
            print(f"[REPORT SAVED] Markdown summary saved to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
