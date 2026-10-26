#!/usr/bin/env python3
"""M3 GPU Telemetry Prometheus Exporter Daemon.

Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks
Role: M3 (GPU and Deep-Dive Lead)

Purpose:
    Exposes real-time NVIDIA GPU telemetry on port 9102 (/metrics) for scraping
    by Prometheus. Follows the shared project contract for metric names, types,
    and labels.

Shared Contract Metrics:
    - llmprof_gpu_utilization_percent (Gauge, label: gpu) - range 0-100
    - llmprof_gpu_memory_used_bytes   (Gauge, label: gpu) - bytes allocated
    - llmprof_gpu_power_watts         (Gauge, label: gpu) - instantaneous watts

Memory Breakdown Metric (Task 5b):
    - llmprof_memory_breakdown_bytes  (Gauge, labels: component, phase) - bytes
      Read from memory_breakdown.json produced by deep_dive mode. Static snapshot;
      updates only when the file changes on disk (mtime or size change).

Design:
    - Uses a dedicated CollectorRegistry for clean test isolation.
    - Decoupled NVML reading (read_gpu_samples) and gauge updating (update_gauges).
    - Logs a single warning per unsupported metric and leaves unsupported gauges absent.
    - Gracefully stops on SIGINT (Ctrl+C) and SIGTERM with guaranteed nvmlShutdown().
    - Single-shot execution via --once for dry-runs and automated verification.

Usage:
    # Run continuous daemon with config/CLI defaults (port 9102, 1.0s interval):
    python llm_profiler/collectors/gpu_collector.py

    # Override port and interval:
    python llm_profiler/collectors/gpu_collector.py --port 9102 --interval 0.5

    # Run single snapshot to stdout (no server started):
    python llm_profiler/collectors/gpu_collector.py --once

    # Override memory breakdown file path (empty string disables):
    python llm_profiler/collectors/gpu_collector.py --memory-breakdown-file /path/to/memory_breakdown.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from prometheus_client import CollectorRegistry, Gauge, generate_latest, start_http_server

try:
    import pynvml
except ImportError:
    pynvml = None

# ─── DEFAULT CONSTANTS ────────────────────────────────────────────────────────
DEFAULT_PORT: int = 9102
DEFAULT_SAMPLE_INTERVAL_S: float = 1.0
DEFAULT_CONFIG_PATH: str = "config.yaml"
DEFAULT_MEMORY_BREAKDOWN_FILE: str = "results/deep_dive/memory_breakdown.json"

# Contract: all 8 (component, phase) combinations that must exist in a valid file
_BREAKDOWN_COMPONENTS = ("weights", "kv_cache", "activations", "other")
_BREAKDOWN_PHASES = ("prefill", "decode")
_BREAKDOWN_KEYS: Tuple[Tuple[str, str], ...] = tuple(
    (comp, phase)
    for phase in _BREAKDOWN_PHASES
    for comp in _BREAKDOWN_COMPONENTS
)

# Global state for termination signal and warning suppression
_stop_event = threading.Event()
_warned_unsupported: Set[Tuple[int, str]] = set()

logger = logging.getLogger("gpu_collector")


def reset_collector_state() -> None:
    """Reset global termination and warning state (primarily for unit tests)."""
    _stop_event.clear()
    _warned_unsupported.clear()


def load_collector_config(
    config_path: Optional[str] = None,
) -> Tuple[int, float, str, str]:
    """Load port, sample interval, and memory_breakdown_file from config.yaml.

    Args:
        config_path: Optional explicit path to configuration YAML.

    Returns:
        Tuple of (port, sample_interval_s, memory_breakdown_file, source_description).
    """
    candidate_paths: List[str] = []
    if config_path:
        candidate_paths.append(config_path)
    else:
        # Standard search paths in repository
        candidate_paths.extend([
            DEFAULT_CONFIG_PATH,
            os.path.join("llm_profiler", DEFAULT_CONFIG_PATH),
            os.path.join(os.path.dirname(__file__), "..", DEFAULT_CONFIG_PATH),
            os.path.join(os.path.dirname(__file__), "..", "..", DEFAULT_CONFIG_PATH),
        ])

    for path in candidate_paths:
        abs_p = os.path.abspath(path)
        if os.path.isfile(abs_p):
            try:
                with open(abs_p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                gpu_cfg = data.get("gpu_collector", {})
                port = int(gpu_cfg.get("port", DEFAULT_PORT))
                interval = float(gpu_cfg.get("sample_interval_s", DEFAULT_SAMPLE_INTERVAL_S))
                breakdown_file = str(
                    gpu_cfg.get("memory_breakdown_file", DEFAULT_MEMORY_BREAKDOWN_FILE)
                )
                source = f"config file '{path}'"
                logger.info(
                    "Loaded configuration from %s (port=%d, interval=%.1fs, breakdown_file=%s)",
                    path, port, interval, breakdown_file,
                )
                return port, interval, breakdown_file, source
            except Exception as exc:
                logger.warning("Failed to parse config at '%s': %s. Falling back.", path, exc)

    source = (
        f"default constants (port={DEFAULT_PORT}, interval={DEFAULT_SAMPLE_INTERVAL_S}s, "
        f"breakdown_file={DEFAULT_MEMORY_BREAKDOWN_FILE})"
    )
    logger.info("No config.yaml found; using %s", source)
    return DEFAULT_PORT, DEFAULT_SAMPLE_INTERVAL_S, DEFAULT_MEMORY_BREAKDOWN_FILE, source


def create_gpu_registry() -> Tuple[CollectorRegistry, Dict[str, Gauge]]:
    """Instantiate a dedicated CollectorRegistry and register the shared contract gauges.

    The breakdown gauge (llmprof_memory_breakdown_bytes) is always registered on
    the same registry so Prometheus scrapes it; samples are set only when valid
    data is loaded (no fake zeros are ever published).

    Returns:
        Tuple of (CollectorRegistry, dict of Gauges).
    """
    registry = CollectorRegistry()
    gauges: Dict[str, Gauge] = {
        "utilization": Gauge(
            "llmprof_gpu_utilization_percent",
            "GPU core utilization percentage (0-100)",
            ["gpu"],
            registry=registry,
        ),
        "memory": Gauge(
            "llmprof_gpu_memory_used_bytes",
            "GPU memory currently allocated in bytes",
            ["gpu"],
            registry=registry,
        ),
        "power": Gauge(
            "llmprof_gpu_power_watts",
            "GPU instantaneous power draw in watts",
            ["gpu"],
            registry=registry,
        ),
        "breakdown": Gauge(
            "llmprof_memory_breakdown_bytes",
            "Memory breakdown from last deep-dive run (static snapshot from memory_breakdown.json)",
            ["component", "phase"],
            registry=registry,
        ),
    }
    return registry, gauges


def init_nvml() -> bool:
    """Initialize the NVML library via pynvml.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    if pynvml is None:
        logger.error(
            "Cannot initialize NVML: pynvml module is not installed. "
            "Please install via 'pip install nvidia-ml-py'."
        )
        return False

    try:
        pynvml.nvmlInit()
        logger.info("NVML successfully initialized.")
        return True
    except Exception as exc:
        logger.error(
            "NVML initialization failed: %s - %s. Ensure an NVIDIA GPU and driver are present.",
            type(exc).__name__,
            exc,
        )
        return False


def shutdown_nvml() -> None:
    """Safely shut down NVML."""
    if pynvml is not None:
        try:
            pynvml.nvmlShutdown()
            logger.info("NVML shutdown completed.")
        except Exception as exc:
            logger.debug("nvmlShutdown exception ignored: %s", exc)


def _log_unsupported_once(gpu_index: int, metric_name: str, exc: Exception) -> None:
    """Log a warning for an unsupported metric exactly once per GPU."""
    key = (gpu_index, metric_name)
    if key not in _warned_unsupported:
        logger.warning(
            "GPU %d metric '%s' is unsupported on this device: %s. Gauge will remain omitted.",
            gpu_index,
            metric_name,
            exc,
        )
        _warned_unsupported.add(key)


def read_gpu_samples() -> List[Dict[str, Any]]:
    """Read telemetry metrics from all available NVIDIA GPUs via NVML.

    Returns:
        A list of dictionaries containing raw samples for each GPU index.
        Unsupported metrics are set to None to avoid publishing fake 0 values.
    """
    if pynvml is None:
        return []

    samples: List[Dict[str, Any]] = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception as exc:
        logger.error("Failed to query NVML device count: %s", exc)
        return []

    for idx in range(device_count):
        gpu_id = str(idx)
        record: Dict[str, Any] = {
            "gpu": gpu_id,
            "utilization_percent": None,
            "memory_used_bytes": None,
            "power_watts": None,
        }

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception as exc:
            logger.error("Failed to get device handle for GPU %d: %s", idx, exc)
            samples.append(record)
            continue

        # 1. Core Utilization percentage (0-100)
        # Shared Contract: llmprof_gpu_utilization_percent{gpu="<index>"}
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            record["utilization_percent"] = float(rates.gpu)
        except Exception as exc:
            _log_unsupported_once(idx, "llmprof_gpu_utilization_percent", exc)

        # 2. VRAM used in BYTES (not MiB)
        # Shared Contract: llmprof_gpu_memory_used_bytes{gpu="<index>"}
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            record["memory_used_bytes"] = int(mem.used)
        except Exception as exc:
            _log_unsupported_once(idx, "llmprof_gpu_memory_used_bytes", exc)

        # 3. Instantaneous power draw in WATTS (NVML returns milliwatts)
        # Shared Contract: llmprof_gpu_power_watts{gpu="<index>"}
        try:
            pmw = pynvml.nvmlDeviceGetPowerUsage(handle)
            record["power_watts"] = round(pmw / 1000.0, 3)
        except Exception as exc:
            _log_unsupported_once(idx, "llmprof_gpu_power_watts", exc)

        samples.append(record)

    return samples


def update_gauges(samples: List[Dict[str, Any]], gauges: Dict[str, Gauge]) -> None:
    """Update Prometheus gauge metrics from collected GPU sample dictionaries.

    If a metric is None (unsupported), its gauge is skipped for that GPU so that
    no misleading fake 0 is exported.
    """
    for item in samples:
        gpu_id = str(item.get("gpu", "0"))

        util_val = item.get("utilization_percent")
        if util_val is not None:
            gauges["utilization"].labels(gpu=gpu_id).set(util_val)

        mem_val = item.get("memory_used_bytes")
        if mem_val is not None:
            gauges["memory"].labels(gpu=gpu_id).set(mem_val)

        power_val = item.get("power_watts")
        if power_val is not None:
            gauges["power"].labels(gpu=gpu_id).set(power_val)


# ─── MEMORY BREAKDOWN EXPORT (Task 5b) ───────────────────────────────────────

def load_memory_breakdown(path: str) -> Optional[Dict[Tuple[str, str], float]]:
    """Load and validate memory_breakdown.json produced by deep_dive mode.

    Validates that all 8 (component, phase) combinations exist, are numeric,
    and are >= 0. Extra keys (nvml_used_bytes, torch_peak_allocated_bytes, etc.)
    are silently ignored. Returns None on any error and logs exactly one warning.

    Args:
        path: Absolute or relative path to memory_breakdown.json.

    Returns:
        Dict mapping (component, phase) -> float (bytes), or None if invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        # Caller handles missing file separately; do not log here
        return None
    except json.JSONDecodeError as exc:
        logger.warning("memory_breakdown.json at '%s' is not valid JSON: %s", path, exc)
        return None
    except Exception as exc:
        logger.warning("Failed to read memory_breakdown.json at '%s': %s", path, exc)
        return None

    if not isinstance(raw, dict):
        logger.warning(
            "memory_breakdown.json at '%s': top-level JSON value must be an object", path
        )
        return None

    phases_data = raw.get("phases")
    if not isinstance(phases_data, dict):
        logger.warning(
            "memory_breakdown.json at '%s': missing or invalid top-level 'phases' key", path
        )
        return None

    result: Dict[Tuple[str, str], float] = {}
    for phase in _BREAKDOWN_PHASES:
        phase_block = phases_data.get(phase)
        if not isinstance(phase_block, dict):
            logger.warning(
                "memory_breakdown.json at '%s': missing phase '%s'", path, phase
            )
            return None
        for comp in _BREAKDOWN_COMPONENTS:
            val = phase_block.get(comp)
            if val is None:
                logger.warning(
                    "memory_breakdown.json at '%s': missing component '%s' in phase '%s'",
                    path, comp, phase,
                )
                return None
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                logger.warning(
                    "memory_breakdown.json at '%s': component '%s' phase '%s' value %r is not a number",
                    path, comp, phase, val,
                )
                return None
            try:
                numeric_value = float(val)
            except (OverflowError, ValueError):
                numeric_value = float("inf")
            if not math.isfinite(numeric_value) or numeric_value < 0:
                logger.warning(
                    "memory_breakdown.json at '%s': component '%s' phase '%s' value %r is not a finite non-negative number",
                    path, comp, phase, val,
                )
                return None
            result[(comp, phase)] = numeric_value

    return result


def update_breakdown_gauge(gauge: Gauge, data: Dict[Tuple[str, str], float]) -> None:
    """Set all 8 (component, phase) samples on the breakdown gauge.

    Args:
        gauge: The llmprof_memory_breakdown_bytes Gauge instance.
        data:  Dict mapping (component, phase) -> float (bytes).
    """
    for (comp, phase), value in data.items():
        gauge.labels(component=comp, phase=phase).set(value)


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle termination signals to initiate clean shutdown."""
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger.info("Received termination signal %s; shutting down...", sig_name)
    _stop_event.set()


def run_collector_loop(
    port: int,
    interval: float,
    registry: Optional[CollectorRegistry] = None,
    gauges: Optional[Dict[str, Gauge]] = None,
    breakdown_file: Optional[str] = None,
) -> None:
    """Start HTTP server and run continuous sampling loop.

    The breakdown gauge is updated only when the breakdown file changes on disk
    (mtime or size), so we avoid redundant JSON parsing every cycle. If the file
    disappears or becomes invalid, the breakdown gauge is cleared (no stale values).

    Args:
        port:            HTTP port for Prometheus scraping.
        interval:        Seconds between sampling cycles.
        registry:        CollectorRegistry to serve (created if None).
        gauges:          Gauge dict (created if None).
        breakdown_file:  Path to memory_breakdown.json; None or '' disables.
    """
    if registry is None or gauges is None:
        registry, gauges = create_gpu_registry()

    logger.info("Starting Prometheus HTTP exporter on 0.0.0.0:%d (endpoint /metrics)", port)
    start_http_server(port, addr="0.0.0.0", registry=registry)
    logger.info("Telemetry loop active (interval: %.1fs). Scrape at http://0.0.0.0:%d/metrics", interval, port)

    # File-watch state: (mtime, size) of the last observed breakdown file
    _bd_stat: Optional[Tuple[float, int]] = None
    _bd_warned_missing: bool = False

    # Announce breakdown file configuration once at startup
    eff_bd_file = breakdown_file if breakdown_file else None
    if eff_bd_file:
        if not os.path.exists(eff_bd_file):
            logger.info(
                "Memory breakdown file not present at startup (will watch): %s", eff_bd_file
            )
            _bd_warned_missing = True
    else:
        logger.info("Memory breakdown export disabled (no breakdown file configured).")

    while not _stop_event.is_set():
        # ── GPU metrics ──────────────────────────────────────────────────────
        try:
            samples = read_gpu_samples()
            update_gauges(samples, gauges)
        except Exception as exc:
            logger.error("Error during GPU sampling iteration: %s", exc)

        # ── Breakdown file watch ─────────────────────────────────────────────
        if eff_bd_file:
            try:
                st = os.stat(eff_bd_file)
                current_stat = (st.st_mtime, st.st_size)
            except FileNotFoundError:
                current_stat = None
            except Exception as exc:
                logger.error("os.stat on breakdown file failed: %s", exc)
                current_stat = None

            if current_stat is None:
                # File is missing
                if _bd_stat is not None:
                    # Was present before: clear series and warn
                    gauges["breakdown"].clear()
                    logger.warning(
                        "Memory breakdown file deleted or inaccessible; cleared breakdown series: %s",
                        eff_bd_file,
                    )
                    _bd_stat = None
                    _bd_warned_missing = True
                elif not _bd_warned_missing:
                    logger.info(
                        "Memory breakdown file not found; breakdown series will publish when available: %s",
                        eff_bd_file,
                    )
                    _bd_warned_missing = True
            elif current_stat != _bd_stat:
                # File appeared or changed — re-read
                data = load_memory_breakdown(eff_bd_file)
                if data is not None:
                    update_breakdown_gauge(gauges["breakdown"], data)
                    _bd_stat = current_stat
                    _bd_warned_missing = False
                    logger.info(
                        "Memory breakdown loaded/refreshed from %s (%d series)", eff_bd_file, len(data)
                    )
                else:
                    # The loader logs the single validation warning; remember this
                    # stat to avoid reparsing and relogging until the file changes.
                    gauges["breakdown"].clear()
                    _bd_stat = current_stat

        _stop_event.wait(interval)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the GPU Prometheus exporter CLI."""
    parser = argparse.ArgumentParser(
        description="M3 GPU Telemetry Prometheus Exporter Daemon (port 9102)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to bind Prometheus HTTP metrics endpoint (default: from config or {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help=f"Sampling interval in seconds (default: from config or {DEFAULT_SAMPLE_INTERVAL_S})",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--memory-breakdown-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to memory_breakdown.json produced by deep_dive mode. "
            "Empty string disables breakdown export. "
            f"(default: from config or {DEFAULT_MEMORY_BREAKDOWN_FILE})"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Take a single telemetry sample, print Prometheus exposition text to stdout, and exit",
    )

    args = parser.parse_args(argv)

    # Configure logging
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    # 1. Load config and apply CLI overrides
    cfg_port, cfg_interval, cfg_breakdown_file, source_info = load_collector_config(args.config)
    effective_port = args.port if args.port is not None else cfg_port
    effective_interval = args.interval if args.interval is not None else cfg_interval

    # CLI flag wins over config; empty string explicitly disables
    if args.memory_breakdown_file is not None:
        effective_breakdown_file: Optional[str] = args.memory_breakdown_file if args.memory_breakdown_file else None
    else:
        effective_breakdown_file = cfg_breakdown_file if cfg_breakdown_file else None

    logger.info(
        "Active settings: port=%d, interval=%.1fs, breakdown_file=%s (source: %s)",
        effective_port,
        effective_interval,
        effective_breakdown_file or "(disabled)",
        source_info,
    )

    # 2. Initialize NVML
    nvml_ok = init_nvml()
    if not nvml_ok:
        logger.error(
            "Exiting with status code 1: NVML is required to export GPU metrics on port %d.",
            effective_port,
        )
        return 1

    # Register termination signals
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    try:
        registry, gauges = create_gpu_registry()

        if args.once:
            # Single snapshot mode: read GPU metrics, load breakdown, print exposition text
            samples = read_gpu_samples()
            update_gauges(samples, gauges)
            if effective_breakdown_file:
                try:
                    os.stat(effective_breakdown_file)
                    bd_data = load_memory_breakdown(effective_breakdown_file)
                    if bd_data is not None:
                        update_breakdown_gauge(gauges["breakdown"], bd_data)
                        logger.info("--once: loaded %d breakdown series", len(bd_data))
                except FileNotFoundError:
                    logger.info("--once: breakdown file not found, skipping: %s", effective_breakdown_file)
            exposition_text = generate_latest(registry).decode("utf-8")
            print(exposition_text, end="")
            return 0

        # Continuous daemon mode
        run_collector_loop(
            port=effective_port,
            interval=effective_interval,
            registry=registry,
            gauges=gauges,
            breakdown_file=effective_breakdown_file,
        )
        return 0

    finally:
        shutdown_nvml()


if __name__ == "__main__":
    sys.exit(main())
