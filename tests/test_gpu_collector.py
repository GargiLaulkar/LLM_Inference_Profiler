"""Unit tests for M3 GPU Prometheus Exporter Collector.

Tests verify:
- Dual GPU metric extraction and gauge updating (utilization, memory bytes, power watts)
- Proper handling of unsupported metrics (e.g. power unsupported on one GPU leaves gauge absent)
- Clean exit code 1 when NVML fails to initialize without starting HTTP server
- Configuration file loading, default fallbacks, and CLI overrides
- Exact Prometheus contract metric names:
    * Without a breakdown file: exactly the 3 GPU metric names have samples
    * With a valid breakdown file: those 3 PLUS llmprof_memory_breakdown_bytes appear in samples
- Memory breakdown: load_memory_breakdown, update_breakdown_gauge, file-watch in run_collector_loop
"""

import json
import re
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from prometheus_client import generate_latest

from llm_profiler.collectors.gpu_collector import (
    DEFAULT_MEMORY_BREAKDOWN_FILE,
    DEFAULT_PORT,
    DEFAULT_SAMPLE_INTERVAL_S,
    create_gpu_registry,
    init_nvml,
    load_collector_config,
    load_memory_breakdown,
    main,
    read_gpu_samples,
    reset_collector_state,
    update_breakdown_gauge,
    update_gauges,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset collector state before each test."""
    reset_collector_state()
    yield
    reset_collector_state()


@pytest.fixture
def mock_two_gpus():
    """Mock pynvml with two NVIDIA GPUs with distinct telemetry profiles."""
    mock = MagicMock()
    mock.nvmlInit = MagicMock()
    mock.nvmlShutdown = MagicMock()
    mock.nvmlDeviceGetCount = MagicMock(return_value=2)

    handle_0 = MagicMock(name="handle_gpu_0")
    handle_1 = MagicMock(name="handle_gpu_1")

    def get_handle(idx: int):
        return handle_0 if idx == 0 else handle_1

    mock.nvmlDeviceGetHandleByIndex = MagicMock(side_effect=get_handle)

    # Utilization: GPU 0 = 45%, GPU 1 = 80%
    rates_0 = MagicMock()
    rates_0.gpu = 45
    rates_1 = MagicMock()
    rates_1.gpu = 80

    def get_util(handle):
        return rates_0 if handle == handle_0 else rates_1

    mock.nvmlDeviceGetUtilizationRates = MagicMock(side_effect=get_util)

    # Memory: GPU 0 = 8 GiB used, GPU 1 = 12 GiB used
    mem_0 = MagicMock()
    mem_0.used = 8589934592  # 8 * 1024^3 bytes
    mem_1 = MagicMock()
    mem_1.used = 12884901888  # 12 * 1024^3 bytes

    def get_mem(handle):
        return mem_0 if handle == handle_0 else mem_1

    mock.nvmlDeviceGetMemoryInfo = MagicMock(side_effect=get_mem)

    # Power: GPU 0 = 65000 mW (65.0 W), GPU 1 = 110500 mW (110.5 W)
    def get_power(handle):
        return 65000 if handle == handle_0 else 110500

    mock.nvmlDeviceGetPowerUsage = MagicMock(side_effect=get_power)

    return mock


def test_two_gpus_metrics_and_conversions(mock_two_gpus):
    """Test two GPUs generate exact gauges, memory in BYTES, and power in WATTS."""
    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_two_gpus):
        samples = read_gpu_samples()

    assert len(samples) == 2

    # GPU 0
    assert samples[0]["gpu"] == "0"
    assert samples[0]["utilization_percent"] == 45.0
    assert samples[0]["memory_used_bytes"] == 8589934592
    assert samples[0]["power_watts"] == 65.0

    # GPU 1
    assert samples[1]["gpu"] == "1"
    assert samples[1]["utilization_percent"] == 80.0
    assert samples[1]["memory_used_bytes"] == 12884901888
    assert samples[1]["power_watts"] == 110.5

    # Test Gauge updating on dedicated registry
    registry, gauges = create_gpu_registry()
    update_gauges(samples, gauges)

    text = generate_latest(registry).decode("utf-8")

    # Verify Prometheus exposition text
    assert 'llmprof_gpu_utilization_percent{gpu="0"} 45.0' in text
    assert 'llmprof_gpu_utilization_percent{gpu="1"} 80.0' in text
    assert 'llmprof_gpu_memory_used_bytes{gpu="0"} 8.589934592e+09' in text
    assert 'llmprof_gpu_memory_used_bytes{gpu="1"} 1.2884901888e+10' in text
    assert 'llmprof_gpu_power_watts{gpu="0"} 65.0' in text
    assert 'llmprof_gpu_power_watts{gpu="1"} 110.5' in text


def test_power_unsupported_on_one_gpu_leaves_gauge_absent(mock_two_gpus):
    """Test unsupported power on one GPU logs once and omits gauge without fake 0."""
    handle_0 = mock_two_gpus.nvmlDeviceGetHandleByIndex(0)

    # GPU 0 supports power (50000 mW -> 50.0 W), GPU 1 throws NotSupported
    def get_power_with_unsupported(handle):
        if handle == handle_0:
            return 50000
        raise Exception("NVMLError_NotSupported: Power draw is unsupported")

    mock_two_gpus.nvmlDeviceGetPowerUsage = MagicMock(side_effect=get_power_with_unsupported)

    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_two_gpus):
        samples = read_gpu_samples()

    assert len(samples) == 2
    assert samples[0]["power_watts"] == 50.0
    assert samples[1]["power_watts"] is None

    # Utilization and memory still collected for both GPUs
    assert samples[1]["utilization_percent"] == 80.0
    assert samples[1]["memory_used_bytes"] == 12884901888

    registry, gauges = create_gpu_registry()
    update_gauges(samples, gauges)

    text = generate_latest(registry).decode("utf-8")

    # GPU 0 power is exported
    assert 'llmprof_gpu_power_watts{gpu="0"} 50.0' in text
    # GPU 1 power is absent (never exported with fake 0)
    assert 'llmprof_gpu_power_watts{gpu="1"}' not in text
    # Other metrics for GPU 1 remain active
    assert 'llmprof_gpu_utilization_percent{gpu="1"} 80.0' in text
    assert 'llmprof_gpu_memory_used_bytes{gpu="1"} 1.2884901888e+10' in text


def test_nvml_init_failure_exits_one_and_skips_server():
    """Test NVML initialization failure prints error, returns 1, and never starts server."""
    failing_pynvml = MagicMock()
    failing_pynvml.nvmlInit = MagicMock(
        side_effect=Exception("NVMLError_LibraryNotFound: NVML Shared Library Not Found")
    )

    with patch("llm_profiler.collectors.gpu_collector.pynvml", failing_pynvml), \
         patch("llm_profiler.collectors.gpu_collector.start_http_server") as mock_http:

        ret = main(["--port", "9102"])
        assert ret == 1
        mock_http.assert_not_called()


def test_config_loading_and_cli_overrides(tmp_path):
    """Test loading from config.yaml, fallback to constants, and CLI overrides."""
    # 1. Custom temporary config.yaml with all three keys
    cfg_file = tmp_path / "custom_config.yaml"
    cfg_content = {
        "gpu_collector": {
            "port": 9555,
            "sample_interval_s": 2.5,
            "memory_breakdown_file": "results/test/mb.json",
        }
    }
    cfg_file.write_text(yaml.safe_dump(cfg_content), encoding="utf-8")

    port, interval, bd_file, src = load_collector_config(str(cfg_file))
    assert port == 9555
    assert interval == 2.5
    assert bd_file == "results/test/mb.json"
    assert str(cfg_file) in src

    # 2. Missing config file falls back to defaults
    missing_file = tmp_path / "nonexistent.yaml"
    def_port, def_interval, def_bd, def_src = load_collector_config(str(missing_file))
    assert def_port == DEFAULT_PORT
    assert def_interval == DEFAULT_SAMPLE_INTERVAL_S
    assert def_bd == DEFAULT_MEMORY_BREAKDOWN_FILE
    assert "default" in def_src.lower()

    # 3. CLI override wins over config file
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit = MagicMock()
    mock_pynvml.nvmlDeviceGetCount = MagicMock(return_value=0)

    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_pynvml), \
         patch("llm_profiler.collectors.gpu_collector.run_collector_loop") as mock_loop:

        # Config specifies 9555/2.5, CLI overrides to 9888/0.25
        ret = main([
            "--config", str(cfg_file),
            "--port", "9888",
            "--interval", "0.25",
        ])
        assert ret == 0
        mock_loop.assert_called_once()
        called_port = mock_loop.call_args[1]["port"]
        called_interval = mock_loop.call_args[1]["interval"]
        assert called_port == 9888
        assert called_interval == 0.25

    # 4. --memory-breakdown-file CLI flag overrides config value
    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_pynvml), \
         patch("llm_profiler.collectors.gpu_collector.run_collector_loop") as mock_loop:

        ret = main([
            "--config", str(cfg_file),
            "--memory-breakdown-file", "/custom/path/mb.json",
        ])
        assert ret == 0
        mock_loop.assert_called_once()
        called_bd = mock_loop.call_args[1]["breakdown_file"]
        assert called_bd == "/custom/path/mb.json"

    # 5. Empty string for --memory-breakdown-file disables breakdown export
    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_pynvml), \
         patch("llm_profiler.collectors.gpu_collector.run_collector_loop") as mock_loop:

        ret = main([
            "--config", str(cfg_file),
            "--memory-breakdown-file", "",
        ])
        assert ret == 0
        mock_loop.assert_called_once()
        called_bd = mock_loop.call_args[1]["breakdown_file"]
        assert called_bd is None

    # 6. Config key is read from a temp config and returned as breakdown_file
    cfg_file2 = tmp_path / "cfg2.yaml"
    cfg_file2.write_text(yaml.safe_dump({
        "gpu_collector": {"memory_breakdown_file": "results/deep_dive/memory_breakdown.json"}
    }), encoding="utf-8")
    _, _, bd2, _ = load_collector_config(str(cfg_file2))
    assert bd2 == "results/deep_dive/memory_breakdown.json"


def test_exact_metric_names_contract(mock_two_gpus):
    """Verify registry sample names: without a breakdown file exactly 3 GPU metrics have samples.

    The breakdown gauge (llmprof_memory_breakdown_bytes) is always REGISTERED on the registry
    (it appears in HELP/TYPE text) but it has ZERO samples when no valid breakdown file is loaded.
    The test checks sample names via registry.collect(), not HELP/TYPE text.
    """
    registry, gauges = create_gpu_registry()
    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_two_gpus):
        samples = read_gpu_samples()
    update_gauges(samples, gauges)
    # Do NOT call update_breakdown_gauge — simulate no breakdown file loaded

    # Collect all metrics that have at least one sample
    metrics_with_samples = set()
    for metric_family in registry.collect():
        if metric_family.samples:
            metrics_with_samples.add(metric_family.name)

    expected_with_samples = {
        "llmprof_gpu_utilization_percent",
        "llmprof_gpu_memory_used_bytes",
        "llmprof_gpu_power_watts",
    }
    assert metrics_with_samples == expected_with_samples, (
        f"Metrics with samples mismatch: {metrics_with_samples} vs {expected_with_samples}"
    )


def test_exact_metric_names_contract_with_breakdown(mock_two_gpus, tmp_path):
    """With a valid breakdown file, the registry has samples for the 3 GPU metrics
    PLUS llmprof_memory_breakdown_bytes (8 additional label-sets)."""
    # Write a minimal valid breakdown file
    bd = {
        "phases": {
            "prefill": {"weights": 1.0, "kv_cache": 2.0, "activations": 3.0, "other": 4.0},
            "decode":  {"weights": 5.0, "kv_cache": 6.0, "activations": 7.0, "other": 8.0},
        }
    }
    bd_path = tmp_path / "memory_breakdown.json"
    bd_path.write_text(json.dumps(bd), encoding="utf-8")

    registry, gauges = create_gpu_registry()
    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_two_gpus):
        samples = read_gpu_samples()
    update_gauges(samples, gauges)

    data = load_memory_breakdown(str(bd_path))
    assert data is not None
    update_breakdown_gauge(gauges["breakdown"], data)

    metrics_with_samples = set()
    for metric_family in registry.collect():
        if metric_family.samples:
            metrics_with_samples.add(metric_family.name)

    expected_with_samples = {
        "llmprof_gpu_utilization_percent",
        "llmprof_gpu_memory_used_bytes",
        "llmprof_gpu_power_watts",
        "llmprof_memory_breakdown_bytes",
    }
    assert metrics_with_samples == expected_with_samples, (
        f"Metrics with samples mismatch: {metrics_with_samples} vs {expected_with_samples}"
    )


def test_once_flag_prints_exposition_and_exits(mock_two_gpus, capsys):
    """Test --once flag executes single sample, outputs exposition text to stdout, and exits 0."""
    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_two_gpus), \
         patch("llm_profiler.collectors.gpu_collector.start_http_server") as mock_server:

        ret = main(["--once"])
        assert ret == 0
        mock_server.assert_not_called()

        captured = capsys.readouterr()
        assert "llmprof_gpu_utilization_percent{gpu=\"0\"} 45.0" in captured.out
        assert "llmprof_gpu_memory_used_bytes{gpu=\"1\"} 1.2884901888e+10" in captured.out
        assert "llmprof_gpu_power_watts{gpu=\"0\"} 65.0" in captured.out
        mock_two_gpus.nvmlShutdown.assert_called_once()


def test_once_flag_includes_breakdown_samples(mock_two_gpus, tmp_path, capsys):
    """The single-snapshot mode prints all valid memory breakdown series."""
    path = _make_bd_file(tmp_path)
    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_two_gpus):
        ret = main(["--once", "--memory-breakdown-file", str(path)])

    assert ret == 0
    output = capsys.readouterr().out
    assert 'llmprof_memory_breakdown_bytes{component="weights",phase="prefill"} 9.88735488e+08' in output
    assert 'llmprof_memory_breakdown_bytes{component="other",phase="decode"} 7.28072192e+08' in output


def test_once_flag_with_missing_breakdown_file_does_not_crash(mock_two_gpus, tmp_path, capsys):
    """A missing optional breakdown file does not prevent the GPU snapshot."""
    missing_path = tmp_path / "missing.json"
    with patch("llm_profiler.collectors.gpu_collector.pynvml", mock_two_gpus):
        ret = main(["--once", "--memory-breakdown-file", str(missing_path)])

    assert ret == 0
    output_lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith('llmprof_gpu_utilization_percent{') for line in output_lines)
    assert not any(line.startswith('llmprof_memory_breakdown_bytes{') for line in output_lines)


# ─── MEMORY BREAKDOWN TESTS ─────────────────────────────────────────────────


VALID_BD = {
    "phases": {
        "prefill": {
            "weights":     988735488.0,   # 942.75 MiB in bytes
            "kv_cache":    1572864.0,     # 1.50 MiB
            "activations": 40089600.0,    # 38.22 MiB
            "other":       691397632.0,   # 659.18 MiB
        },
        "decode": {
            "weights":     988735488.0,
            "kv_cache":    2349056.0,     # 2.24 MiB
            "activations": 2590720.0,     # 2.47 MiB
            "other":       728072192.0,   # 694.18 MiB
        },
    },
    "diagnostics": {"allocated_after_weights": 1234},  # extra key – must be ignored
    "request_id": "test-001",                           # extra key – must be ignored
}


def _make_bd_file(tmp_path, content=None):
    """Write a memory_breakdown.json to tmp_path and return its Path."""
    p = tmp_path / "memory_breakdown.json"
    p.write_text(json.dumps(content if content is not None else VALID_BD), encoding="utf-8")
    return p


def test_load_memory_breakdown_valid_gives_8_series(tmp_path):
    """Valid JSON returns exactly 8 (component, phase) entries with correct byte values."""
    path = _make_bd_file(tmp_path)
    data = load_memory_breakdown(str(path))
    assert data is not None
    assert len(data) == 8

    expected = {
        ("weights", "prefill"): 988735488.0,
        ("kv_cache", "prefill"): 1572864.0,
        ("activations", "prefill"): 40089600.0,
        ("other", "prefill"): 691397632.0,
        ("weights", "decode"): 988735488.0,
        ("kv_cache", "decode"): 2349056.0,
        ("activations", "decode"): 2590720.0,
        ("other", "decode"): 728072192.0,
    }
    assert data == expected

    # All values are non-negative floats
    for (comp, phase), val in data.items():
        assert isinstance(val, float), f"Expected float, got {type(val)} for ({comp}, {phase})"
        assert val >= 0.0


def test_load_memory_breakdown_extra_keys_ignored(tmp_path):
    """Extra keys (nvml_used_bytes, torch_peak_allocated_bytes, diagnostics) are silently ignored."""
    extra = dict(VALID_BD)
    extra["phases"] = dict(VALID_BD["phases"])
    extra["phases"]["prefill"] = dict(VALID_BD["phases"]["prefill"])
    extra["phases"]["prefill"]["nvml_used_bytes"] = 999999
    extra["phases"]["prefill"]["torch_peak_allocated_bytes"] = 888888
    path = _make_bd_file(tmp_path, extra)
    data = load_memory_breakdown(str(path))
    assert data is not None
    assert len(data) == 8   # still exactly 8, extra keys ignored


def test_load_memory_breakdown_missing_file_returns_none():
    """A path to a non-existent file returns None (caller handles startup INFO)."""
    data = load_memory_breakdown("/nonexistent/path/memory_breakdown.json")
    assert data is None


def test_load_memory_breakdown_invalid_json_returns_none(tmp_path):
    """Bad JSON returns None and does not raise."""
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    data = load_memory_breakdown(str(p))
    assert data is None


def test_load_memory_breakdown_missing_component_returns_none(tmp_path):
    """A schema with one component missing returns None (no partial export)."""
    bad = {
        "phases": {
            "prefill": {"weights": 1.0, "kv_cache": 2.0, "activations": 3.0},  # missing "other"
            "decode":  {"weights": 5.0, "kv_cache": 6.0, "activations": 7.0, "other": 8.0},
        }
    }
    path = _make_bd_file(tmp_path, bad)
    data = load_memory_breakdown(str(path))
    assert data is None


def test_load_memory_breakdown_negative_value_returns_none(tmp_path):
    """A negative byte value makes the whole file invalid."""
    bad = dict(VALID_BD)
    bad = {
        "phases": {
            "prefill": {"weights": -1.0, "kv_cache": 2.0, "activations": 3.0, "other": 4.0},
            "decode":  {"weights": 5.0,  "kv_cache": 6.0, "activations": 7.0, "other": 8.0},
        }
    }
    path = _make_bd_file(tmp_path, bad)
    data = load_memory_breakdown(str(path))
    assert data is None


def test_load_memory_breakdown_wrong_type_returns_none(tmp_path):
    """A string value instead of a number makes the file invalid."""
    bad = {
        "phases": {
            "prefill": {"weights": "not-a-number", "kv_cache": 2.0, "activations": 3.0, "other": 4.0},
            "decode":  {"weights": 5.0, "kv_cache": 6.0, "activations": 7.0, "other": 8.0},
        }
    }
    path = _make_bd_file(tmp_path, bad)
    data = load_memory_breakdown(str(path))
    assert data is None


def test_load_memory_breakdown_wrong_top_level_type_returns_none(tmp_path):
    """A valid JSON scalar/container with the wrong root shape is rejected cleanly."""
    path = tmp_path / "wrong-root.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    assert load_memory_breakdown(str(path)) is None


def test_load_memory_breakdown_missing_phase_returns_none(tmp_path):
    """A schema with one phase missing returns None."""
    bad = {
        "phases": {
            "prefill": {"weights": 1.0, "kv_cache": 2.0, "activations": 3.0, "other": 4.0},
            # "decode" missing
        }
    }
    path = _make_bd_file(tmp_path, bad)
    data = load_memory_breakdown(str(path))
    assert data is None


def test_update_breakdown_gauge_sets_8_label_sets(tmp_path):
    """update_breakdown_gauge sets all 8 (component, phase) label combinations."""
    path = _make_bd_file(tmp_path)
    data = load_memory_breakdown(str(path))
    assert data is not None

    registry, gauges = create_gpu_registry()
    update_breakdown_gauge(gauges["breakdown"], data)

    # Verify via registry.collect()
    bd_samples = []
    for mf in registry.collect():
        if mf.name == "llmprof_memory_breakdown_bytes":
            bd_samples = mf.samples
            break

    assert len(bd_samples) == 8, f"Expected 8 samples, got {len(bd_samples)}"

    # Check labels and values for two known series
    sample_map = {(s.labels["component"], s.labels["phase"]): s.value for s in bd_samples}
    assert ("weights", "prefill") in sample_map
    assert sample_map[("weights", "prefill")] == pytest.approx(988735488.0)
    assert sample_map[("kv_cache", "decode")] == pytest.approx(2349056.0)

    # All series have component in the contract set and phase in {prefill, decode}
    valid_components = {"weights", "kv_cache", "activations", "other"}
    valid_phases = {"prefill", "decode"}
    for comp, phase in sample_map:
        assert comp in valid_components, f"Unexpected component label: {comp}"
        assert phase in valid_phases, f"Unexpected phase label: {phase}"


def test_breakdown_missing_file_gives_no_samples():
    """With no breakdown file present, the breakdown gauge has zero samples."""
    registry, gauges = create_gpu_registry()
    # Do not call update_breakdown_gauge

    bd_samples = []
    for mf in registry.collect():
        if mf.name == "llmprof_memory_breakdown_bytes":
            bd_samples = mf.samples
            break

    assert len(bd_samples) == 0, "Breakdown gauge should have no samples without a file"


def test_breakdown_gauge_clear_removes_series(tmp_path):
    """gauge.clear() removes all previously set samples (simulates file-deleted scenario)."""
    path = _make_bd_file(tmp_path)
    data = load_memory_breakdown(str(path))
    registry, gauges = create_gpu_registry()
    update_breakdown_gauge(gauges["breakdown"], data)

    # Confirm 8 samples set
    before = sum(
        len(mf.samples)
        for mf in registry.collect()
        if mf.name == "llmprof_memory_breakdown_bytes"
    )
    assert before == 8

    # Simulate file deleted: clear
    gauges["breakdown"].clear()

    after = sum(
        len(mf.samples)
        for mf in registry.collect()
        if mf.name == "llmprof_memory_breakdown_bytes"
    )
    assert after == 0, "After clear(), breakdown gauge must have zero samples"


def test_breakdown_file_change_detected_on_next_cycle(tmp_path):
    """The real sampling loop reloads the breakdown after a file stat change."""
    v1 = {
        "phases": {
            "prefill": {"weights": 1.0, "kv_cache": 2.0, "activations": 3.0, "other": 4.0},
            "decode":  {"weights": 5.0, "kv_cache": 6.0, "activations": 7.0, "other": 8.0},
        }
    }
    bd_path = _make_bd_file(tmp_path, v1)

    import llm_profiler.collectors.gpu_collector as _gc
    registry, gauges = _gc.create_gpu_registry()

    v2 = {
        "phases": {
            "prefill": {"weights": 999.0, "kv_cache": 2.0, "activations": 3.0, "other": 4.0},
            "decode":  {"weights": 5.0,   "kv_cache": 6.0, "activations": 7.0, "other": 8.0},
        }
    }

    waits = 0

    def change_file_after_first_cycle(_interval):
        nonlocal waits
        waits += 1
        if waits == 1:
            bd_path.write_text(json.dumps(v2), encoding="utf-8")
        else:
            _gc._stop_event.set()

    with patch.object(_gc, "start_http_server"), \
         patch.object(_gc, "read_gpu_samples", return_value=[]), \
         patch.object(_gc._stop_event, "wait", side_effect=change_file_after_first_cycle):
        _gc.run_collector_loop(0, 0.01, registry, gauges, str(bd_path))

    sample = next(
        s
        for mf in registry.collect()
        if mf.name == "llmprof_memory_breakdown_bytes"
        for s in mf.samples
        if s.labels == {"component": "weights", "phase": "prefill"}
    )
    assert sample.value == 999.0


def test_breakdown_file_deleted_clears_series(tmp_path):
    """The real sampling loop clears the gauge after the source file is deleted."""
    bd_path = _make_bd_file(tmp_path)
    import llm_profiler.collectors.gpu_collector as _gc
    registry, gauges = _gc.create_gpu_registry()
    waits = 0

    def delete_file_after_first_cycle(_interval):
        nonlocal waits
        waits += 1
        if waits == 1:
            bd_path.unlink()
        else:
            _gc._stop_event.set()

    with patch.object(_gc, "start_http_server"), \
         patch.object(_gc, "read_gpu_samples", return_value=[]), \
         patch.object(_gc._stop_event, "wait", side_effect=delete_file_after_first_cycle):
        _gc.run_collector_loop(0, 0.01, registry, gauges, str(bd_path))

    assert sum(
        len(mf.samples) for mf in registry.collect()
        if mf.name == "llmprof_memory_breakdown_bytes"
    ) == 0, "Gauge must be empty after file is deleted"


def test_invalid_file_change_clears_exported_series(tmp_path, caplog):
    """An invalid update to a previously valid file clears its exported samples."""
    import llm_profiler.collectors.gpu_collector as _gc

    path = _make_bd_file(tmp_path)
    registry, gauges = _gc.create_gpu_registry()
    waits = 0

    def corrupt_file_after_first_cycle(_interval):
        nonlocal waits
        waits += 1
        if waits == 1:
            path.write_text("{invalid", encoding="utf-8")
        else:
            _gc._stop_event.set()

    with patch.object(_gc, "start_http_server"), \
         patch.object(_gc, "read_gpu_samples", return_value=[]), \
         patch.object(_gc._stop_event, "wait", side_effect=corrupt_file_after_first_cycle):
        _gc.run_collector_loop(0, 0.01, registry, gauges, str(path))

    assert not any(
        mf.samples
        for mf in registry.collect()
        if mf.name == "llmprof_memory_breakdown_bytes"
    )
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1


def test_unchanged_invalid_file_is_not_reloaded_each_cycle(tmp_path, caplog):
    """An invalid file is parsed and warned once until its stat changes."""
    import llm_profiler.collectors.gpu_collector as _gc

    path = tmp_path / "invalid.json"
    path.write_text("{invalid", encoding="utf-8")
    registry, gauges = _gc.create_gpu_registry()
    waits = 0

    def stop_after_three_cycles(_interval):
        nonlocal waits
        waits += 1
        if waits == 3:
            _gc._stop_event.set()

    with patch.object(_gc, "start_http_server"), \
         patch.object(_gc, "read_gpu_samples", return_value=[]), \
         patch.object(_gc, "load_memory_breakdown", wraps=_gc.load_memory_breakdown) as loader, \
         patch.object(_gc._stop_event, "wait", side_effect=stop_after_three_cycles):
        _gc.run_collector_loop(0, 0.01, registry, gauges, str(path))

    assert loader.call_count == 1
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert not any(
        mf.samples
        for mf in registry.collect()
        if mf.name == "llmprof_memory_breakdown_bytes"
    )
