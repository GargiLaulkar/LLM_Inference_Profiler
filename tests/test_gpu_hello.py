"""Unit tests for M3 GPU Hello-World and Environment Verification.

Tests verify telemetry extraction, metric conversions (mW to W, bytes to MiB),
graceful degradation when NVML is missing, handling of unsupported metrics,
and JSON / Markdown output formatting using unittest.mock.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_profiler.collectors.gpu_hello import (
    _format_cuda_driver_version,
    check_tooling_and_env,
    collect_gpu_telemetry,
    generate_markdown_report,
    main,
    run_diagnostics,
    sample_live_metrics,
)


@pytest.fixture
def mock_pynvml():
    """Mock pynvml module with standard NVIDIA Tesla T4 telemetry data."""
    mock = MagicMock()
    mock.nvmlInit = MagicMock()
    mock.nvmlShutdown = MagicMock()
    mock.nvmlSystemGetDriverVersion = MagicMock(return_value="535.104.05")
    mock.nvmlSystemGetCudaDriverVersion = MagicMock(return_value=12020)
    mock.nvmlDeviceGetCount = MagicMock(return_value=1)

    handle = MagicMock()
    mock.nvmlDeviceGetHandleByIndex = MagicMock(return_value=handle)
    mock.nvmlDeviceGetName = MagicMock(return_value="NVIDIA Tesla T4")

    # 15360 MiB total, 1024 MiB used
    mem = MagicMock()
    mem.total = 16106127360
    mem.used = 1073741824
    mem.free = 15032385536
    mock.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mem)

    # 42% core util, 15% mem util
    rates = MagicMock()
    rates.gpu = 42
    rates.memory = 15
    mock.nvmlDeviceGetUtilizationRates = MagicMock(return_value=rates)

    # 25000 mW = 25.0 W power draw; 70000 mW = 70.0 W power limit
    mock.nvmlDeviceGetPowerUsage = MagicMock(return_value=25000)
    mock.nvmlDeviceGetPowerManagementLimit = MagicMock(return_value=70000)
    mock.nvmlDeviceGetTemperature = MagicMock(return_value=48)
    mock.NVML_TEMPERATURE_GPU = 0

    return mock


def test_format_cuda_driver_version():
    """Verify numeric CUDA driver versions format properly."""
    assert _format_cuda_driver_version(12020) == "12.2"
    assert _format_cuda_driver_version(11080) == "11.8"
    assert _format_cuda_driver_version("12.4") == "12.4"


def test_normal_gpu_telemetry_conversions(mock_pynvml):
    """Test (a): Normal GPU values, mW to W conversion, bytes to MiB conversion."""
    with patch("llm_profiler.collectors.gpu_hello.pynvml", mock_pynvml):
        ok, err, devices, driver_info = collect_gpu_telemetry()

    assert ok is True
    assert err is None
    assert driver_info["driver_version"] == "535.104.05"
    assert driver_info["cuda_driver_version"] == "12.2"
    assert driver_info["device_count"] == 1

    assert len(devices) == 1
    dev = devices[0]
    assert dev["index"] == 0
    assert dev["name"] == "NVIDIA Tesla T4"

    # Verify bytes to MiB conversion
    assert dev["memory_total_bytes"] == 16106127360
    assert dev["memory_used_bytes"] == 1073741824
    assert dev["memory_total_mib"] == 15360.0
    assert dev["memory_used_mib"] == 1024.0

    # Verify mW to W conversion
    assert dev["power_draw_watts"] == 25.0
    assert dev["power_limit_watts"] == 70.0

    # Core & Memory utilization
    assert dev["gpu_utilization_percent"] == 42.0
    assert dev["memory_utilization_percent"] == 15.0
    assert dev["temperature_celsius"] == 48

    # Ensure nvmlShutdown was invoked
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_nvml_library_not_found():
    """Test (b): NVML library not found, script reports it and does not crash."""
    mock_failing_pynvml = MagicMock()
    mock_failing_pynvml.nvmlInit = MagicMock(
        side_effect=Exception("NVMLError_LibraryNotFound: NVML Shared Library Not Found")
    )

    with patch("llm_profiler.collectors.gpu_hello.pynvml", mock_failing_pynvml):
        ok, err, devices, driver_info = collect_gpu_telemetry()

        assert ok is False
        assert "NVMLError_LibraryNotFound" in err
        assert devices == []

        # Ensure main() executes with exit code 0 despite missing NVML
        exit_code = main(["--samples", "1", "--interval", "0.0"])
        assert exit_code == 0


def test_power_unsupported_still_collects_other_metrics(mock_pynvml):
    """Test (c): Power unsupported on GPU still prints other metrics without failing."""
    mock_pynvml.nvmlDeviceGetPowerUsage = MagicMock(
        side_effect=Exception("NVMLError_NotSupported: Not Supported")
    )
    mock_pynvml.nvmlDeviceGetPowerManagementLimit = MagicMock(
        side_effect=Exception("NVMLError_NotSupported: Not Supported")
    )
    mock_pynvml.nvmlDeviceGetEnforcedPowerLimit = MagicMock(
        side_effect=Exception("NVMLError_NotSupported: Not Supported")
    )

    with patch("llm_profiler.collectors.gpu_hello.pynvml", mock_pynvml):
        ok, err, devices, driver_info = collect_gpu_telemetry()

    assert ok is True
    assert err is None
    assert len(devices) == 1
    dev = devices[0]

    # Power metrics are marked unsupported
    assert dev["power_draw_watts"] == "unsupported"
    assert dev["power_limit_watts"] == "unsupported"

    # All other metrics remain intact and valid
    assert dev["gpu_utilization_percent"] == 42.0
    assert dev["memory_total_mib"] == 15360.0
    assert dev["temperature_celsius"] == 48


def test_json_output_is_valid_json(mock_pynvml, capsys):
    """Test (d): --json output produces valid parseable JSON."""
    with patch("llm_profiler.collectors.gpu_hello.pynvml", mock_pynvml):
        exit_code = main(["--json", "--samples", "2", "--interval", "0.0"])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "timestamp" in payload
    assert "system" in payload
    assert "nvml" in payload
    assert payload["nvml"]["available"] is True
    assert len(payload["nvml"]["devices"]) == 1
    assert payload["nvml"]["devices"][0]["power_draw_watts"] == 25.0
    assert len(payload["samples"]) == 2


def test_system_tooling_checks():
    """Verify system check dictionary includes expected tools and platforms."""
    info = check_tooling_and_env()
    assert "os" in info
    assert "python" in info
    assert "nvidia-smi" in info
    assert "nsys" in info
    assert "perf_event_paranoid" in info
    assert "torch" in info
    assert "docker" in info


def test_live_sampling_metrics(mock_pynvml):
    """Verify live sampling records multiple time snapshots."""
    with patch("llm_profiler.collectors.gpu_hello.pynvml", mock_pynvml):
        samples = sample_live_metrics(samples=3, interval=0.0)

    assert len(samples) == 3
    for idx, s in enumerate(samples, start=1):
        assert s["sample_index"] == idx
        assert len(s["gpus"]) == 1
        assert s["gpus"][0]["power_watts"] == 25.0
        assert s["gpus"][0]["memory_used_mib"] == 1024.0


def test_markdown_report_generation(mock_pynvml, tmp_path):
    """Verify markdown report generation and --out option."""
    out_file = tmp_path / "test_report.md"
    with patch("llm_profiler.collectors.gpu_hello.pynvml", mock_pynvml):
        exit_code = main(["--out", str(out_file), "--samples", "1", "--interval", "0.0"])

    assert exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# M3 GPU Environment & Telemetry Verification Report" in content
    assert "NVIDIA Tesla T4" in content
    assert "llmprof_gpu_power_watts" in content
