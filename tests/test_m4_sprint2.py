"""Focused tests for M4 Sprint 2 readers, analysis, CLI, and exporter."""

import csv
import json
import os
import sys
import urllib.parse

import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from llm_profiler.analyzer import cli as analyzer_cli
from llm_profiler.analyzer.bottleneck_analyzer import BottleneckAnalyzer
from llm_profiler.analyzer.cost_exporter import (
    LiveCostExporter,
    load_exporter_settings,
    main as cost_exporter_main,
)
from llm_profiler.analyzer.cost_model import CostModel
from llm_profiler.analyzer.readers import (
    PrometheusClient,
    PrometheusUnreachableError,
    join_by_request_id,
    read_memory_breakdown_json,
    read_request_csv,
    read_trace_jsonl,
)


def _write_request_csv(path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "request_id", "send_ts", "end_ts", "total_latency_s",
            "prompt_tokens", "output_tokens", "status", "target_rps", "scenario_tag",
        ])
        writer.writerow(["r1", "1", "2", "1", "10", "5", "ok", "1", "test"])
        writer.writerow(["r2", "invalid", "2", "1", "10", "5", "ok", "1", "test"])


def test_request_csv_skips_malformed_rows_and_trace_validation(tmp_path):
    csv_path = tmp_path / "requests.csv"
    _write_request_csv(csv_path)
    requests = read_request_csv(csv_path)
    assert [row["request_id"] for row in requests] == ["r1"]

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        "\n".join([
            json.dumps({
                "request_id": "r1", "stage": "decode", "start_ts": 1,
                "end_ts": 1.5, "duration_s": 0.5,
            }),
            json.dumps({
                "request_id": "r1", "stage": "decode", "start_ts": 1.5,
                "end_ts": 2, "duration_s": 0.5,
            }),
            json.dumps({
                "request_id": "r1", "stage": "made_up", "start_ts": 1,
                "end_ts": 2, "duration_s": 1,
            }),
            "{invalid json",
            json.dumps({
                "request_id": "r1", "stage": "prefill", "start_ts": 2,
                "end_ts": 1, "duration_s": 1,
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    traces, stats = read_trace_jsonl(trace_path)
    joined = join_by_request_id(requests, traces)
    assert len(traces) == 2
    assert stats == {
        "total_lines": 5,
        "valid_traces": 2,
        "skipped_malformed": 2,
        "skipped_unknown_stages": 1,
    }
    assert joined["r1"]["stages"] == {"decode": 1.0}


def test_memory_breakdown_optional_and_schema_validation(tmp_path):
    assert read_memory_breakdown_json(None) == (None, "memory evidence unavailable")
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({
        "phases": {
            phase: {component: 1024 for component in ("weights", "kv_cache", "activations", "other")}
            for phase in ("prefill", "decode")
        }
    }), encoding="utf-8")
    data, status = read_memory_breakdown_json(path)
    assert status == "valid"
    assert data["phases"]["decode"]["kv_cache"] == 1024

    path.write_text('{"phases":{"prefill":{"weights":-1}}}', encoding="utf-8")
    data, status = read_memory_breakdown_json(path)
    assert data is None
    assert "schema mismatch" in status


class _MockResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def getcode(self):
        return 200

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_prometheus_client_uses_http_api_and_parses_contract_metrics(monkeypatch):
    def mock_urlopen(request, timeout):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)["query"][0]
        if "histogram_quantile(0.50" in query:
            result = [{"metric": {"stage": "decode"}, "value": [0, "0.4"]}]
        elif "histogram_quantile(0.95" in query:
            result = [{"metric": {"stage": "decode"}, "value": [0, "0.8"]}]
        elif "stage_latency_seconds_sum" in query:
            result = [{"metric": {"stage": "decode"}, "value": [0, "0.5"]}]
        elif "request_total_seconds_sum" in query:
            result = [{"metric": {}, "value": [0, "0.6"]}]
        elif "tokens_in_total" in query:
            result = [{"metric": {}, "value": [0, "12"]}]
        elif "tokens_out_total" in query:
            result = [{"metric": {}, "value": [0, "8"]}]
        elif "gpu_utilization_percent" in query:
            result = [{"metric": {"gpu": "0"}, "value": [0, "31"]}]
        elif "gpu_memory_used_bytes" in query:
            result = [{"metric": {"gpu": "0"}, "value": [0, "2048"]}]
        elif "gpu_power_watts" in query:
            result = [{"metric": {"gpu": "0"}, "value": [0, "42"]}]
        else:
            result = [{
                "metric": {"phase": "decode", "component": "kv_cache"},
                "value": [0, "1024"],
            }]
        return _MockResponse({"status": "success", "data": {"result": result}})

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    client = PrometheusClient(base_url="http://prometheus.local")
    assert client.get_stage_latency_quantiles() == {"decode": {"p50": 0.4, "p95": 0.8}}
    assert client.get_stage_latency_means() == {"decode": 0.5}
    assert client.get_stage_latency_summary() == {
        "decode": {"p50": 0.4, "p95": 0.8, "mean": 0.5},
        "total": {"mean": 0.6},
    }
    assert client.get_token_rates() == {"tokens_in_rate": 12.0, "tokens_out_rate": 8.0}
    assert client.get_gpu_gauges() == {
        "gpu_utilization_percent": 31.0,
        "gpu_memory_used_bytes": 2048.0,
        "gpu_power_watts": 42.0,
    }
    assert client.get_memory_breakdown_gauges() == {"decode": {"kv_cache": 1024.0}}


def test_prometheus_client_reports_unreachable(monkeypatch):
    def unreachable(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", unreachable)
    with pytest.raises(PrometheusUnreachableError, match="is unreachable"):
        PrometheusClient(base_url="http://127.0.0.1:1").query_instant("up")


def _analysis_inputs(tmp_path):
    csv_path = tmp_path / "requests.csv"
    csv_path.write_text(
        "request_id,send_ts,end_ts,total_latency_s,prompt_tokens,output_tokens,status,target_rps,scenario_tag\n"
        "r1,1,2,1,10,5,ok,1,test\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        "\n".join([
            json.dumps({
                "request_id": "r1", "stage": "queue_wait", "start_ts": 1,
                "end_ts": 1.62, "duration_s": 0.62,
            }),
            json.dumps({
                "request_id": "r1", "stage": "prefill", "start_ts": 1.62,
                "end_ts": 2, "duration_s": 0.38,
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    return csv_path, trace_path


def test_analyzer_stage_gpu_and_memory_evidence_modes(tmp_path):
    csv_path, trace_path = _analysis_inputs(tmp_path)
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps({
        "phases": {
            phase: {component: 1024 * 1024 for component in ("weights", "kv_cache", "activations", "other")}
            for phase in ("prefill", "decode")
        }
    }), encoding="utf-8")
    analyzer = BottleneckAnalyzer(gpu_price_per_hour_usd=1)

    stage_only = analyzer.analyze(csv_path, trace_path)
    stage_gpu = analyzer.analyze(
        csv_path, trace_path,
        gpu_metrics={"gpu_utilization_percent": 31.0, "gpu_power_watts": 42.0},
    )
    stage_gpu_memory = analyzer.analyze(
        csv_path, trace_path, memory_breakdown_path=memory_path,
        gpu_metrics={"gpu_utilization_percent": 31.0, "gpu_power_watts": 42.0},
    )

    assert stage_only["top_bottlenecks"][0]["stage"] == "queue_wait"
    assert "queue_wait is 62.0% of mean latency" in stage_only["top_bottlenecks"][0]["evidence"]
    assert "memory evidence unavailable" in stage_only["top_bottlenecks"][0]["evidence"]
    assert "GPU utilization avg 31.0%" in stage_gpu["top_bottlenecks"][0]["evidence"]
    assert "memory: decode" not in stage_gpu["top_bottlenecks"][0]["evidence"]
    assert "memory: prefill" in stage_gpu_memory["top_bottlenecks"][1]["evidence"]


def test_prometheus_stage_summary_analyzes_histogram_means(tmp_path):
    csv_path, trace_path = _analysis_inputs(tmp_path)
    result = BottleneckAnalyzer(gpu_price_per_hour_usd=1).analyze(
        csv_path,
        traces_path=None,
        stage_metrics={
            "queue_wait": {"mean": 0.62, "p50": 0.60, "p95": 0.7},
            "prefill": {"mean": 0.38, "p50": 0.35, "p95": 0.45},
            "total": {"mean": 1.0},
        },
        gpu_metrics={"gpu_utilization_percent": 31},
    )
    assert result["top_bottlenecks"][0]["stage"] == "queue_wait"
    assert result["explained_share"] == 1.0
    assert "Prometheus histogram" in result["top_bottlenecks"][0]["evidence"]


def test_prometheus_absent_stage_metrics_falls_back_to_trace(tmp_path, monkeypatch, capsys):
    csv_path, trace_path = _analysis_inputs(tmp_path)
    monkeypatch.setattr(PrometheusClient, "get_stage_latency_summary", lambda self: {})
    monkeypatch.setattr(
        PrometheusClient, "get_gpu_gauges",
        lambda self: {"gpu_utilization_percent": 31.0, "gpu_power_watts": None, "gpu_memory_used_bytes": None},
    )
    monkeypatch.setattr(PrometheusClient, "get_memory_breakdown_gauges", lambda self: {})
    args = analyzer_cli.build_parser().parse_args([
        "--csv", str(csv_path), "--trace", str(trace_path), "--source", "prometheus",
        "--prometheus-url", "http://prometheus.local",
    ])
    result = analyzer_cli.run_analysis(args)
    assert result["top_bottlenecks"][0]["stage"] == "queue_wait"
    assert "per-stage metrics not available yet" in capsys.readouterr().err


def test_cli_prometheus_down_uses_trace_or_reports_clear_error(tmp_path, monkeypatch):
    csv_path, trace_path = _analysis_inputs(tmp_path)

    def unavailable(self):
        raise PrometheusUnreachableError("Prometheus is down")

    monkeypatch.setattr(PrometheusClient, "get_stage_latency_summary", unavailable)
    args = analyzer_cli.build_parser().parse_args([
        "--csv", str(csv_path), "--trace", str(trace_path), "--source", "prometheus",
        "--prometheus-url", "http://prometheus.local",
    ])
    assert analyzer_cli.run_analysis(args)["top_bottlenecks"][0]["stage"] == "queue_wait"

    args.trace = None
    with pytest.raises(PrometheusUnreachableError, match="no valid --trace fallback"):
        analyzer_cli.run_analysis(args)


def test_cli_runs_every_synthetic_scenario_with_contract_json(tmp_path, monkeypatch):
    from tests.synthetic.make_synthetic_data import generate_all_synthetic_scenarios

    manifest = generate_all_synthetic_scenarios(base_dir=str(tmp_path / "scenarios"), num_requests=5)
    monkeypatch.setattr(PrometheusClient, "get_gpu_gauges", lambda self: {})
    monkeypatch.setattr(PrometheusClient, "get_memory_breakdown_gauges", lambda self: {})
    for item in manifest.values():
        args = analyzer_cli.build_parser().parse_args([
            "--csv", item["csv_path"],
            "--trace", item["traces_path"],
            "--source", "trace",
            "--prometheus-url", "http://prometheus.local",
        ])
        result = analyzer_cli.run_analysis(args)
        contract_result = json.loads(json.dumps(result))
        assert contract_result["top_bottlenecks"][0]["stage"] == item["ground_truth"]["expected_top_stage"]
        assert isinstance(contract_result["explained_share"], float)
        assert isinstance(contract_result["cost_per_million_tokens_usd"], float)
        assert ("warning" in contract_result) == item["ground_truth"]["expected_warning"]


def test_live_cost_exporter_counter_window_formula_and_format():
    class CounterClient:
        def __init__(self):
            self.values = iter(["100", "200"])

        def query_instant(self, query):
            assert query == "sum(llmprof_tokens_out_total)"
            return [{"value": [0, next(self.values)]}]

    exporter = LiveCostExporter(
        client=CounterClient(),
        cost_model=CostModel(gpu_price_per_hour_usd=1),
        time_share=0.5,
        host="127.0.0.1",
        port=0,
    )
    assert exporter.refresh(now=10) == 0.0
    exposition = exporter.render_metrics(now=3610)
    assert "# TYPE llmprof_cost_per_million_tokens_usd gauge" in exposition
    assert "llmprof_cost_per_million_tokens_usd 5000.000000" in exposition


def test_live_exporter_settings_read_environment_and_cli_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "cost_exporter:\n  host: 127.0.0.1\n  port: 9103\n  time_share: 0.75\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("M4_TIME_SHARE", "0.25")
    settings = load_exporter_settings(str(config_path))
    assert settings["port"] == 9103
    assert settings["time_share"] == 0.25
    overridden = load_exporter_settings(
        str(config_path), port_override=9200, time_share_override=0.5,
        host_override="0.0.0.0",
    )
    assert overridden["port"] == 9200
    assert overridden["time_share"] == 0.5
    assert overridden["host"] == "0.0.0.0"


def test_cost_exporter_missing_time_share_logs_actionable_message(monkeypatch, capsys):
    monkeypatch.delenv("M4_TIME_SHARE", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        cost_exporter_main([
            "--host", "127.0.0.1",
            "--port", "9103",
            "--prometheus-url", "http://127.0.0.1:9090",
        ])
    assert exc_info.value.code == 2
    error_output = capsys.readouterr().err
    assert "M4_TIME_SHARE is unset" in error_output
    assert "pass --time-share" in error_output


def test_cli_main_prints_contract_json_and_writes_requested_file(tmp_path, monkeypatch, capsys):
    csv_path, trace_path = _analysis_inputs(tmp_path)
    monkeypatch.setattr(PrometheusClient, "get_gpu_gauges", lambda self: {})
    monkeypatch.setattr(PrometheusClient, "get_memory_breakdown_gauges", lambda self: {})
    out_path = tmp_path / "nested" / "result.json"
    analyzer_cli.main([
        "--csv", str(csv_path), "--trace", str(trace_path), "--source", "trace",
        "--prometheus-url", "http://prometheus.local", "--out", str(out_path),
    ])
    stdout = capsys.readouterr().out
    printed = json.loads(stdout)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert printed == saved
    assert set(printed) >= {
        "top_bottlenecks", "cost_per_million_tokens_usd", "explained_share",
    }


def test_cost_model_requires_config_price_when_no_override(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cost configuration file not found"):
        CostModel(config_path=str(tmp_path / "missing.yaml"))
