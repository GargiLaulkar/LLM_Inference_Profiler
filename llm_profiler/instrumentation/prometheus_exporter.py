"""Prometheus text-exposition registry and small HTTP exporter for Member 2.

This module intentionally avoids external dependencies. The project does not list
`prometheus_client` in `requirements.txt`, and the M2.5 implementation therefore
uses the standard library to expose valid Prometheus text metrics without
installing or assuming Docker or GPU runtime availability.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterable, List, Optional, Tuple

from .stage_timing import STAGE_NAMES, validate_stage_name


class MetricValidationError(ValueError):
    """Raised when a recorded metric value is negative, non-finite, or otherwise invalid."""

def _validate_ingest_payload(payload: object) -> Dict[str, object]:
    """Validate a request-ingest payload before updating the MetricRegistry."""
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object.")

    required_keys = {
        "request_total_seconds",
        "tokens_in_total",
        "tokens_out_total",
    }
    allowed_keys = required_keys | {"stage_durations"}

    missing = required_keys - set(payload.keys())
    extra = set(payload.keys()) - allowed_keys

    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if extra:
            details.append(f"unexpected keys: {sorted(extra)}")
        raise ValueError("; ".join(details))

    validated: Dict[str, object] = {
        "request_total_seconds": _validate_duration_seconds(
            payload["request_total_seconds"], "request_total_seconds"
        ),
        "tokens_in_total": _validate_token_count(
            payload["tokens_in_total"], "tokens_in_total"
        ),
        "tokens_out_total": _validate_token_count(
            payload["tokens_out_total"], "tokens_out_total"
        ),
    }

    if "stage_durations" in payload:
        stage_durations = payload["stage_durations"]

        if not isinstance(stage_durations, dict):
            raise ValueError("stage_durations must be a JSON object.")

        validated_stages: Dict[str, float] = {}

        for stage_name, duration in stage_durations.items():
            if not isinstance(stage_name, str):
                raise ValueError("stage names must be strings.")

            normalized_stage = validate_stage_name(stage_name)

            if normalized_stage in validated_stages:
                raise ValueError(
                    f"duplicate stage name after normalization: {normalized_stage}"
                )

            validated_stages[normalized_stage] = _validate_duration_seconds(
                duration, f"duration for stage '{normalized_stage}'"
            )

        validated["stage_durations"] = validated_stages

    return validated


def _validate_duration_seconds(value: float, name: str) -> float:
    """Validate that a duration-like metric value is a finite, non-negative float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricValidationError(f"{name} must be a real number in seconds.")

    duration = float(value)
    if not math.isfinite(duration):
        raise MetricValidationError(f"{name} must be finite.")
    if duration < 0:
        raise MetricValidationError(f"{name} cannot be negative.")
    return duration


def _validate_token_count(value: int, name: str) -> int:
    """Validate a token counter value as a finite, non-negative integer."""
    if isinstance(value, bool):
        raise MetricValidationError(f"{name} must be a non-negative integer.")

    if isinstance(value, int):
        token_count = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise MetricValidationError(f"{name} must be finite.")
        if value < 0 or not value.is_integer():
            raise MetricValidationError(f"{name} must be a non-negative integer.")
        token_count = int(value)
    else:
        raise MetricValidationError(f"{name} must be a non-negative integer.")

    if token_count < 0:
        raise MetricValidationError(f"{name} cannot be negative.")
    return token_count


def _format_label_string(labels: Optional[Dict[str, str]]) -> str:
    """Render Prometheus label pairs in a predictable, escaped string."""
    if not labels:
        return ""

    rendered = []
    for key, value in labels.items():
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        rendered.append(f'{key}="{escaped}"')
    return "{" + ",".join(rendered) + "}"


class _HistogramSeries:
    """A small Prometheus histogram-like accumulator for M2 metric recording."""

    _DEFAULT_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.buckets: Dict[str, int] = {"+Inf": 0}
        for bucket in self._DEFAULT_BUCKETS:
            self.buckets[str(bucket)] = 0

    def observe(self, value: float) -> None:
        """Add one observation to the histogram-like series."""
        duration = _validate_duration_seconds(value, "observation")
        self.count += 1
        self.sum += duration

        for bucket in self._DEFAULT_BUCKETS:
            if duration <= float(bucket):
                self.buckets[str(bucket)] += 1

        self.buckets["+Inf"] += 1

    def render(self, metric_name: str, *, labels: Optional[Dict[str, str]] = None) -> List[str]:
        """Render the histogram as Prometheus text format."""
        label_string = _format_label_string(labels)
        lines: List[str] = []

        for bucket in self._DEFAULT_BUCKETS:
            bucket_key = str(bucket)
            lines.append(f'{metric_name}_bucket{label_string}{{le="{bucket_key}"}} {self.buckets[bucket_key]}')

        lines.append(f'{metric_name}_bucket{label_string}{{le="+Inf"}} {self.buckets["+Inf"]}')
        lines.append(f'{metric_name}_sum{label_string} {self.sum}')
        lines.append(f'{metric_name}_count{label_string} {self.count}')
        return lines


class MetricRegistry:
    """In-memory Prometheus metric registry for the M2 exporter."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stage_latencies: Dict[str, _HistogramSeries] = {}
        self._request_total = _HistogramSeries()
        self._tokens_in_total = 0
        self._tokens_out_total = 0

    @property
    def stage_latency_names(self) -> List[str]:
        return list(STAGE_NAMES)

    def record_stage_duration(self, stage_name: str, duration_s: float) -> float:
        """Record one observed duration for a canonical stage label."""
        normalized = validate_stage_name(stage_name)
        value = _validate_duration_seconds(duration_s, f"duration for stage '{normalized}'")

        with self._lock:
            series = self._stage_latencies.setdefault(normalized, _HistogramSeries())
            series.observe(value)
        return value

    def record_request_total_duration(self, duration_s: float) -> float:
        """Record end-to-end request duration separately from individual stage durations."""
        value = _validate_duration_seconds(duration_s, "request_total_seconds")
        with self._lock:
            self._request_total.observe(value)
        return value

    def record_tokens_in_total(self, token_count: int) -> int:
        """Record cumulative prompt tokens."""
        value = _validate_token_count(token_count, "tokens_in_total")
        with self._lock:
            self._tokens_in_total += value
            return self._tokens_in_total

    def record_tokens_out_total(self, token_count: int) -> int:
        """Record cumulative generated output tokens."""
        value = _validate_token_count(token_count, "tokens_out_total")
        with self._lock:
            self._tokens_out_total += value
            return self._tokens_out_total

    def render(self) -> str:
        """Render the registry as Prometheus text exposition."""
        with self._lock:
            lines: List[str] = []

            lines.append("# HELP llmprof_stage_latency_seconds Per-stage request latency in seconds.")
            lines.append("# TYPE llmprof_stage_latency_seconds histogram")
            for stage_name in STAGE_NAMES:
                if stage_name in self._stage_latencies:
                    lines.extend(
                        self._stage_latencies[stage_name].render(
                            "llmprof_stage_latency_seconds",
                            labels={"stage": stage_name},
                        )
                    )

            lines.append("# HELP llmprof_request_total_seconds End-to-end request duration in seconds.")
            lines.append("# TYPE llmprof_request_total_seconds histogram")
            lines.extend(self._request_total.render("llmprof_request_total_seconds"))

            lines.append("# HELP llmprof_tokens_in_total Total observed input prompt tokens.")
            lines.append("# TYPE llmprof_tokens_in_total counter")
            lines.append(f"llmprof_tokens_in_total {self._tokens_in_total}")

            lines.append("# HELP llmprof_tokens_out_total Total observed output tokens.")
            lines.append("# TYPE llmprof_tokens_out_total counter")
            lines.append(f"llmprof_tokens_out_total {self._tokens_out_total}")

            return "\n".join(lines) + "\n"


class PrometheusExporter:
    """Tiny HTTP exporter exposing the registry at /metrics on a configurable host/port."""

    def __init__(
        self,
        registry: Optional[MetricRegistry] = None,
        *,
        host: str = "127.0.0.1",
        port: int = 9101,
    ) -> None:
        self.registry = registry or MetricRegistry()
        self.host = host
        self.port = int(port)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "PrometheusExporter":
        """Start the HTTP server in a background thread."""
        if self._server is not None:
            return self

        handler = self._build_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        """Stop the HTTP server cleanly."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None

    def _build_handler(self):
        registry = self.registry

        class MetricsRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return

                payload = registry.render().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/ingest":
                    self.send_response(404)
                    self.end_headers()
                    return

                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                if not raw_body:
                    self.send_response(400)
                    self.end_headers()
                    return

                try:
                    payload = json.loads(raw_body.decode("utf-8"))
                    validated = _validate_ingest_payload(payload)
                except (TypeError, ValueError, UnicodeDecodeError):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "invalid ingest payload"}')
                    return

                registry.record_request_total_duration(
                    validated["request_total_seconds"]
                )
                registry.record_tokens_in_total(validated["tokens_in_total"])
                registry.record_tokens_out_total(validated["tokens_out_total"])

                for stage_name, duration_s in validated.get("stage_durations", {}).items():
                    registry.record_stage_duration(stage_name, duration_s)



                response_body = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, format: str, *args) -> None:
                return

        return MetricsRequestHandler

    def __enter__(self) -> "PrometheusExporter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()


def create_registry() -> MetricRegistry:
    """Create a new in-memory metric registry for instrumentation code."""
    return MetricRegistry()


def run_exporter(
    *,
    registry: Optional[MetricRegistry] = None,
    host: str = "127.0.0.1",
    port: int = 9101,
) -> PrometheusExporter:
    """Convenience function to construct and start an exporter.

    This is a runnable helper for local tests and debugging, not a background
    service launch. It returns the exporter instance for the caller to manage.
    """
    exporter = PrometheusExporter(registry=registry, host=host, port=port)
    exporter.start()
    return exporter
