"""Prometheus exporter for live cost per million generated tokens."""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .cost_model import CostModel
from .readers import PrometheusClient, PrometheusUnreachableError


def load_exporter_settings(
    config_path: Optional[str] = None,
    port_override: Optional[int] = None,
    time_share_override: Optional[float] = None,
    host_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Load settings, preferring CLI, then environment, then config values."""
    candidates = (
        [Path(config_path)]
        if config_path
        else [Path(__file__).resolve().parents[2] / "config.yaml"]
    )

    config: Dict[str, Any] = {}
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
            if not isinstance(config, dict):
                raise ValueError(f"Configuration in {candidate} must be a YAML mapping")
            break
    else:
        if config_path:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        raise FileNotFoundError("config.yaml not found; pass --config")

    section = config.get("cost_exporter", {})
    if not isinstance(section, dict):
        raise ValueError("config.yaml cost_exporter section must be a mapping")

    port = port_override if port_override is not None else section.get("port")
    if port is None:
        raise ValueError("Set cost_exporter.port in config.yaml or pass --port")
    if isinstance(port, bool):
        raise ValueError("Exporter port must be an integer between 0 and 65535")
    port = int(port)
    if not 0 <= port <= 65535:
        raise ValueError("Exporter port must be between 0 and 65535")

    time_share = time_share_override
    if time_share is None:
        time_share = os.environ.get("M4_TIME_SHARE") or section.get("time_share")
    if time_share is None:
        raise ValueError(
            "M4_TIME_SHARE is unset; set the environment variable or pass --time-share "
            "to choose the workload billing share (0.0-1.0)"
        )
    if isinstance(time_share, bool):
        raise ValueError("time_share must be a finite value between 0.0 and 1.0")
    time_share = float(time_share)
    if not math.isfinite(time_share) or not 0.0 <= time_share <= 1.0:
        raise ValueError("time_share must be a finite value between 0.0 and 1.0")
    host = host_override if host_override is not None else section.get("host")
    if not host:
        raise ValueError("Set cost_exporter.host in config.yaml or pass --host")

    return {
        "config": config,
        "port": port,
        "time_share": time_share,
        "host": str(host),
    }


class LiveCostExporter:
    """Expose cost computed from observed output-token counter growth and elapsed time."""

    def __init__(
        self,
        client: PrometheusClient,
        cost_model: CostModel,
        time_share: float,
        host: str,
        port: int,
    ) -> None:
        if not math.isfinite(time_share) or not 0.0 <= time_share <= 1.0:
            raise ValueError("time_share must be a finite value between 0.0 and 1.0")
        self.client = client
        self.cost_model = cost_model
        self.time_share = time_share
        self.host = host
        self.port = int(port)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._baseline_tokens: Optional[float] = None
        self._started_at: Optional[float] = None
        self._cost = 0.0

    def refresh(self, now: Optional[float] = None) -> float:
        """Update the gauge using the counter delta and elapsed wall-clock window."""
        results = self.client.query_instant("sum(llmprof_tokens_out_total)")
        if not results:
            raise PrometheusUnreachableError(
                "Prometheus returned no llmprof_tokens_out_total counter data"
            )
        current_tokens = PrometheusClient._value(results[0])
        if current_tokens is None or current_tokens < 0:
            raise PrometheusUnreachableError(
                "Prometheus returned an invalid llmprof_tokens_out_total counter value"
            )
        timestamp = time.monotonic() if now is None else float(now)
        if not math.isfinite(timestamp):
            raise ValueError("wall-clock timestamp must be finite")

        with self._lock:
            if self._baseline_tokens is None or self._started_at is None or current_tokens < self._baseline_tokens:
                self._baseline_tokens = current_tokens
                self._started_at = timestamp
                self._cost = 0.0
                return self._cost

            elapsed = max(0.0, timestamp - self._started_at)
            tokens_produced = current_tokens - self._baseline_tokens
            gpu_hours = (elapsed / 3600.0) * self.time_share
            if tokens_produced == 0:
                self._cost = 0.0
            else:
                self._cost = (
                    self.cost_model.gpu_price_per_hour_usd
                    * gpu_hours
                    / tokens_produced
                    * 1_000_000.0
                )
            return self._cost

    def render_metrics(self, now: Optional[float] = None) -> str:
        """Refresh and render the cost gauge in Prometheus text exposition format."""
        value = self.refresh(now=now)
        return self.cost_model.render_prometheus_metrics(value)

    def start(self) -> "LiveCostExporter":
        if self._server is not None:
            return self
        exporter = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/metrics":
                    self.send_error(404)
                    return
                try:
                    body = exporter.render_metrics().encode("utf-8")
                except (PrometheusUnreachableError, ValueError) as exc:
                    self.send_error(503, str(exc))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        if self._server is None:
            self.start()
        if self._thread is None:
            raise RuntimeError("Cost exporter server failed to start")
        self._thread.join()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None

    def __enter__(self) -> "LiveCostExporter":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export live inference cost per million generated tokens."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--prometheus-url", default=None, help="Prometheus HTTP API base URL")
    parser.add_argument("--gpu-price", type=float, default=None, help="Override GPU price per hour")
    parser.add_argument("--host", default=None, help="Exporter bind address")
    parser.add_argument("--port", type=int, default=None, help="Exporter port (overrides config)")
    parser.add_argument(
        "--time-share", type=float, default=None,
        help="Fraction of the wall-clock window billed to this workload (0.0-1.0); "
        "overrides M4_TIME_SHARE",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_exporter_settings(
            config_path=args.config,
            port_override=args.port,
            time_share_override=args.time_share,
            host_override=args.host,
        )
        client = PrometheusClient(base_url=args.prometheus_url, config_path=args.config)
        if not client.base_url:
            raise ValueError(
                "Prometheus URL is not configured; pass --prometheus-url or set prometheus_url in config.yaml"
            )
        model = CostModel(
            gpu_price_per_hour_usd=args.gpu_price,
            config_path=args.config,
        )
        exporter = LiveCostExporter(
            client=client,
            cost_model=model,
            time_share=settings["time_share"],
            host=settings["host"],
            port=settings["port"],
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"[INFO] Serving cost metric at http://{exporter.host}:{exporter.port}/metrics", file=sys.stderr)
    try:
        exporter.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        exporter.shutdown()


if __name__ == "__main__":
    main()
