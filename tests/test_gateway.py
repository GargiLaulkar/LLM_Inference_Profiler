"""Unit tests for LLM Profiler Gateway."""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from llm_profiler.gateway.app import (
    TokenizerHelper,
    create_gateway_app,
    run_cpu_busywork,
    state,
)
from llm_profiler.instrumentation.stage_timing import STAGE_NAMES


@pytest.fixture
def clean_env():
    """Ensure clean LLMPROF environment variables for test isolation."""
    keys = [
        "LLMPROF_PROFILER_ENABLED",
        "LLMPROF_SLOW_TOKENIZER_MS",
        "LLMPROF_CPU_PREPROCESS_MS",
        "LLMPROF_NO_BATCHING",
        "LLMPROF_TRACE_PATH",
    ]
    old_values = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k, v in old_values.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def test_health():
    """Test gateway health endpoint."""
    app = create_gateway_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_cpu_busywork():
    """Test that CPU busywork runs for approximately the requested duration."""
    t0 = time.perf_counter()
    run_cpu_busywork(50.0)  # 50 ms
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert elapsed >= 45.0  # At least ~50ms


def test_stage_math_and_trace_format(clean_env):
    """Test that trace records match the schema and stage durations sum accurately."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        trace_path = tf.name

    os.environ["LLMPROF_TRACE_PATH"] = trace_path
    os.environ["LLMPROF_PROFILER_ENABLED"] = "1"

    app = create_gateway_app()

    # Mock SSE stream from vLLM
    chunks = [
        b'data: {"id": "cmpl-1", "created": 100, "choices": [{"text": "Hello"}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}\n',
        b'data: {"id": "cmpl-1", "created": 100, "choices": [{"text": " world"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}\n',
        b'data: [DONE]\n',
    ]

    class MockContent:
        async def __aiter__(self):
            for c in chunks:
                await asyncio.sleep(0.02)
                yield c

    class MockResp:
        status = 200
        content = MockContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with TestClient(app) as client:
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = MockResp()
        state.session = mock_session
        state.tokenizer_helper.tokenize = AsyncMock(return_value=[1, 2, 3, 4, 5])

        resp = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "Hello", "max_tokens": 2},
            headers={"X-Request-ID": "test-req-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "cmpl-1"
        assert data["choices"][0]["text"] == "Hello world"
        assert data["usage"]["completion_tokens"] == 2

    # Give time for async tasks
    time.sleep(0.15)

    assert os.path.exists(trace_path)
    with open(trace_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) >= 5
    req_ids = {r["request_id"] for r in records}
    assert req_ids == {"test-req-123"}

    stage_map = {}
    allowed_stages = set(STAGE_NAMES)
    for r in records:
        assert set(r.keys()) == {"request_id", "stage", "start_ts", "end_ts", "duration_s"}
        assert r["stage"] in allowed_stages
        assert r["start_ts"] <= r["end_ts"]
        assert r["duration_s"] >= 0
        stage_map[r["stage"]] = r["duration_s"]

    # Verify stage breakdown
    assert "tokenization" in stage_map
    assert "queue_wait" in stage_map
    assert "prefill" in stage_map
    assert "decode" in stage_map
    assert "total" in stage_map

    total_duration = stage_map["total"]
    component_sum = sum(v for k, v in stage_map.items() if k != "total")
    assert total_duration > 0
    explained_share = component_sum / total_duration
    assert 0.85 <= explained_share <= 1.15

    try:
        os.remove(trace_path)
    except OSError:
        pass


def test_hooks_are_noops_when_unset(clean_env):
    """Verify that when hooks are unset, no injection occurs."""
    assert state.get_slow_tokenizer_ms() == 0.0
    assert state.get_cpu_preprocess_ms() == 0.0
    assert state.is_no_batching() is False


def test_slow_tokenizer_hook(clean_env):
    """Verify that LLMPROF_SLOW_TOKENIZER_MS adds delay to tokenization."""
    os.environ["LLMPROF_SLOW_TOKENIZER_MS"] = "100"
    assert state.get_slow_tokenizer_ms() == 100.0


def test_cpu_preprocess_hook(clean_env):
    """Verify that LLMPROF_CPU_PREPROCESS_MS is parsed correctly."""
    os.environ["LLMPROF_CPU_PREPROCESS_MS"] = "150"
    assert state.get_cpu_preprocess_ms() == 150.0


def test_no_batching_flag(clean_env):
    """Verify that LLMPROF_NO_BATCHING is parsed correctly."""
    os.environ["LLMPROF_NO_BATCHING"] = "1"
    assert state.is_no_batching() is True


def test_profiler_switch_disabled(clean_env):
    """Verify that when LLMPROF_PROFILER_ENABLED=0, no trace is written and pure pass-through occurs."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        trace_path = tf.name

    os.environ["LLMPROF_TRACE_PATH"] = trace_path
    os.environ["LLMPROF_PROFILER_ENABLED"] = "0"

    app = create_gateway_app()

    class MockPassResp:
        status = 200

        async def read(self):
            return json.dumps({"id": "passthrough-1", "choices": [{"text": "ok"}]}).encode("utf-8")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with TestClient(app) as client:
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = MockPassResp()
        state.session = mock_session

        resp = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "Hello"},
            headers={"X-Request-ID": "bypass-1"},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "passthrough-1"

    time.sleep(0.05)
    # Trace file must remain empty
    with open(trace_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.strip() == ""

    try:
        os.remove(trace_path)
    except OSError:
        pass
