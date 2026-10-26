"""Tests for llm_profiler/collectors/deep_dive.py (Task 4, M3).

Design goals
------------
* Most tests require NO torch — they run on the Windows dev laptop where torch is not installed.
* Tests that actually need torch use pytest.importorskip("torch").
* Stub tokenizer and stub model are plain Python objects (no torch dependency).
* The stub model is used only in the run_request test (which skips without torch).

Contract checked
----------------
* build_prompt     → returns exactly target_tokens ids
* write_stage_jsonl → produces valid JSON lines with exactly the 5 required fields
* summarize         → correct total_s, mean, p95 for known inputs
* stage()           → asserts on unknown stage name
* CONTRACT_STAGES   → contains exactly the 4 required names
* run_request       → 4 stage records, contract order, correct decode-token count
* --help            → exits 0 without torch
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, List

import pytest

# ── Import the module under test ──────────────────────────────────────────────
# deep_dive.py lives in llm_profiler/collectors/; make sure it is importable
# from the project root (where pytest is run).
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_profiler.collectors.deep_dive import (
    CONTRACT_STAGES,
    build_prompt,
    summarize,
    write_stage_jsonl,
)

# ─── Stubs ───────────────────────────────────────────────────────────────────


class StubTokenizer:
    """Minimal tokenizer stub — word-count proxy, no torch dependency."""

    pad_token_id: int = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Each word maps to a unique token id (its index).
        words = text.split()
        return list(range(len(words)))

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


# ─── Constants ────────────────────────────────────────────────────────────────


class TestContractStages:
    """Verify the CONTRACT_STAGES list."""

    def test_exact_four_stages(self) -> None:
        assert len(CONTRACT_STAGES) == 4

    def test_stage_names(self) -> None:
        assert CONTRACT_STAGES == [
            "tokenization",
            "prefill",
            "decode",
            "detokenization",
        ]


# ─── build_prompt ─────────────────────────────────────────────────────────────


class TestBuildPrompt:
    """Tests for build_prompt() — no torch needed."""

    def test_exact_token_count_small(self) -> None:
        tok = StubTokenizer()
        _, ids = build_prompt(tok, 10)
        assert len(ids) == 10

    def test_exact_token_count_large(self) -> None:
        tok = StubTokenizer()
        _, ids = build_prompt(tok, 200)
        assert len(ids) == 200

    def test_exact_token_count_one(self) -> None:
        tok = StubTokenizer()
        _, ids = build_prompt(tok, 1)
        assert len(ids) == 1

    def test_returns_tuple_of_str_and_list(self) -> None:
        tok = StubTokenizer()
        result = build_prompt(tok, 5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, ids = result
        assert isinstance(text, str)
        assert isinstance(ids, list)

    def test_raises_on_zero(self) -> None:
        tok = StubTokenizer()
        with pytest.raises(ValueError, match="target_tokens must be > 0"):
            build_prompt(tok, 0)

    def test_raises_on_negative(self) -> None:
        tok = StubTokenizer()
        with pytest.raises(ValueError):
            build_prompt(tok, -5)

    def test_pad_token_used_when_needed(self) -> None:
        """Stub that always returns fewer tokens than requested → pads with pad_token_id."""

        class TinyTokenizer:
            pad_token_id = 99

            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                # Always returns at most 3 tokens regardless of input length
                words = text.split()
                return list(range(min(len(words), 3)))

            def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
                return " ".join(str(i) for i in ids)

        tok = TinyTokenizer()
        # Request 10 tokens; encode returns at most 3 → must pad to 10
        _, ids = build_prompt(tok, 10)
        assert len(ids) == 10
        # The extra entries should be pad_token_id (99)
        assert ids[-1] == 99


# ─── write_stage_jsonl ────────────────────────────────────────────────────────


class TestWriteStageJsonl:
    """Tests for write_stage_jsonl() — no torch needed."""

    REQUIRED_FIELDS = {"request_id", "stage", "start_ts", "end_ts", "duration_s"}

    def _make_records(self) -> List[dict]:
        """Return 4 synthetic stage records in contract order."""
        base = 1_700_000_000.0
        return [
            {"stage": s, "start_ts": base + i, "end_ts": base + i + 0.1, "duration_s": 0.1}
            for i, s in enumerate(CONTRACT_STAGES)
        ]

    def test_produces_correct_line_count(self) -> None:
        records = self._make_records()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_stage_jsonl(path, "req-001", records)
        lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4

    def test_each_line_is_valid_json(self) -> None:
        records = self._make_records()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_stage_jsonl(path, "req-001", records)
        for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
            obj = json.loads(line)  # must not raise
            assert isinstance(obj, dict)

    def test_exactly_five_fields_per_line(self) -> None:
        records = self._make_records()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_stage_jsonl(path, "req-001", records)
        for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
            obj = json.loads(line)
            assert set(obj.keys()) == self.REQUIRED_FIELDS, (
                f"Expected keys {self.REQUIRED_FIELDS}, got {set(obj.keys())}"
            )

    def test_request_id_propagated(self) -> None:
        records = self._make_records()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_stage_jsonl(path, "my-request-42", records)
        for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
            obj = json.loads(line)
            assert obj["request_id"] == "my-request-42"

    def test_stage_names_are_contract_stages(self) -> None:
        records = self._make_records()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_stage_jsonl(path, "req-x", records)
        found_stages = [
            json.loads(line)["stage"]
            for line in Path(path).read_text(encoding="utf-8").strip().splitlines()
        ]
        for s in found_stages:
            assert s in CONTRACT_STAGES, f"Unexpected stage name '{s}'"

    def test_duration_s_is_float(self) -> None:
        records = self._make_records()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        write_stage_jsonl(path, "req-001", records)
        for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
            obj = json.loads(line)
            assert isinstance(obj["duration_s"], float)


# ─── summarize ────────────────────────────────────────────────────────────────


class TestSummarize:
    """Tests for summarize() — no torch needed (torch import is guarded inside summarize)."""

    def _make_records(self, durations: list) -> List[dict]:
        base = 1_700_000_000.0
        assert len(durations) == 4
        return [
            {
                "stage": s,
                "start_ts": base + i,
                "end_ts": base + i + d,
                "duration_s": d,
            }
            for i, (s, d) in enumerate(zip(CONTRACT_STAGES, durations))
        ]

    def test_total_s_correct(self) -> None:
        durations = [0.01, 0.20, 0.50, 0.02]
        records = self._make_records(durations)
        result = summarize(
            request_id="r1",
            mode="torch",
            model_name="test-model",
            device_name="cpu",
            prompt_tokens=128,
            output_tokens=64,
            stage_records=records,
            decode_token_s=[],
        )
        assert abs(result["total_s"] - sum(durations)) < 1e-9

    def test_mean_decode_token_s_correct(self) -> None:
        decode_times = [0.01, 0.02, 0.03, 0.04]
        records = self._make_records([0.01, 0.2, 0.5, 0.02])
        result = summarize(
            request_id="r1",
            mode="torch",
            model_name="m",
            device_name="cpu",
            prompt_tokens=8,
            output_tokens=5,
            stage_records=records,
            decode_token_s=decode_times,
        )
        expected_mean = sum(decode_times) / len(decode_times)
        assert abs(result["mean_decode_token_s"] - expected_mean) < 1e-9

    def test_p95_decode_token_s_correct(self) -> None:
        # N=20 values; p95 index = ceil(0.95*20)-1 = ceil(19)-1 = 18
        decode_times = [float(i) for i in range(20)]  # 0.0 .. 19.0
        records = self._make_records([0.01, 0.2, 0.5, 0.02])
        result = summarize(
            request_id="r1",
            mode="torch",
            model_name="m",
            device_name="cpu",
            prompt_tokens=8,
            output_tokens=21,
            stage_records=records,
            decode_token_s=decode_times,
        )
        sorted_times = sorted(decode_times)
        expected_p95 = sorted_times[math.ceil(0.95 * len(sorted_times)) - 1]
        assert abs(result["p95_decode_token_s"] - expected_p95) < 1e-9

    def test_empty_decode_times_gives_zero(self) -> None:
        records = self._make_records([0.01, 0.2, 0.5, 0.02])
        result = summarize(
            request_id="r1",
            mode="nsys",
            model_name="m",
            device_name="cpu",
            prompt_tokens=8,
            output_tokens=1,
            stage_records=records,
            decode_token_s=[],
        )
        assert result["mean_decode_token_s"] == 0.0
        assert result["p95_decode_token_s"] == 0.0

    def test_summary_contains_required_keys(self) -> None:
        records = self._make_records([0.01, 0.2, 0.5, 0.02])
        result = summarize(
            request_id="r99",
            mode="torch",
            model_name="some-model",
            device_name="cuda:0",
            prompt_tokens=128,
            output_tokens=64,
            stage_records=records,
            decode_token_s=[0.01, 0.02],
        )
        required_keys = {
            "request_id", "mode", "model", "device", "torch_version",
            "prompt_tokens", "output_tokens", "stage_duration_s",
            "total_s", "decode_token_s", "mean_decode_token_s", "p95_decode_token_s",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_p95_single_value(self) -> None:
        """p95 of a single element list is that element itself."""
        records = self._make_records([0.01, 0.2, 0.5, 0.02])
        result = summarize(
            request_id="r1",
            mode="torch",
            model_name="m",
            device_name="cpu",
            prompt_tokens=8,
            output_tokens=2,
            stage_records=records,
            decode_token_s=[0.042],
        )
        assert abs(result["p95_decode_token_s"] - 0.042) < 1e-9


# ─── stage() context manager ─────────────────────────────────────────────────


class TestStageContextManager:
    """Tests for the stage() context manager — needs torch only for torch/nsys mode annotation."""

    def test_unknown_stage_raises_assertion_no_torch(self) -> None:
        """stage() must assert unknown names even before torch is touched."""
        from llm_profiler.collectors.deep_dive import stage

        # Use mode="nsys" and cuda_available=False → no torch import attempted
        records: list = []
        with pytest.raises(AssertionError, match="Unknown stage"):
            with stage("queue_wait", "nsys", False, records):
                pass

    def test_known_stage_records_entry_no_torch(self) -> None:
        """In nsys mode with cuda_available=False, torch is never imported."""
        from llm_profiler.collectors.deep_dive import stage

        records: list = []
        with stage("tokenization", "nsys", False, records):
            pass  # body: no torch needed
        assert len(records) == 1
        rec = records[0]
        assert rec["stage"] == "tokenization"
        assert "start_ts" in rec
        assert "end_ts" in rec
        assert "duration_s" in rec
        assert rec["duration_s"] >= 0.0

    def test_detokenization_recorded_no_torch(self) -> None:
        from llm_profiler.collectors.deep_dive import stage

        records: list = []
        with stage("detokenization", "nsys", False, records):
            pass
        assert records[0]["stage"] == "detokenization"


# ─── --help (CLI) ─────────────────────────────────────────────────────────────


class TestCLIHelp:
    """Verify --help exits 0 without torch."""

    def test_help_exits_zero(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "llm_profiler" / "collectors" / "deep_dive.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--help exited {result.returncode}:\n{result.stderr}"

    def test_help_mentions_mode(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "llm_profiler" / "collectors" / "deep_dive.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert "--mode" in result.stdout

    def test_help_mentions_prompt_tokens(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent / "llm_profiler" / "collectors" / "deep_dive.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert "--prompt-tokens" in result.stdout


# ─── run_request (torch required) ────────────────────────────────────────────


class TestRunRequest:
    """Integration test for run_request() — skipped when torch is absent."""

    @pytest.fixture(autouse=True)
    def require_torch(self) -> None:
        pytest.importorskip("torch", reason="torch not installed — skipping run_request tests")

    def _make_stub_model(self, vocab_size: int = 32, seq_len: int = 10) -> Any:
        """Return a minimal stub model compatible with run_request()."""
        import torch as _torch

        class _StubOutput:
            def __init__(self, n_tokens: int) -> None:
                # logits shape: [1, n_tokens, vocab_size]
                self.logits = _torch.zeros(1, n_tokens, vocab_size)
                self.past_key_values = None  # simplest valid value

        class _StubModel:
            def __call__(self, input_ids: Any, past_key_values: Any = None, use_cache: bool = True) -> Any:
                n = input_ids.shape[-1]
                return _StubOutput(n)

            def __enter__(self) -> "_StubModel":
                return self

            def __exit__(self, *args: Any) -> None:
                pass

        return _StubModel()

    def _make_stub_tokenizer(self, vocab_size: int = 32) -> Any:
        """Return a stub tokenizer compatible with run_request() / build_prompt()."""

        class _StubTok:
            pad_token_id = 0

            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                words = text.split()
                return list(range(len(words)))

            def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
                return " ".join(str(i) for i in ids)

        return _StubTok()

    def test_returns_four_stage_records(self) -> None:
        import torch as _torch
        from llm_profiler.collectors.deep_dive import run_request

        model = self._make_stub_model()
        tok = self._make_stub_tokenizer()
        device = _torch.device("cpu")

        records, decode_times = run_request(
            model=model,
            tokenizer=tok,
            device=device,
            prompt_tokens=10,
            output_tokens=5,
            mode="nsys",
            cuda_available=False,
        )
        assert len(records) == 4

    def test_stage_records_in_contract_order(self) -> None:
        import torch as _torch
        from llm_profiler.collectors.deep_dive import run_request

        model = self._make_stub_model()
        tok = self._make_stub_tokenizer()
        device = _torch.device("cpu")

        records, _ = run_request(
            model=model,
            tokenizer=tok,
            device=device,
            prompt_tokens=10,
            output_tokens=5,
            mode="nsys",
            cuda_available=False,
        )
        stage_names = [r["stage"] for r in records]
        assert stage_names == CONTRACT_STAGES

    def test_decode_token_count_is_output_tokens_minus_one(self) -> None:
        """First token comes from prefill → decode loop runs output_tokens - 1 times."""
        import torch as _torch
        from llm_profiler.collectors.deep_dive import run_request

        model = self._make_stub_model()
        tok = self._make_stub_tokenizer()
        device = _torch.device("cpu")
        output_tokens = 8

        _, decode_times = run_request(
            model=model,
            tokenizer=tok,
            device=device,
            prompt_tokens=10,
            output_tokens=output_tokens,
            mode="nsys",
            cuda_available=False,
        )
        assert len(decode_times) == output_tokens - 1

    def test_each_stage_record_has_required_fields(self) -> None:
        import torch as _torch
        from llm_profiler.collectors.deep_dive import run_request

        model = self._make_stub_model()
        tok = self._make_stub_tokenizer()
        device = _torch.device("cpu")

        records, _ = run_request(
            model=model,
            tokenizer=tok,
            device=device,
            prompt_tokens=10,
            output_tokens=4,
            mode="nsys",
            cuda_available=False,
        )
        for rec in records:
            assert "stage" in rec
            assert "start_ts" in rec
            assert "end_ts" in rec
            assert "duration_s" in rec
            assert rec["duration_s"] >= 0.0
