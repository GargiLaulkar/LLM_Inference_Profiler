"""Tests for llm_profiler/collectors/memory_breakdown.py (Task 5a, M3).

Design goals:
- All unit tests run on the development laptop without torch or an NVIDIA GPU.
- Tests that require real torch / CUDA are isolated and protected by pytest.importorskip / skipif.
- Fake tensor and stub cache classes simulate PyTorch objects.
- Tests cover analytical KV maths, component decomposition, all three cache formats,
  JSON contract validation, evidence calculation, matplotlib chart generation, and CLI --help.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List

import pytest

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_profiler.collectors.memory_breakdown import (
    CONTRACT_COMPONENTS,
    CONTRACT_PHASES,
    analytical_kv_bytes,
    build_breakdown_json,
    build_evidence,
    build_figure,
    cache_nbytes,
    compute_components,
    extract_prefill_and_cleanup,
    plot_breakdown,
)


# ─── FAKE TENSORS & CACHE STUBS ───────────────────────────────────────────────

class FakeTensor:
    """Minimal fake PyTorch tensor exposing numel() and element_size()."""

    def __init__(self, numel: int, element_size: int = 2):
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class StubModernLayer:
    """Simulates a layer object in modern transformers Cache with keys and values."""

    def __init__(self, key_tensor: FakeTensor, value_tensor: FakeTensor):
        self.keys = key_tensor
        self.values = value_tensor


class StubModernCache:
    """Structure (i): Cache object with a .layers list of layer objects."""

    def __init__(self, layers: List[StubModernLayer]):
        self.layers = layers


class StubOlderCache:
    """Structure (ii): Cache object with .key_cache and .value_cache lists."""

    def __init__(self, key_cache: List[FakeTensor], value_cache: List[FakeTensor]):
        self.key_cache = key_cache
        self.value_cache = value_cache


# ─── 1. ANALYTICAL KV CACHE TESTS ────────────────────────────────────────────

class TestAnalyticalKVBytes:
    """Tests for analytical_kv_bytes()."""

    def test_qwen_numbers_single_token(self) -> None:
        """Qwen2.5-0.5B: 24 layers, 2 KV heads, head_dim 64, fp16 (2 B) -> 12,288 B/token."""
        bytes_1tok = analytical_kv_bytes(
            num_layers=24,
            num_kv_heads=2,
            head_dim=64,
            tokens=1,
            dtype_bytes=2,
        )
        assert bytes_1tok == 12_288

    def test_qwen_numbers_192_tokens(self) -> None:
        """192 tokens (128 prompt + 64 output) -> 2,359,296 bytes."""
        bytes_192tok = analytical_kv_bytes(
            num_layers=24,
            num_kv_heads=2,
            head_dim=64,
            tokens=192,
            dtype_bytes=2,
        )
        assert bytes_192tok == 2_359_296

    def test_zero_tokens(self) -> None:
        assert analytical_kv_bytes(24, 2, 64, tokens=0, dtype_bytes=2) == 0

    def test_negative_tokens_raises(self) -> None:
        with pytest.raises(ValueError):
            analytical_kv_bytes(24, 2, 64, tokens=-1, dtype_bytes=2)


# ─── 2. COMPONENT MATHS TESTS ────────────────────────────────────────────────

class TestComputeComponents:
    """Tests for compute_components()."""

    def test_known_numbers_normal_case(self) -> None:
        """Verify decomposition when nvml_used >= peak_alloc."""
        weights = 1_000_000
        kv = 200_000
        peak_alloc = 1_500_000
        nvml_used = 2_000_000

        comps = compute_components(weights, kv, peak_alloc, nvml_used)

        assert comps["weights"] == weights
        assert comps["kv_cache"] == kv
        # activations = max(0, 1_500_000 - 1_000_000 - 200_000) = 300_000
        assert comps["activations"] == 300_000
        # other = max(0, 2_000_000 - 1_000_000 - 200_000 - 300_000) = 500_000
        assert comps["other"] == 500_000

        # Exact invariant: sum equals nvml_used
        total = sum(comps.values())
        assert total == nvml_used

    def test_hand_checked_kaggle_numbers_prefill(self) -> None:
        """Prefill: start 961.13, peak 1000.85, kv growth 1.50 -> ~38.22 MiB activations."""
        weights = 942_290_000
        start = 961_130_000
        peak = 1_000_850_000
        kv = 1_500_000
        nvml_used = 1_348_000_000

        comps = compute_components(
            weights=weights,
            kv=kv,
            peak_alloc=peak,
            nvml_used=nvml_used,
            allocated_at_start=start,
            kv_start=0,
        )

        expected_activations = peak - start - kv  # 1_000_850_000 - 961_130_000 - 1_500_000 = 38_220_000
        assert comps["activations"] == expected_activations
        assert comps["activations"] == 38_220_000
        assert comps["weights"] == weights
        assert comps["kv_cache"] == kv
        assert sum(comps.values()) == nvml_used

    def test_hand_checked_kaggle_numbers_decode(self) -> None:
        """Decode: start 962.63, peak 965.84, kv growth 0.74 (1.50->2.24) -> ~2.47 MiB activations."""
        weights = 942_290_000
        start = 962_630_000
        peak = 965_840_000
        kv_start = 1_500_000
        kv_end = 2_240_000
        nvml_used = 1_348_000_000

        comps = compute_components(
            weights=weights,
            kv=kv_end,
            peak_alloc=peak,
            nvml_used=nvml_used,
            allocated_at_start=start,
            kv_start=kv_start,
        )

        kv_growth = kv_end - kv_start  # 740_000
        expected_activations = peak - start - kv_growth  # 965_840_000 - 962_630_000 - 740_000 = 2_470_000
        assert comps["activations"] == expected_activations
        assert comps["activations"] == 2_470_000
        assert comps["weights"] == weights
        assert comps["kv_cache"] == kv_end
        assert sum(comps.values()) == nvml_used

    def test_activations_never_negative(self) -> None:
        """When peak is less than start + kv_growth, activations clamp to 0."""
        comps = compute_components(
            weights=1000,
            kv=300,
            peak_alloc=1050,
            nvml_used=2000,
            allocated_at_start=1100,
            kv_start=100,
        )
        assert comps["activations"] == 0
        for v in comps.values():
            assert v >= 0
        assert sum(comps.values()) == 2000

    def test_peak_alloc_less_than_weights_plus_kv(self) -> None:
        """Activations clamped to 0, no negative numbers, sum equals nvml_used."""
        weights = 1_000_000
        kv = 300_000
        peak_alloc = 1_100_000  # less than weights + kv (1_300_000)
        nvml_used = 2_000_000

        comps = compute_components(weights, kv, peak_alloc, nvml_used)

        assert comps["activations"] == 0
        assert comps["other"] == 700_000  # 2_000_000 - 1_000_000 - 300_000 - 0
        for val in comps.values():
            assert val >= 0
        assert sum(comps.values()) == nvml_used

    def test_zero_kv_and_zero_activations(self) -> None:
        weights = 500_000
        kv = 0
        peak_alloc = 500_000
        nvml_used = 800_000

        comps = compute_components(weights, kv, peak_alloc, nvml_used)
        assert comps["activations"] == 0
        assert comps["other"] == 300_000
        assert sum(comps.values()) == nvml_used

    def test_exact_keys(self) -> None:
        comps = compute_components(100, 20, 150, 200)
        assert set(comps.keys()) == set(CONTRACT_COMPONENTS)
        assert comps["other"] == 300_000
        assert sum(comps.values()) == nvml_used

    def test_exact_keys(self) -> None:
        comps = compute_components(100, 20, 150, 200)
        assert set(comps.keys()) == set(CONTRACT_COMPONENTS)


# ─── 3. CACHE SIZE CALCULATION TESTS ──────────────────────────────────────────

class TestCacheNBytes:
    """Tests for cache_nbytes() across all 3 supported cache shapes."""

    def test_shape_i_modern_cache_with_layers(self) -> None:
        """Structure (i): Cache with .layers list."""
        layers = [
            StubModernLayer(FakeTensor(100, 2), FakeTensor(100, 2)),
            StubModernLayer(FakeTensor(100, 2), FakeTensor(100, 2)),
        ]
        cache = StubModernCache(layers)
        # 4 tensors * 100 elements * 2 bytes = 800 bytes
        assert cache_nbytes(cache) == 800

    def test_shape_ii_older_cache_with_lists(self) -> None:
        """Structure (ii): Cache with .key_cache and .value_cache lists."""
        keys = [FakeTensor(200, 2), FakeTensor(200, 2)]
        values = [FakeTensor(200, 2), FakeTensor(200, 2)]
        cache = StubOlderCache(keys, values)
        # 4 tensors * 200 elements * 2 bytes = 1600 bytes
        assert cache_nbytes(cache) == 1600

    def test_shape_iii_legacy_tuple_of_tuples(self) -> None:
        """Structure (iii): Legacy tuple of (key, value) pairs."""
        cache = (
            (FakeTensor(50, 4), FakeTensor(50, 4)),
            (FakeTensor(50, 4), FakeTensor(50, 4)),
        )
        # 4 tensors * 50 elements * 4 bytes = 800 bytes
        assert cache_nbytes(cache) == 800

    def test_empty_or_none_cache(self) -> None:
        assert cache_nbytes(None) == 0
        assert cache_nbytes(()) == 0
        assert cache_nbytes([]) == 0


# ─── 4. BREAKDOWN JSON CONTRACT TESTS ─────────────────────────────────────────

class TestBuildBreakdownJSON:
    """Tests for build_breakdown_json() contract schema."""

    def _sample_breakdown(self) -> dict:
        prefill_c = {"weights": 1000, "kv_cache": 100, "activations": 200, "other": 300}
        decode_c = {"weights": 1000, "kv_cache": 150, "activations": 50, "other": 400}
        return build_breakdown_json(
            request_id="mb-test-01",
            generated_at_epoch=1700000000.0,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            device="cuda:0",
            gpu_total_bytes=16_000_000_000,
            prompt_tokens=128,
            output_tokens=64,
            method="torch_inference_mode_eager",
            nvml_idle_bytes=800_000_000,
            prefill_components=prefill_c,
            prefill_peak_alloc=1300,
            prefill_nvml_used=1600,
            decode_components=decode_c,
            decode_peak_alloc=1200,
            decode_nvml_used=1600,
            kv_per_token_measured=12288.0,
            kv_per_token_analytical=12288,
            kv_matches_analytical=True,
            prefill_alloc_start=1000,
            decode_alloc_start=1100,
        )

    def test_top_level_contract_keys(self) -> None:
        data = self._sample_breakdown()
        expected = {
            "request_id",
            "generated_at_epoch",
            "model",
            "device",
            "gpu_total_bytes",
            "prompt_tokens",
            "output_tokens",
            "method",
            "nvml_idle_bytes",
            "phases",
            "diagnostics",
            "kv_cache_bytes_per_token_measured",
            "kv_cache_bytes_per_token_analytical",
            "kv_matches_analytical",
        }
        assert set(data.keys()) == expected

    def test_phases_structure_and_keys(self) -> None:
        data = self._sample_breakdown()
        phases = data["phases"]
        assert set(phases.keys()) == set(CONTRACT_PHASES)

        required_phase_keys = {
            "weights",
            "kv_cache",
            "activations",
            "other",
            "nvml_used_bytes",
            "torch_peak_allocated_bytes",
        }
        for phase in CONTRACT_PHASES:
            assert set(phases[phase].keys()) == required_phase_keys

    def test_diagnostics_structure_and_keys(self) -> None:
        data = self._sample_breakdown()
        diagnostics = data["diagnostics"]
        expected_keys = {
            "allocated_after_load_bytes",
            "allocated_after_warmup_bytes",
            "prefill_allocated_at_start_bytes",
            "decode_allocated_at_start_bytes",
            "reserved_at_prefill_end_bytes",
            "reserved_at_decode_end_bytes",
            "resident_beyond_weights_bytes",
        }
        assert set(diagnostics.keys()) == expected_keys
        assert diagnostics["prefill_allocated_at_start_bytes"] == 1000
        assert diagnostics["decode_allocated_at_start_bytes"] == 1100

    def test_diagnostics_populated_from_dict(self) -> None:
        raw_diag = {
            "allocated_after_load_bytes": 942_000,
            "allocated_after_warmup_bytes": 960_000,
            "prefill_allocated_at_start_bytes": 961_000,
            "decode_allocated_at_start_bytes": 962_000,
            "reserved_at_prefill_end_bytes": 1_200_000,
            "reserved_at_decode_end_bytes": 1_200_000,
            "resident_beyond_weights_bytes": 19_000,
        }
        data = build_breakdown_json(
            request_id="mb-test-02",
            generated_at_epoch=1700000000.0,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            device="cuda:0",
            gpu_total_bytes=16_000_000_000,
            prompt_tokens=128,
            output_tokens=64,
            method="torch_inference_mode_eager",
            nvml_idle_bytes=800_000_000,
            prefill_components={"weights": 1000, "kv_cache": 100, "activations": 200, "other": 300},
            prefill_peak_alloc=1300,
            prefill_nvml_used=1600,
            decode_components={"weights": 1000, "kv_cache": 150, "activations": 50, "other": 400},
            decode_peak_alloc=1200,
            decode_nvml_used=1600,
            kv_per_token_measured=12288.0,
            kv_per_token_analytical=12288,
            kv_matches_analytical=True,
            diagnostics=raw_diag,
        )
        assert data["diagnostics"] == raw_diag


# ─── 5. PREFILL EXTRACTION AND CLEANUP TESTS ─────────────────────────────────

class TestExtractPrefillAndCleanup:
    """Verify that extract_prefill_and_cleanup drops references to logits/output."""

    def test_drops_references_weakref_check(self) -> None:
        import gc
        import weakref

        class StubLogits:
            def __getitem__(self, item: Any) -> "StubLogits":
                return self

            def argmax(self, dim: int = -1) -> int:
                return 42

        class StubPrefillOutput:
            def __init__(self) -> None:
                self.logits = StubLogits()
                self.past_key_values = [FakeTensor(100, 2)]

        output_instance = StubPrefillOutput()
        logits_ref = weakref.ref(output_instance.logits)
        output_ref = weakref.ref(output_instance)

        pkv, next_tok = extract_prefill_and_cleanup(output_instance)
        assert pkv is not None
        assert next_tok == 42

        # Release local references
        del output_instance
        gc.collect()

        assert output_ref() is None, "StubPrefillOutput reference was retained!"
        assert logits_ref() is None, "StubLogits reference was retained!"


# ─── 5. EVIDENCE ARTIFACT TESTS ───────────────────────────────────────────────

class TestBuildEvidence:
    """Tests for build_evidence() calculations."""

    def test_evidence_calculations(self) -> None:
        prefill_c = {"weights": 1_000_000, "kv_cache": 50_000, "activations": 100_000, "other": 200_000}
        decode_c = {"weights": 1_000_000, "kv_cache": 100_000, "activations": 20_000, "other": 280_000}
        nvml_used_decode = 1_400_000
        gpu_total_bytes = 10_000_000
        kv_per_token = 1000.0

        ev = build_evidence(
            gpu_total_bytes=gpu_total_bytes,
            weights_bytes=1_000_000,
            kv_cache_bytes_per_token=kv_per_token,
            kv_cache_bytes_end_of_request=100_000,
            nvml_used_decode=nvml_used_decode,
            activations_peak_prefill_bytes=100_000,
            activations_peak_decode_bytes=20_000,
            other_bytes_decode=280_000,
            prefill_components=prefill_c,
            decode_components=decode_c,
        )

        # kv_share = 100_000 / 1_400_000 = ~0.071429
        assert abs(ev["kv_share_of_used_memory_decode"] - (100_000 / 1_400_000)) < 1e-4

        # free bytes = 10_000_000 - 1_400_000 = 8_600_000
        assert ev["nvml_free_after_decode_bytes"] == 8_600_000

        # tokens fit = 8_600_000 // 1000 = 8600
        assert ev["kv_tokens_that_fit_in_free_memory_estimate"] == 8600

        # Dominant components: weights (1_000_000) > other (200k/280k)
        assert ev["dominant_component_prefill"] == "weights"
        assert ev["dominant_component_decode"] == "weights"

    def test_evidence_with_summary_data(self) -> None:
        prefill_c = {"weights": 100, "kv_cache": 10, "activations": 20, "other": 30}
        decode_c = {"weights": 100, "kv_cache": 20, "activations": 10, "other": 40}
        summary = {
            "stage_duration_s": {"prefill": 0.25, "decode": 0.75},
            "mean_decode_token_s": 0.012,
        }
        ev = build_evidence(
            gpu_total_bytes=1000,
            weights_bytes=100,
            kv_cache_bytes_per_token=10.0,
            kv_cache_bytes_end_of_request=20,
            nvml_used_decode=170,
            activations_peak_prefill_bytes=20,
            activations_peak_decode_bytes=10,
            other_bytes_decode=40,
            prefill_components=prefill_c,
            decode_components=decode_c,
            summary_data=summary,
        )
        assert ev["prefill_s"] == 0.25
        assert ev["decode_total_s"] == 0.75
        assert ev["decode_mean_token_s"] == 0.012
        assert len(ev["notes"]) >= 2

    def test_evidence_notes_contain_resident_beyond_weights(self) -> None:
        ev = build_evidence(
            gpu_total_bytes=1000,
            weights_bytes=100,
            kv_cache_bytes_per_token=10.0,
            kv_cache_bytes_end_of_request=20,
            nvml_used_decode=170,
            activations_peak_prefill_bytes=20,
            activations_peak_decode_bytes=10,
            other_bytes_decode=40,
            prefill_components={"weights": 100, "kv_cache": 10, "activations": 20, "other": 40},
            decode_components={"weights": 100, "kv_cache": 20, "activations": 10, "other": 40},
        )
        assert any("resident_beyond_weights_bytes" in n for n in ev["notes"])


# ─── 6. MATPLOTLIB VISUALIZATION TESTS ────────────────────────────────────────

class TestPlotBreakdown:
    """Tests for build_figure() and plot_breakdown()."""

    def test_generates_non_empty_png(self) -> None:
        data = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cuda:0",
            "prompt_tokens": 128,
            "output_tokens": 64,
            "phases": {
                "prefill": {
                    "weights": 980 * 1024 * 1024,
                    "kv_cache": 10 * 1024 * 1024,
                    "activations": 80 * 1024 * 1024,
                    "other": 300 * 1024 * 1024,
                },
                "decode": {
                    "weights": 980 * 1024 * 1024,
                    "kv_cache": 25 * 1024 * 1024,
                    "activations": 15 * 1024 * 1024,
                    "other": 310 * 1024 * 1024,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = os.path.join(tmp_dir, "test_breakdown.png")
            plot_breakdown(data, out_png)
            assert os.path.isfile(out_png)
            assert os.path.getsize(out_png) > 20_000

    def test_build_figure_and_plot_breakdown_kaggle_numbers(self) -> None:
        """Validate figure axes, bar counts (14), ylim scaling, and non-blank PNG content."""
        import matplotlib.image as mpimg
        import numpy as np

        mib = 1024 * 1024
        kaggle_data = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cuda:0",
            "prompt_tokens": 128,
            "output_tokens": 64,
            "phases": {
                "prefill": {
                    "weights": int(942.29 * mib),
                    "kv_cache": int(1.50 * mib),
                    "activations": int(38.22 * mib),
                    "other": int(659.18 * mib),
                },
                "decode": {
                    "weights": int(942.29 * mib),
                    "kv_cache": int(2.24 * mib),
                    "activations": int(2.47 * mib),
                    "other": int(694.18 * mib),
                },
            },
        }

        fig = build_figure(kaggle_data)
        assert len(fig.axes) == 2, f"Expected 2 axes, got {len(fig.axes)}"
        ax1, ax2 = fig.axes

        # Count drawn bar rectangles with height > 0
        rects_ax1 = [p for p in ax1.patches if p.get_height() > 0]
        rects_ax2 = [p for p in ax2.patches if p.get_height() > 0]
        assert len(rects_ax1) == 8, f"Expected 8 bars in left subplot, got {len(rects_ax1)}"
        assert len(rects_ax2) == 6, f"Expected 6 bars in right subplot, got {len(rects_ax2)}"
        assert len(rects_ax1) + len(rects_ax2) == 14

        # Verify each axes ylim top is above its tallest stack
        tallest1 = max(
            942.29 + 1.50 + 38.22 + 659.18,  # prefill = 1641.19 MiB
            942.29 + 2.24 + 2.47 + 694.18,   # decode = 1641.18 MiB
        )
        tallest2 = max(
            1.50 + 38.22 + 659.18,           # prefill = 698.90 MiB
            2.24 + 2.47 + 694.18,            # decode = 698.89 MiB
        )
        assert ax1.get_ylim()[1] > tallest1
        assert ax2.get_ylim()[1] > tallest2

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = os.path.join(tmp_dir, "kaggle_breakdown.png")
            plot_breakdown(kaggle_data, out_png)
            assert os.path.isfile(out_png)

            # File size > 20,000 bytes
            file_size = os.path.getsize(out_png)
            assert file_size > 20_000, f"PNG size {file_size} is <= 20,000 bytes"

            # Image verification: not blank (> 5% non-white pixels, > 20 distinct colours)
            img = mpimg.imread(out_png)
            rgb = img[:, :, :3]
            is_white = np.all(rgb >= 0.99, axis=2)
            non_white_frac = np.mean(~is_white)
            assert non_white_frac > 0.05, f"Non-white pixel fraction {non_white_frac:.4f} <= 0.05"

            unique_colors = len(np.unique(rgb.reshape(-1, 3), axis=0))
            assert unique_colors > 20, f"Distinct colours {unique_colors} <= 20"

    def test_kaggle_annotation_labels_are_separated_and_inside_figure(self) -> None:
        from matplotlib.text import Annotation

        mib = 1024 * 1024
        kaggle_data = {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cuda:0",
            "prompt_tokens": 128,
            "output_tokens": 64,
            "phases": {
                "prefill": {
                    "weights": int(942.29 * mib),
                    "kv_cache": int(1.50 * mib),
                    "activations": int(38.22 * mib),
                    "other": int(659.18 * mib),
                },
                "decode": {
                    "weights": int(942.29 * mib),
                    "kv_cache": int(2.24 * mib),
                    "activations": int(2.47 * mib),
                    "other": int(694.18 * mib),
                },
            },
        }

        fig = build_figure(kaggle_data)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        figure_bbox = fig.bbox
        annotations = [
            artist
            for ax in fig.axes
            for artist in ax.texts
            if isinstance(artist, Annotation)
        ]
        assert annotations, "Expected callout annotations for small memory segments"
        boxes = [annotation.get_window_extent(renderer) for annotation in annotations]

        for annotation, box in zip(annotations, boxes):
            assert box.x0 >= figure_bbox.x0, f"Annotation clipped on left: {annotation.get_text()}"
            assert box.y0 >= figure_bbox.y0, f"Annotation clipped on bottom: {annotation.get_text()}"
            assert box.x1 <= figure_bbox.x1, f"Annotation clipped on right: {annotation.get_text()}"
            assert box.y1 <= figure_bbox.y1, f"Annotation clipped on top: {annotation.get_text()}"

        for index, box in enumerate(boxes):
            for other in boxes[index + 1:]:
                assert not box.overlaps(other), (
                    f"Callout labels overlap: {box} and {other}"
                )


# ─── 7. CLI HELP TEST ────────────────────────────────────────────────────────

class TestCLIHelp:
    """Verify CLI --help runs cleanly without torch."""

    def test_help_exits_zero(self) -> None:
        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / "llm_profiler" / "collectors" / "memory_breakdown.py"),
            "--help",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "--prompt-tokens" in res.stdout
        assert "--no-chart" in res.stdout


# ─── 8. REAL TORCH / CUDA INTEGRATION (SKIPPED ON LAPTOP) ─────────────────────

class TestRealGPUIntegration:
    """Tests requiring a live GPU and CUDA."""

    @pytest.fixture(autouse=True)
    def require_torch_and_cuda(self) -> None:
        torch = pytest.importorskip("torch", reason="torch not installed")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_run_phases_stub(self) -> None:
        pass
