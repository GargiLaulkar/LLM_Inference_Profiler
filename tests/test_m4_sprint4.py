from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "experiments" / script_name), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_experiments_dry_run(tmp_path):
    out_dir = tmp_path / "results"
    result = run_script("run_experiments.py", "--dry-run", "--out-dir", str(out_dir))
    assert result.returncode == 0, result.stderr
    summary = json.loads((out_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert summary["data_source"] == "synthetic"
    assert "E2" in summary["experiment_ids"]


def test_run_overhead_dry_run(tmp_path):
    out_file = tmp_path / "e6_overhead.json"
    result = run_script("run_overhead.py", "--dry-run", "--out", str(out_file))
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["status"] == "not_evaluated"
    assert payload["data_source"] == "synthetic"
    assert payload["measurement"] is None
    assert "SYNTHETIC - not an E6 result" in result.stdout
    assert "PASS/FAIL: NOT_EVALUATED" in result.stdout


def test_compare_baselines_dry_run(tmp_path):
    out_file = tmp_path / "baseline_compare.json"
    result = run_script("compare_baselines.py", "--dry-run", "--out", str(out_file))
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["status"] == "synthetic_only"


def test_make_report_charts_dry_run(tmp_path):
    out_dir = tmp_path / "charts"
    summary_file = tmp_path / "suite_summary.json"
    summary_file.write_text(
        json.dumps({"data_source": "synthetic", "results": {}}),
        encoding="utf-8",
    )
    result = run_script("make_report_charts.py", "--summary", str(summary_file), "--out-dir", str(out_dir))
    assert result.returncode == 0, result.stderr
    assert out_dir.exists()
