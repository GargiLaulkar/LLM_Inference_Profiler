"""Generate simple result charts for M4 experiments.

Charts are created only from real result files where they exist. If the required
file is missing, the script prints a skip message and continues.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "experiments" / "results" / "charts"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_chart_from_summary(summary_path: Path, out_dir: Path) -> None:
    if not summary_path.exists():
        print(f"[SKIP] Missing file: {summary_path}")
        return
    summary = _load_json(summary_path)
    if not summary:
        print(f"[SKIP] Empty file: {summary_path}")
        return
    if plt is None:
        print("[SKIP] matplotlib not installed; cannot generate chart.")
        return
    stage_counts = []
    for value in summary.get("results", {}).values():
        for scenario in value.get("scenarios", []):
            stage_counts.append(scenario.get("top_bottleneck", "unknown"))
    if not stage_counts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "SYNTHETIC", ha="center", va="center", fontsize=18)
        ax.set_axis_off()
        fig.savefig(out_dir / "synthetic_placeholder.png", dpi=150)
        return

    counts = {}
    for stage in stage_counts:
        counts[stage] = counts.get(stage, 0) + 1
    labels = list(counts.keys())
    values = list(counts.values())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.set_title("SYNTHETIC" if summary.get("data_source") == "synthetic" else "Top bottleneck counts")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_dir / "top_bottlenecks.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate report charts from M4 experiment result files.")
    parser.add_argument("--summary", type=str, default=str(ROOT / "experiments" / "results" / "suite_summary.json"), help="Summary JSON file to chart.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT), help="Directory for chart files.")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    build_chart_from_summary(summary_path, out_dir)
    print(f"[INFO] Charts saved under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
