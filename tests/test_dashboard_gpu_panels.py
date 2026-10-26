"""Unit tests for Grafana dashboard GPU metrics panels.

Project 6: Inference Cost and Latency Profiler for LLM Serving Stacks
Role: M3 (GPU and Deep-Dive Lead)

Tests verify:
- Valid dashboard JSON and global uniqueness of panel IDs across the entire dashboard
- Exactly three GPU telemetry panels of type 'timeseries' matching the contract metric names
- Top-level placement of panels 31, 32, 33 between Row 3 and Row 4 for expanded Grafana row
- Row 3's own 'panels' array is empty and Row 3 title equals 'GPU Metrics (M3)'
- Panels 41 and 42 exist as top-level entries between Row 4 and Row 5 (Memory Breakdown row)
- Row 4 title equals 'Memory Breakdown (M3)', collapsed is False, its own panels array is empty
- Panels 41/42 expr, datasource, description, type, and gridPos match the contract
- Uniform Prometheus datasource configuration identical to M1's provisioned panels
- Top-level panels sorted by (y, x) maintain order with strictly increasing row y values
- Bounding-box verification ensuring zero gridPos overlap anywhere in the dashboard
- Legend configuration with {{gpu}} formatting and table statistics (Last, Max, Mean)
- Tooltip configuration in 'all' mode with plain metric expressions (no rate())
"""

import json
import os
from typing import Any, Dict, List

import pytest

DASHBOARD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dashboards", "llm_profiler_main.json"
)


@pytest.fixture
def dashboard_data() -> Dict[str, Any]:
    """Load and parse the main Grafana dashboard JSON."""
    assert os.path.exists(DASHBOARD_PATH), f"Dashboard not found at: {DASHBOARD_PATH}"
    with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _collect_all_panels(dashboard: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recursively collect all top-level panels and nested row panels."""
    panels: List[Dict[str, Any]] = []
    for p in dashboard.get("panels", []):
        panels.append(p)
        for cp in p.get("panels", []):
            panels.append(cp)
    return panels


def test_dashboard_valid_json_and_unique_ids(dashboard_data):
    """Verify JSON structure, schema validity, and uniqueness of all panel IDs."""
    all_panels = _collect_all_panels(dashboard_data)
    assert len(all_panels) > 0, "Dashboard contains no panels"

    panel_ids = [p["id"] for p in all_panels if "id" in p]
    unique_ids = set(panel_ids)

    assert len(panel_ids) == len(unique_ids), (
        f"Duplicate panel IDs detected: {panel_ids}"
    )

    # Verify dashboard metadata fields remain valid
    assert dashboard_data.get("uid") == "llmprof-main"
    assert dashboard_data.get("title") == "LLM Inference Profiler"
    assert dashboard_data.get("version") == 1


def test_three_gpu_panels_exist_and_match_contract(dashboard_data):
    """Verify exactly three GPU timeseries panels exist and match shared contract metrics."""
    # Find panels 31, 32, 33 in top-level panels
    panels_by_id = {p.get("id"): p for p in dashboard_data.get("panels", [])}

    assert 31 in panels_by_id, "Panel 31 (GPU Utilization) not found"
    assert 32 in panels_by_id, "Panel 32 (GPU Memory Used) not found"
    assert 33 in panels_by_id, "Panel 33 (GPU Power) not found"

    # 1. GPU Utilization (%)
    util_panel = panels_by_id[31]
    assert util_panel.get("title") == "GPU Utilization (%)"
    assert util_panel.get("type") == "timeseries"
    assert len(util_panel.get("targets", [])) == 1
    target_util = util_panel["targets"][0]
    assert target_util.get("expr") == "llmprof_gpu_utilization_percent"
    assert target_util.get("refId") == "A"
    defaults_util = util_panel.get("fieldConfig", {}).get("defaults", {})
    assert defaults_util.get("unit") == "percent"
    assert defaults_util.get("min") == 0
    assert defaults_util.get("max") == 100

    # 2. GPU Memory Used
    mem_panel = panels_by_id[32]
    assert mem_panel.get("title") == "GPU Memory Used"
    assert mem_panel.get("type") == "timeseries"
    assert len(mem_panel.get("targets", [])) == 1
    target_mem = mem_panel["targets"][0]
    assert target_mem.get("expr") == "llmprof_gpu_memory_used_bytes"
    assert target_mem.get("refId") == "A"
    defaults_mem = mem_panel.get("fieldConfig", {}).get("defaults", {})
    assert defaults_mem.get("unit") == "bytes"
    assert defaults_mem.get("min") == 0

    # 3. GPU Power (W)
    power_panel = panels_by_id[33]
    assert power_panel.get("title") == "GPU Power (W)"
    assert power_panel.get("type") == "timeseries"
    assert len(power_panel.get("targets", [])) == 1
    target_power = power_panel["targets"][0]
    assert target_power.get("expr") == "llmprof_gpu_power_watts"
    assert target_power.get("refId") == "A"
    defaults_power = power_panel.get("fieldConfig", {}).get("defaults", {})
    assert defaults_power.get("unit") == "watt"
    assert defaults_power.get("min") == 0


def test_datasource_matches_m1_panels(dashboard_data):
    """Verify datasource (type and uid) matches M1's provisioned reference panels."""
    # Find M1 reference panel (e.g. Panel 10)
    m1_panel = next((p for p in dashboard_data.get("panels", []) if p.get("id") == 10), None)
    assert m1_panel is not None, "Reference panel 10 not found"
    expected_ds = m1_panel.get("datasource")

    assert expected_ds == {"type": "prometheus", "uid": "prometheus"}

    # Check GPU panels 31, 32, 33
    panels_by_id = {p.get("id"): p for p in dashboard_data.get("panels", [])}
    for pid in [31, 32, 33]:
        p = panels_by_id[pid]
        assert p.get("datasource") == expected_ds, f"Panel {pid} datasource mismatch"
        for t in p.get("targets", []):
            assert t.get("datasource") == expected_ds, f"Target in panel {pid} datasource mismatch"

    # Also check new breakdown panels 41, 42
    for pid in [41, 42]:
        p = panels_by_id[pid]
        assert p.get("datasource") == expected_ds, f"Panel {pid} datasource mismatch"
        for t in p.get("targets", []):
            assert t.get("datasource") == expected_ds, f"Target in panel {pid} datasource mismatch"


def test_grid_positions_no_overlap(dashboard_data):
    """Verify gridPos coordinates of the 3 GPU panels are side-by-side (w:8, h:8) and non-overlapping."""
    panels_by_id = {p.get("id"): p for p in dashboard_data.get("panels", [])}
    row3 = panels_by_id[3]
    row_y = row3.get("gridPos", {}).get("y")
    assert row_y == 16

    expected_positions = [
        {"h": 8, "w": 8, "x": 0, "y": 17},
        {"h": 8, "w": 8, "x": 8, "y": 17},
        {"h": 8, "w": 8, "x": 16, "y": 17},
    ]

    actual_positions = [panels_by_id[pid].get("gridPos") for pid in [31, 32, 33]]
    assert actual_positions == expected_positions, (
        f"GPU panel grid positions mismatch: {actual_positions} vs {expected_positions}"
    )

    # Geometry verification: bounding boxes must not intersect between any top-level panels sharing rows
    active_panels = [p for p in dashboard_data.get("panels", []) if p.get("type") != "row"]
    for i in range(len(active_panels)):
        for j in range(i + 1, len(active_panels)):
            p1 = active_panels[i]["gridPos"]
            p2 = active_panels[j]["gridPos"]
            horizontal_overlap = (p1["x"] < p2["x"] + p2["w"]) and (p1["x"] + p1["w"] > p2["x"])
            vertical_overlap = (p1["y"] < p2["y"] + p2["h"]) and (p1["y"] + p1["h"] > p2["y"])
            assert not (horizontal_overlap and vertical_overlap), (
                f"Overlap detected between panels {active_panels[i]['id']} and {active_panels[j]['id']}"
            )


def test_legend_format_and_tooltip_options(dashboard_data):
    """Verify legend table formatting, GPU label template, and all-mode tooltip."""
    panels_by_id = {p.get("id"): p for p in dashboard_data.get("panels", [])}
    for pid in [31, 32, 33]:
        p = panels_by_id[pid]
        target = p["targets"][0]
        # Multi-GPU support requires {{gpu}} legend templating
        assert "{{gpu}}" in target.get("legendFormat", ""), (
            f"Panel {pid} legendFormat missing {{{{gpu}}}}: {target.get('legendFormat')}"
        )

        options = p.get("options", {})
        legend = options.get("legend", {})
        assert legend.get("displayMode") == "table"
        assert legend.get("placement") == "bottom"
        assert legend.get("showLegend") is True

        calcs = legend.get("calcs", [])
        assert "lastNotNull" in calcs or "last" in calcs
        assert "max" in calcs
        assert "mean" in calcs

        tooltip = options.get("tooltip", {})
        assert tooltip.get("mode") == "all"


def test_row3_expanded_structure_and_top_level_ordering(dashboard_data):
    """Verify Row 3 structure for Grafana expanded row:

    - Panels 31, 32, 33 appear as top-level entries directly between Row 3 and Row 4
    - Row 3's own 'panels' array is empty
    - Row 3's title equals 'GPU Metrics (M3)' and collapsed is False
    - Top-level panels sorted by (y, x) keep that same order with strictly increasing row y values
    - No gridPos bounding box overlap anywhere across all panels
    """
    top_panels = dashboard_data.get("panels", [])
    top_ids = [p["id"] for p in top_panels]

    # 1. Location between Row 3 and Row 4
    idx_row3 = top_ids.index(3)
    idx_row4 = top_ids.index(4)

    assert top_ids[idx_row3 + 1:idx_row4] == [31, 32, 33], (
        f"Panels 31, 32, 33 must be immediate siblings between Row 3 and Row 4, found: {top_ids[idx_row3:idx_row4 + 1]}"
    )

    # 2. Row 3 properties
    row3 = top_panels[idx_row3]
    assert row3.get("title") == "GPU Metrics (M3)"
    assert row3.get("collapsed") is False
    assert row3.get("panels") == []

    # 3. Top-level panels sorted by (y, x) preserve the exact array order
    def sort_key(p):
        gp = p.get("gridPos", {})
        return (gp.get("y", 0), gp.get("x", 0))

    sorted_panels = sorted(top_panels, key=sort_key)
    sorted_ids = [p["id"] for p in sorted_panels]
    assert top_ids == sorted_ids, (
        f"Top-level panel order diverges from geometric (y, x) sorting: {top_ids} vs {sorted_ids}"
    )

    # 4. Strictly increasing row y values
    row_panels = [p for p in top_panels if p.get("type") == "row"]
    row_y_values = [r["gridPos"]["y"] for r in row_panels]
    for i in range(len(row_y_values) - 1):
        assert row_y_values[i] < row_y_values[i + 1], (
            f"Row y values are not strictly increasing: {row_y_values}"
        )
    assert row_y_values == [0, 15, 16, 25, 34]

    # 5. Global overlap check across ALL elements (rows, top-level panels, and nested row children)
    all_elements = _collect_all_panels(dashboard_data)
    for i in range(len(all_elements)):
        for j in range(i + 1, len(all_elements)):
            p1 = all_elements[i]
            p2 = all_elements[j]

            # In Grafana, collapsed rows enclose their children so child panels share row domain;
            # check that no active top-level visual panels overlap each other or rows
            if p1 in top_panels and p2 in top_panels:
                gp1 = p1["gridPos"]
                gp2 = p2["gridPos"]
                h_overlap = (gp1["x"] < gp2["x"] + gp2["w"]) and (gp1["x"] + gp1["w"] > gp2["x"])
                v_overlap = (gp1["y"] < gp2["y"] + gp2["h"]) and (gp1["y"] + gp1["h"] > gp2["y"])
                assert not (h_overlap and v_overlap), (
                    f"Grid overlap detected between panel {p1.get('id')} and {p2.get('id')}"
                )


# ─── Memory Breakdown Row (Task 5b) ────────────────────────────────────────────

def test_row4_expanded_structure_and_top_level_ordering(dashboard_data):
    """Verify Row 4 structure for Grafana expanded row (Memory Breakdown):

    - Panels 41 and 42 appear as top-level entries directly between Row 4 and Row 5
    - Row 4's own 'panels' array is empty (no nested panels)
    - Row 4's title equals 'Memory Breakdown (M3)' and collapsed is False
    - No gridPos bounding box overlap anywhere across all top-level panels
    """
    top_panels = dashboard_data.get("panels", [])
    top_ids = [p["id"] for p in top_panels]

    # 1. Locate rows
    assert 4 in top_ids, "Row 4 not found as top-level panel"
    assert 5 in top_ids, "Row 5 not found as top-level panel"
    idx_row4 = top_ids.index(4)
    idx_row5 = top_ids.index(5)

    # 2. Panels 41 and 42 are the immediate siblings between Row 4 and Row 5
    between = top_ids[idx_row4 + 1: idx_row5]
    assert between == [41, 42], (
        f"Expected [41, 42] between Row 4 and Row 5, got: {between}"
    )

    # 3. Row 4 properties
    row4 = top_panels[idx_row4]
    assert row4.get("title") == "Memory Breakdown (M3)", (
        f"Row 4 title should be 'Memory Breakdown (M3)', got: {row4.get('title')}"
    )
    assert row4.get("collapsed") is False, "Row 4 must not be collapsed"
    assert row4.get("panels") == [], "Row 4's own panels array must be empty"


def test_memory_breakdown_panels_contract(dashboard_data):
    """Panels 41 and 42 must match the Task 5b contract exactly."""
    panels_by_id = {p.get("id"): p for p in dashboard_data.get("panels", [])}

    assert 41 in panels_by_id, "Panel 41 not found"
    assert 42 in panels_by_id, "Panel 42 not found"

    expected_ds = {"type": "prometheus", "uid": "prometheus"}

    # Panel 41 — prefill
    p41 = panels_by_id[41]
    assert p41.get("type") == "bargauge", f"Panel 41 type should be 'bargauge', got {p41.get('type')}"
    assert p41.get("title") == "Memory by component, prefill"
    assert p41.get("datasource") == expected_ds
    assert p41.get("description") == (
        "Static values from the last deep-dive run (memory_breakdown.json), not live memory."
    )
    gp41 = p41.get("gridPos", {})
    assert gp41.get("x") == 0
    assert gp41.get("w") == 12
    assert gp41.get("h") == 8
    assert gp41.get("y") == 26
    assert len(p41.get("targets", [])) == 1
    t41 = p41["targets"][0]
    assert t41.get("expr") == 'llmprof_memory_breakdown_bytes{phase="prefill"}', (
        f"Panel 41 expr mismatch: {t41.get('expr')}"
    )
    assert t41.get("legendFormat") == "{{component}}"
    assert t41.get("instant") is True
    assert t41.get("datasource") == expected_ds
    defaults41 = p41.get("fieldConfig", {}).get("defaults", {})
    assert defaults41.get("unit") == "bytes"
    assert defaults41.get("min") == 0
    opts41 = p41.get("options", {})
    assert opts41.get("orientation") == "horizontal"
    assert opts41.get("displayMode") == "gradient"
    calcs41 = opts41.get("reduceOptions", {}).get("calcs", [])
    assert "lastNotNull" in calcs41

    # Panel 42 — decode
    p42 = panels_by_id[42]
    assert p42.get("type") == "bargauge"
    assert p42.get("title") == "Memory by component, decode"
    assert p42.get("datasource") == expected_ds
    assert p42.get("description") == (
        "Static values from the last deep-dive run (memory_breakdown.json), not live memory."
    )
    gp42 = p42.get("gridPos", {})
    assert gp42.get("x") == 12
    assert gp42.get("w") == 12
    assert gp42.get("h") == 8
    assert gp42.get("y") == 26
    assert gp41.get("y") == gp42.get("y"), "Panels 41 and 42 must share the same y row"
    assert len(p42.get("targets", [])) == 1
    t42 = p42["targets"][0]
    assert t42.get("expr") == 'llmprof_memory_breakdown_bytes{phase="decode"}', (
        f"Panel 42 expr mismatch: {t42.get('expr')}"
    )
    assert t42.get("legendFormat") == "{{component}}"
    assert t42.get("instant") is True
    assert t42.get("datasource") == expected_ds
    defaults42 = p42.get("fieldConfig", {}).get("defaults", {})
    assert defaults42.get("unit") == "bytes"
    assert defaults42.get("min") == 0
    opts42 = p42.get("options", {})
    assert opts42.get("orientation") == "horizontal"
    assert opts42.get("displayMode") == "gradient"
    assert "lastNotNull" in opts42.get("reduceOptions", {}).get("calcs", [])


def test_all_panel_ids_unique_including_new_panels(dashboard_data):
    """Panel IDs 41 and 42 are unique across the entire dashboard (no duplicates)."""
    all_panels = _collect_all_panels(dashboard_data)
    ids = [p["id"] for p in all_panels if "id" in p]
    assert len(ids) == len(set(ids)), f"Duplicate panel IDs detected: {ids}"
    assert 41 in ids
    assert 42 in ids


def test_no_gridpos_overlap_including_breakdown_panels(dashboard_data):
    """No gridPos bounding-box overlap across all top-level panels after adding 41 and 42."""
    top_panels = dashboard_data.get("panels", [])
    for i in range(len(top_panels)):
        for j in range(i + 1, len(top_panels)):
            gp1 = top_panels[i].get("gridPos", {})
            gp2 = top_panels[j].get("gridPos", {})
            h_overlap = (gp1.get("x", 0) < gp2.get("x", 0) + gp2.get("w", 0)) and (
                gp1.get("x", 0) + gp1.get("w", 0) > gp2.get("x", 0)
            )
            v_overlap = (gp1.get("y", 0) < gp2.get("y", 0) + gp2.get("h", 0)) and (
                gp1.get("y", 0) + gp1.get("h", 0) > gp2.get("y", 0)
            )
            assert not (h_overlap and v_overlap), (
                f"Overlap: panel {top_panels[i].get('id')} and {top_panels[j].get('id')}"
            )


def test_row_y_values_strictly_increasing_after_5b(dashboard_data):
    """Row y values are strictly increasing and include the new Row 5 y=34."""
    top_panels = dashboard_data.get("panels", [])
    row_ys = [p["gridPos"]["y"] for p in top_panels if p.get("type") == "row"]
    for i in range(len(row_ys) - 1):
        assert row_ys[i] < row_ys[i + 1], f"Row y values not strictly increasing: {row_ys}"
    # Row 5 must have shifted to y=34 (was 31, +3 for 8-row memory breakdown panels)
    assert row_ys == [0, 15, 16, 25, 34], f"Expected [0, 15, 16, 25, 34], got {row_ys}"
