#!/usr/bin/env python3
"""Contract Validator for LLM Inference Profiler Request Log CSVs.

Validates that CSV files comply strictly with the schema contract:
- Column names and exact sequence
- Row data types
- Non-negative latencies and valid chronological timestamps
- Status format conventions

Exits 0 on full compliance, nonzero on any violation.
"""

import argparse
import csv
import os
import sys
from typing import List

EXPECTED_HEADERS = [
    "request_id",
    "send_ts",
    "end_ts",
    "total_latency_s",
    "prompt_tokens",
    "output_tokens",
    "status",
    "target_rps",
    "scenario_tag",
]


def validate_csv(csv_path: str) -> bool:
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file does not exist: {csv_path}", file=sys.stderr)
        return False

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            print(f"[ERROR] CSV file is empty: {csv_path}", file=sys.stderr)
            return False

        # 1. Header Validation
        if headers != EXPECTED_HEADERS:
            print(f"[ERROR] CSV column mismatch in {csv_path}:", file=sys.stderr)
            print(f"  Found:    {headers}", file=sys.stderr)
            print(f"  Expected: {EXPECTED_HEADERS}", file=sys.stderr)
            return False

        # 2. Row Data Validation
        row_count = 0
        errors: List[str] = []

        for row_idx, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) != len(EXPECTED_HEADERS):
                errors.append(f"Line {row_idx}: expected {len(EXPECTED_HEADERS)} fields, found {len(row)}")
                continue

            (
                req_id,
                send_ts_str,
                end_ts_str,
                lat_str,
                prompt_tok_str,
                out_tok_str,
                status,
                rps_str,
                scenario_tag,
            ) = row

            # Validate request_id
            if not req_id.strip():
                errors.append(f"Line {row_idx}: request_id is empty")

            # Validate send_ts
            try:
                send_ts = float(send_ts_str)
                if send_ts <= 0:
                    errors.append(f"Line {row_idx}: send_ts {send_ts} must be positive")
            except ValueError:
                errors.append(f"Line {row_idx}: invalid float send_ts '{send_ts_str}'")

            # Validate end_ts
            try:
                end_ts = float(end_ts_str)
                if end_ts < send_ts:
                    errors.append(f"Line {row_idx}: end_ts {end_ts} is before send_ts {send_ts}")
            except ValueError:
                errors.append(f"Line {row_idx}: invalid float end_ts '{end_ts_str}'")

            # Validate total_latency_s (no negative latencies!)
            try:
                latency = float(lat_str)
                if latency < 0.0:
                    errors.append(f"Line {row_idx}: negative latency detected: {latency}s")
            except ValueError:
                errors.append(f"Line {row_idx}: invalid float latency '{lat_str}'")

            # Validate prompt_tokens
            try:
                prompt_tokens = int(prompt_tok_str)
                if prompt_tokens < 0:
                    errors.append(f"Line {row_idx}: negative prompt_tokens: {prompt_tokens}")
            except ValueError:
                errors.append(f"Line {row_idx}: invalid integer prompt_tokens '{prompt_tok_str}'")

            # Validate output_tokens
            try:
                output_tokens = int(out_tok_str)
                if output_tokens < 0:
                    errors.append(f"Line {row_idx}: negative output_tokens: {output_tokens}")
            except ValueError:
                errors.append(f"Line {row_idx}: invalid integer output_tokens '{out_tok_str}'")

            # Validate status
            if not (status == "ok" or status.startswith("error:")):
                errors.append(f"Line {row_idx}: status '{status}' must be 'ok' or start with 'error:'")

            # Validate target_rps
            try:
                rps = float(rps_str)
                if rps <= 0.0:
                    errors.append(f"Line {row_idx}: target_rps {rps} must be positive")
            except ValueError:
                errors.append(f"Line {row_idx}: invalid float target_rps '{rps_str}'")

            # Validate scenario_tag
            if not scenario_tag.strip():
                errors.append(f"Line {row_idx}: scenario_tag is empty")

            if len(errors) > 20:
                errors.append("Too many errors encountered; stopping validation early.")
                break

        if errors:
            print(f"[FAIL] Contract validation failed for {csv_path} ({len(errors)} issues found):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return False

        if row_count == 0:
            print(f"[WARN] CSV contains header but zero data rows: {csv_path}", file=sys.stderr)

        print(f"[PASS] Contract validation PASSED for {csv_path} ({row_count} rows verified).")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate CSV against schema contracts")
    parser.add_argument("csv_path", help="Path to request log CSV file")
    args = parser.parse_args()

    success = validate_csv(args.csv_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
