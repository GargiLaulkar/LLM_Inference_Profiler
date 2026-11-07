# M1 Experiments Progress

STEP 1: DONE - read docs, identified gateway URL config and X-Request-ID header; gateway cmd: python experiments/load_generator.py --config config_gateway.yaml --rps 2 --num-requests 50 --scenario-tag test
STEP 2: DONE - prometheus target profiler_exporter DOWN (M2 double-brace label bug, cannot fix); m4_cost_exporter restarted with M4_TIME_SHARE=1.0 test value; metrics available on port 9101
STEP 3: DONE - 8 E1 injection runs (control x 2, slow_tokenizer x 2, cpu_preprocess x 2, no_batching x 2), 50 req each via gateway port 8080; all runs valid, 50/50 traces per run
STEP 4: DONE - analyzer ran on results/e1_runs; detection_accuracy=0.333 (no_batching x 2 pass, tokenization injections fail: decode still dominates); top3_hit_rate=1.0; 0 control false positives; wrote docs/results/e1_summary.json and docs/results/e1_detection.csv
STEP 5: DONE - E6 overhead: off_1=597.9ms, on_1=609.2ms, off_2=594.8ms, on_2=617.2ms; overhead=+2.82% (under 5% target); wrote docs/results/e6_overhead.json
STEP 6: DONE - E2/E3 matrix: 6 scenarios via gateway, 50 req each, 100% success; rps_1=581ms, rps_5=700ms, rps_10=707ms, short_prompt=574ms, long_prompt=618ms, long_output=2468ms; wrote docs/results/e2_e3_summary.md and docs/results/e2_e3_raw.json
STEP 7: DONE - wrote docs/results/RESULTS.md with E1/E6/E2-E3 tables, stage breakdowns, limitations section
STEP 8: DONE - full pytest suite passed (224 passed, 1 skipped in 41.22s); git restore tests/synthetic/data verified clean; docs/ committed and pushed to origin/m1-experiments
