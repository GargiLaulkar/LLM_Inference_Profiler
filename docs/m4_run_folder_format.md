# M4 external E1 run-folder format

M1 can send a parent folder containing one directory per measured run. Each run folder must contain exactly the following required files:

```text
e1_runs/
  slow_tokenizer_repeat_01/
    requests.csv
    traces.jsonl
    run.json
  control_repeat_01/
    requests.csv
    traces.jsonl
    run.json
```

`run.json` is a JSON object with required `injection` and `repeat` fields and an optional `model` field. `injection` must be `slow_tokenizer`, `no_batching`, `cpu_preprocess`, or `control`; `repeat` is a positive integer; `model`, when present, is a non-empty model name. Example:

```json
{"injection":"slow_tokenizer","repeat":1,"model":"Qwen/Qwen2.5-0.5B-Instruct"}
```

`requests.csv` must include the shared request columns:

```text
request_id,send_ts,end_ts,total_latency_s,prompt_tokens,output_tokens,status,target_rps,scenario_tag
```

`traces.jsonl` has one JSON object per measured stage and these exact fields:

```text
request_id,stage,start_ts,end_ts,duration_s
```

Allowed stage names are `tokenization`, `queue_wait`, `prefill`, `decode`, `detokenization`, `network`, and `total`. Trace request IDs must correspond to IDs in `requests.csv`; at most 5% of CSV requests may lack valid trace records. Unknown stages are counted and make that run invalid. Missing/unexpected IDs and invalid runs are reported; they are not silently repaired or filled in.

Run analysis from PowerShell at the repository root:

```powershell
python .\experiments\analyze_run_dir.py --run-dir C:\results\e1_runs --out-dir .\experiments\results
```

The command writes `e1_detection.csv` and `e1_summary.json` into the output directory. These results are labeled `data_source: "live"` because they represent externally supplied run files; the analyzer does not launch a server or generate substitute measurements.
