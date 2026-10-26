# M1 Progress Log

STEP 1: DONE - Repo skeleton, contracts, config, git init - ls; python -c "import yaml; print(yaml.safe_load(open('llm_profiler/config.yaml')))"
STEP 2: DONE - Environment verification script check_env.py - python experiments/check_env.py
STEP 3: DONE - Reference server & start_server scripts - curl http://127.0.0.1:8000/v1/completions; curl http://127.0.0.1:8000/metrics
STEP 4: DONE - Open-loop async load generator load_generator.py - python experiments/load_generator.py --dry-run; python experiments/load_generator.py --num-requests 5 --rps 1
STEP 5: DONE - Baseline B1 runner run_baseline_b1.py - python experiments/run_baseline_b1.py --scenario low_load
STEP 6: DONE - Test suite and sync docs - pytest; README.md and docs/team_sync_template.md
STEP 7: DONE - Contract validator, runner scripts, Makefile - python experiments/check_contracts.py results/baseline_b1_low_load.csv
STEP 8: DONE - Docker Compose stack with GPU-optimized vLLM, Prometheus, Grafana - docker compose -f docker/docker-compose.yml config; docker compose up -d
STEP 9: DONE - Prometheus scraping vLLM metrics at 5s interval - curl http://localhost:9090/api/v1/targets; curl http://localhost:9090/api/v1/query?query=vllm:num_requests_running
STEP 10: DONE - Grafana automated provisioning with datasource uid prometheus - curl http://localhost:3000/api/health
STEP 11: DONE - Grafana dashboard with live vLLM metrics and teammate placeholder rows - curl http://localhost:3000/api/dashboards/uid/llmprof-main; curl http://localhost:9090/api/v1/query?query=vllm:e2e_request_latency_seconds_count
STEP 12: DONE - Baseline B2 runner with vLLM histogram metrics snapshotting - python experiments/run_baseline_b2.py --scenario low_load
STEP 13: DONE - Scenario matrix sweep runner with repeat and index logging - python experiments/run_matrix.py --scenario rps_1 --repeat 1
STEP 14: DONE - Complete Makefile, clean-start README documentation, and architecture guide - ls Makefile docs/architecture_m1.md
STEP 15: DONE - Clean-machine teardown and launch verification (217.4s to working stack) - make down; make up; make smoke
STEP 16: DONE - M3 GPU collector integrated and verified on RTX 3060 - curl.exe http://localhost:9102/metrics; python -m pytest tests/test_gpu_collector.py

### M3 Integration Status Notes
- GPU Telemetry Collector: VERIFIED (utilization, memory used bytes, and power watts live on RTX 3060 and scraped by Prometheus).
- Test Suite: VERIFIED (all 146 tests passing across project, platform-independent exponent formatting in test_gpu_collector.py).
- NOT YET VERIFIED: Memory Breakdown panels (Row 4 in Grafana), because results/deep_dive/memory_breakdown.json from M3 is missing.
- NOT YET VERIFIED: Nsight deep-dive, not run on this machine.
