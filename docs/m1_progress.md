# M1 Progress Log

STEP 1: DONE - Repo skeleton, contracts, config, git init - ls; python -c "import yaml; print(yaml.safe_load(open('llm_profiler/config.yaml')))"
STEP 2: DONE - Environment verification script check_env.py - python experiments/check_env.py
STEP 3: DONE - Reference server & start_server scripts - curl http://127.0.0.1:8000/v1/completions; curl http://127.0.0.1:8000/metrics
STEP 4: DONE - Open-loop async load generator load_generator.py - python experiments/load_generator.py --dry-run; python experiments/load_generator.py --num-requests 5 --rps 1
STEP 5: DONE - Baseline B1 runner run_baseline_b1.py - python experiments/run_baseline_b1.py --scenario low_load
STEP 6: DONE - Test suite and sync docs - pytest; README.md and docs/team_sync_template.md
STEP 7: DONE - Contract validator, runner scripts, Makefile - python experiments/check_contracts.py results/baseline_b1_low_load.csv
