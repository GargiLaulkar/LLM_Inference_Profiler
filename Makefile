.PHONY: help up down logs smoke baseline-b1 baseline-b2 matrix test clean

COMPOSE_FILE := docker/docker-compose.yml
ENV_FILE := .env

help:
	@echo "LLM Inference Profiler Management"
	@echo "  make up          - Start stack (vLLM, Prometheus, Grafana)"
	@echo "  make down        - Stop and remove stack containers and networks"
	@echo "  make logs        - Tail container logs"
	@echo "  make smoke       - Run quick load generator smoke test"
	@echo "  make baseline-b1 - Run Baseline B1 (client-side latency)"
	@echo "  make baseline-b2 - Run Baseline B2 (built-in vLLM metrics)"
	@echo "  make matrix      - Run scenario matrix sweep"
	@echo "  make test        - Run test suite and contract validation"
	@echo "  make clean       - Clean test caches and temporary artifacts"

up:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) down

logs:
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) logs -f

smoke:
	python experiments/load_generator.py --num-requests 5 --rps 1 --scenario-tag smoke_test --out results/smoke_test.csv

baseline-b1:
	python experiments/run_baseline_b1.py --scenario low_load

baseline-b2:
	python experiments/run_baseline_b2.py --scenario low_load

matrix:
	python experiments/run_matrix.py --scenario rps_1 --repeat 1

test:
	pytest -v
	python experiments/check_contracts.py results/baseline_b1_low_load.csv

clean:
	python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True) + glob.glob('.pytest_cache')]"
