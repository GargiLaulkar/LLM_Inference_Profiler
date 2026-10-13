.PHONY: help server smoke baseline test validate clean

help:
	@echo "Available targets:"
	@echo "  make server    - Launch reference server"
	@echo "  make smoke     - Run load generator smoke test"
	@echo "  make baseline  - Run Baseline B1 suite (low_load)"
	@echo "  make test      - Run pytest and contract validation"
	@echo "  make validate  - Validate generated results CSV contracts"

server:
	python experiments/reference_server.py --config config.yaml

smoke:
	python experiments/load_generator.py --num-requests 5 --rps 1 --scenario-tag smoke_test --out results/smoke_test.csv

baseline:
	python experiments/run_baseline_b1.py --scenario low_load

test:
	pytest -v
	python experiments/check_contracts.py results/baseline_b1_low_load.csv

validate:
	python experiments/check_contracts.py results/baseline_b1_low_load.csv

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
