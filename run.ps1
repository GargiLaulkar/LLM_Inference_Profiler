# PowerShell target runner script
param (
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet("server", "smoke", "baseline", "test", "validate")]
    [string]$Target
)

$ErrorActionPreference = "Stop"

switch ($Target) {
    "server" {
        Write-Host ">>> Running target: server"
        python experiments/reference_server.py --config config.yaml
    }
    "smoke" {
        Write-Host ">>> Running target: smoke"
        python experiments/load_generator.py --num-requests 5 --rps 1 --scenario-tag smoke_test --out results/smoke_test.csv
    }
    "baseline" {
        Write-Host ">>> Running target: baseline"
        python experiments/run_baseline_b1.py --scenario low_load
    }
    "test" {
        Write-Host ">>> Running target: test"
        pytest -v
        python experiments/check_contracts.py results/baseline_b1_low_load.csv
    }
    "validate" {
        Write-Host ">>> Running target: validate"
        python experiments/check_contracts.py results/baseline_b1_low_load.csv
    }
}
