# PowerShell Reference Server Launcher for Windows
param (
    [string]$ConfigFile = "llm_profiler/config.yaml",
    [double]$GpuMemoryUtilization = 0.75,
    [int]$MaxModelLen = 2048,
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ConfigFile)) {
    if (Test-Path "config.yaml") {
        $ConfigFile = "config.yaml"
    } else {
        Write-Error "Config file not found at $ConfigFile or config.yaml"
    }
}

# Parse config settings using python
$configJson = python -c @"
import yaml, json
with open('$ConfigFile', 'r') as f:
    cfg = yaml.safe_load(f)
print(json.dumps({
    'model_name': cfg.get('model_name', 'Qwen/Qwen2.5-0.5B-Instruct'),
    'server_host': cfg.get('server_host', '127.0.0.1'),
    'server_port': cfg.get('server_port', 8000)
}))
"@

$parsed = $configJson | ConvertFrom-Json
$modelName = $parsed.model_name
$serverHost = $parsed.server_host
$serverPort = $parsed.server_port
$healthUrl = "http://${serverHost}:${serverPort}/health"
$metricsUrl = "http://${serverHost}:${serverPort}/metrics"
$logFile = "vllm_server.log"

Write-Host "=================================================="
Write-Host "Starting Reference Server"
Write-Host "Model:       $modelName"
Write-Host "Host:        $serverHost"
Write-Host "Port:        $serverPort"
Write-Host "Max Len:     $MaxModelLen"
Write-Host "GPU Memory:  $GpuMemoryUtilization"
Write-Host "Health URL:  $healthUrl"
Write-Host "Log file:    $logFile"
Write-Host "=================================================="

# Check if WSL is available to run official vLLM Linux server
$wslAvailable = (Get-Command wsl -ErrorAction SilentlyContinue) -ne $null

if ($wslAvailable) {
    Write-Host "WSL detected. Starting vLLM via WSL..."
    Start-Process -FilePath "wsl" -ArgumentList "bash", "-c", "cd /mnt/c/LLM_Inference_Profiler && bash llm_profiler/experiments/start_server.sh" -NoNewWindow
} else {
    $vllmInstalled = python -c "import vllm" 2>$null; $?
    if ($vllmInstalled) {
        Write-Host "Running vLLM directly on host..."
        Start-Process -FilePath "python" -ArgumentList "-m", "vllm.entrypoints.openai.api_server", "--model", "$modelName", "--host", "$serverHost", "--port", "$serverPort", "--max-model-len", "$MaxModelLen", "--gpu-memory-utilization", "$GpuMemoryUtilization" -RedirectStandardOutput $logFile -RedirectStandardError $logFile
    } else {
        Write-Host "Launching reference OpenAI-compatible server..."
        Start-Process -FilePath "python" -ArgumentList "llm_profiler/experiments/reference_server.py", "--config", "$ConfigFile" -RedirectStandardOutput $logFile -RedirectStandardError $logFile
    }
}

# Poll /health endpoint
Write-Host "Waiting for server to become healthy at $healthUrl (timeout: ${TimeoutSeconds}s)..."
$startTime = [DateTime]::UtcNow
$isHealthy = $false

while ((([DateTime]::UtcNow - $startTime).TotalSeconds) -lt $TimeoutSeconds) {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -Method Get -TimeoutSec 3 -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $isHealthy = $true
            break
        }
    } catch {
        # Server not yet ready
    }
    Start-Sleep -Seconds 2
}

if ($isHealthy) {
    Write-Host "Server is HEALTHY (HTTP 200)!"
    Write-Host "Health:  $healthUrl"
    Write-Host "Metrics: $metricsUrl"
} else {
    Write-Error "Server health check timed out after ${TimeoutSeconds} seconds."
}
