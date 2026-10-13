#!/usr/bin/env bash
set -euo pipefail

# Find script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_FILE="${PROJECT_ROOT}/llm_profiler/config.yaml"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
fi

# Activate virtual environment if present
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using existing active virtual environment: ${VIRTUAL_ENV}"
fi

# Extract settings from config.yaml using python
read -r MODEL_NAME SERVER_HOST SERVER_PORT <<< $(python3 -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('model_name', 'Qwen/Qwen2.5-0.5B-Instruct'), cfg.get('server_host', '127.0.0.1'), cfg.get('server_port', 8000))
")

MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
# On RTX 3060 Laptop (6GB VRAM), default utilization to 0.75-0.80
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.75}"
HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/health"
METRICS_URL="http://${SERVER_HOST}:${SERVER_PORT}/metrics"
TIMEOUT_S=300
LOG_FILE="${PROJECT_ROOT}/vllm_server.log"

echo "=================================================="
echo "Starting vLLM Reference Server"
echo "Model:       ${MODEL_NAME}"
echo "Host:        ${SERVER_HOST}"
echo "Port:        ${SERVER_PORT}"
echo "Max Len:     ${MAX_MODEL_LEN}"
echo "GPU Memory:  ${GPU_UTIL}"
echo "Log file:    ${LOG_FILE}"
echo "=================================================="

# Check if vllm is installed in environment
if python3 -c "import vllm" 2>/dev/null; then
    echo "vLLM detected. Launching vllm OpenAI-compatible API server..."
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_NAME}" \
        --host "${SERVER_HOST}" \
        --port "${SERVER_PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_UTIL}" \
        > "${LOG_FILE}" 2>&1 &
    SERVER_PID=$!
    echo "Server process launched with PID ${SERVER_PID}"
else
    echo "vLLM package not installed in active environment."
    echo "Checking for reference server launcher fallback..."
    # Launch reference OpenAI-compatible server if present
    if [[ -f "${SCRIPT_DIR}/reference_server.py" ]]; then
        nohup python3 "${SCRIPT_DIR}/reference_server.py" --config "${CONFIG_FILE}" > "${LOG_FILE}" 2>&1 &
        SERVER_PID=$!
        echo "Reference server launched with PID ${SERVER_PID}"
    else
        echo "ERROR: vLLM is not installed and no fallback server found."
        exit 1
    fi
fi

# Poll /health until healthy (200 OK) or timeout
echo "Waiting for server to become healthy at ${HEALTH_URL} (timeout: ${TIMEOUT_S}s)..."
START_TIME=$(date +%s)
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [[ ${ELAPSED} -gt ${TIMEOUT_S} ]]; then
        echo "ERROR: Server health check timed out after ${TIMEOUT_S}s."
        echo "Last 50 lines of server log:"
        tail -n 50 "${LOG_FILE}" || true
        exit 1
    fi

    # Check health status code
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_URL}" 2>/dev/null || echo "000")
    if [[ "${STATUS_CODE}" == "200" ]]; then
        echo "Server is HEALTHY (HTTP 200) after ${ELAPSED}s!"
        break
    fi

    sleep 2
done

echo "Server successfully started and healthy."
echo "Health:  ${HEALTH_URL}"
echo "Metrics: ${METRICS_URL}"
