#!/usr/bin/env bash
# Start System 1 (Ollama) server with env hook support
set -e

OLLAMA_URL="${BATON_OLLAMA_URL:-http://127.0.0.1:11434}"

echo "=================================================="
echo "  Starting System 1 (Ollama) Server"
echo "=================================================="

# Check if already running
if curl -s --connect-timeout 1 "${OLLAMA_URL}/api/version" >/dev/null 2>&1; then
    echo "[INFO] Ollama server is already running and reachable at ${OLLAMA_URL}"
    exit 0
fi

# Custom start hook
if [ -n "${BATON_OLLAMA_START_CMD}" ]; then
    echo "[INFO] Executing custom start hook: ${BATON_OLLAMA_START_CMD}"
    eval "${BATON_OLLAMA_START_CMD}"
else
    echo "[INFO] Launching 'ollama serve' in background..."
    nohup ollama serve >/dev/null 2>&1 &
fi

# Wait up to 15 seconds for server to respond
echo "[INFO] Waiting for server at ${OLLAMA_URL} to initialize..."
for i in $(seq 1 15); do
    sleep 1
    if curl -s --connect-timeout 1 "${OLLAMA_URL}/api/version" >/dev/null 2>&1; then
        echo "[SUCCESS] Ollama server is online and listening at ${OLLAMA_URL}"
        exit 0
    fi
done

echo "[WARNING] Ollama healthcheck timed out at ${OLLAMA_URL}"
exit 1
