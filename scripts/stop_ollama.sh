#!/usr/bin/env bash
# Stop System 1 (Ollama) server with env hook support
set -e

OLLAMA_URL="${BATON_OLLAMA_URL:-http://127.0.0.1:11434}"

echo "=================================================="
echo "  Stopping System 1 (Ollama) Server"
echo "=================================================="

# Custom stop hook
if [ -n "${BATON_OLLAMA_STOP_CMD}" ]; then
    echo "[INFO] Executing custom stop hook: ${BATON_OLLAMA_STOP_CMD}"
    eval "${BATON_OLLAMA_STOP_CMD}"
else
    echo "[INFO] Terminating Ollama processes..."
    pkill -f "ollama" 2>/dev/null || true
    pkill -f "llama-server" 2>/dev/null || true
fi

sleep 1
if curl -s --connect-timeout 1 "${OLLAMA_URL}/api/version" >/dev/null 2>&1; then
    echo "[WARNING] Ollama server is still responding at ${OLLAMA_URL}"
else
    echo "[SUCCESS] Verified: System 1 (Ollama) server is offline."
fi
