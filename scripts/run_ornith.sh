#!/usr/bin/env bash
# ==============================================================================
# Script: run_ornith.sh
# Purpose: WSL/Linux launcher for Ornith-1.5-35B-A3B MoE on llama-server
# ==============================================================================

set -e

export PATH="$HOME/.local/bin:/mnt/c/llama.cpp:$PATH"

SERVER_BIN=$(command -v llama-server || command -v llama-server.exe || echo "/mnt/c/llama.cpp/llama-server.exe")

if [ ! -x "$SERVER_BIN" ] && [ ! -f "$SERVER_BIN" ]; then
    echo "[ERROR] llama-server not found in PATH or /mnt/c/llama.cpp"
    exit 1
fi

MODEL_PATH="${BATON_ORNITH_MODEL:-${BATON_SYSTEM2_MODEL_PATH:-/mnt/c/colibri_models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-Q4_K_M.gguf}}"
PORT="${BATON_SYSTEM2_PORT:-8000}"

# Filter out 'serve' argument if passed
ARGS=()
for arg in "$@"; do
    if [ "$arg" != "serve" ]; then
        ARGS+=("$arg")
    fi
done

echo "=============================================================================="
echo "  Launching Ornith-1.5-35B-A3B on llama-server (CUDA)"
echo "=============================================================================="
echo "  Model:       $MODEL_PATH"
echo "  VRAM Dense:  -ngl 99 (RTX 3060 12GB)"
echo "  RAM Experts: -ot \".ffn_.*_exps.=CPU\" (System RAM MoE routing)"
echo "  Speculative: --spec-type draft-mtp (Multi-Token Prediction)"
echo "  Context:     -c 32768 (32,768 tokens)"
echo "  Endpoint:    http://127.0.0.1:$PORT/v1"
echo "=============================================================================="

exec "$SERVER_BIN" \
  -m "$MODEL_PATH" \
  -ngl 99 \
  -ot ".ffn_.*_exps.=CPU" \
  --spec-type draft-mtp \
  -c 32768 \
  --host 0.0.0.0 \
  --port "$PORT" "${ARGS[@]}"
