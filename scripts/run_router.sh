#!/usr/bin/env bash
# ==============================================================================
# Script: run_router.sh
# Purpose: WSL/Linux launcher for llama-server Multi-Model Router (Ornith + Qwen3)
# ==============================================================================

export PATH="$HOME/.local/bin:/mnt/c/llama.cpp:$PATH"

SERVER_BIN=$(command -v llama-server || command -v llama-server.exe || echo "/mnt/c/llama.cpp/llama-server.exe")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET_FILE="${BATON_ROUTER_PRESET:-}"

if [ -z "$PRESET_FILE" ]; then
    if [ -f "$SCRIPT_DIR/../models_preset.ini" ]; then
        PRESET_FILE="$SCRIPT_DIR/../models_preset.ini"
    elif [ -f "$SCRIPT_DIR/models_preset.ini" ]; then
        PRESET_FILE="$SCRIPT_DIR/models_preset.ini"
    elif [ -f "/mnt/e/dev/Antigravity/local_llm/models_preset.ini" ]; then
        PRESET_FILE="/mnt/e/dev/Antigravity/local_llm/models_preset.ini"
    fi
fi

if [ ! -f "$PRESET_FILE" ]; then
    echo "[ERROR] Preset configuration not found: $PRESET_FILE"
    exit 1
fi

PORT="${BATON_ROUTER_PORT:-8080}"

EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" == "serve" ]; then
        continue
    fi
    EXTRA_ARGS+=("$arg")
done

echo "=============================================================================="
echo "  Launching llama-server Multi-Model Router (WSL)"
echo "=============================================================================="
echo "  Preset:     $PRESET_FILE"
echo "  Max VRAM:   1 model active at a time (--models-max 1)"
echo "  Models:     - Ornith-1.5-35B-A3B (MoE: VRAM dense + RAM experts, MTP)"
echo "              - Qwen3:14B (Dense: 100% VRAM offloaded)"
echo "  Endpoint:   http://127.0.0.1:$PORT/v1"
echo "=============================================================================="

"$SERVER_BIN" \
  --models-preset "$PRESET_FILE" \
  --models-max 1 \
  --reasoning-budget 1200 \
  --reasoning-budget-message "Plan complete. Finish thinking and write your output now." \
  --host 0.0.0.0 \
  --port "$PORT" "${EXTRA_ARGS[@]}"
