#!/usr/bin/env bash
# ==============================================================================
# Script: stop_router.sh
# Purpose: WSL/Linux script to terminate llama-server router
# ==============================================================================

echo "=================================================="
echo "  Stopping llama-server Router"
echo "=================================================="

pkill -f "llama-server" >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[INFO] llama-server stopped successfully."
else
    echo "[INFO] llama-server is not currently running."
fi

echo "=================================================="
