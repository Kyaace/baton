#!/usr/bin/env bash
# ==============================================================================
# Script: stop_ornith.sh
# Purpose: WSL/Linux script to stop running llama-server (Ornith)
# ==============================================================================

echo "=================================================="
echo "  Stopping llama-server (Ornith)"
echo "=================================================="

pkill -f llama-server 2>/dev/null || true
taskkill.exe /F /T /IM llama-server.exe 2>/dev/null || true

echo "=================================================="
