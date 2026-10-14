#!/usr/bin/env bash
# ==============================================================================
# Script: run_deepseek_v4_gpu.sh
# Purpose: Launch DeepSeek-V4-Flash in Colibrì with NVIDIA RTX 3060 CUDA GPU
#          acceleration (dense attention in VRAM, MoE experts in 82GB RAM cache).
# ==============================================================================

set -e

MODEL_PATH="${MODEL_PATH:-/mnt/c/colibri_models/DeepSeek-V4-Flash}"
RAM_GB="${RAM_GB:-82}"
COLIBRI_DIR="${COLIBRI_DIR:-/mnt/e/dev/Antigravity/local_llm/colibri}"

# Ensure WSL NVIDIA driver passthrough is present in LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

# Keep MoE experts 100% on CPU + system RAM (eliminates PCIe ping-pong bottleneck);
# Dense attention (MLA, DSA, QKV projections) stays in RTX 3060 VRAM.
export DSV4_CUDA_EXPERT_MIRRORS="${DSV4_CUDA_EXPERT_MIRRORS:-0}"

# Enable zero-I/O prompt lookup speculative drafting for faster repeated/code tokens
export V4_DRAFT="${V4_DRAFT:-3}"

COMMAND="${1:-serve}"
shift || true

echo -e "\033[1;36m================================================================\033[0m"
echo -e "\033[1;32m   Launching DeepSeek-V4-Flash on Colibrì + NVIDIA RTX 3060     \033[0m"
echo -e "\033[1;36m================================================================\033[0m"
echo -e " Mode:       \033[1;33m$COMMAND\033[0m"
echo -e " Model:      \033[1;32m$MODEL_PATH\033[0m"
echo -e " RAM Budget: \033[1;32m${RAM_GB} GB\033[0m"
echo -e " GPU Target: \033[1;32mNVIDIA GeForce RTX 3060 (12GB VRAM via CUDA sm_86)\033[0m"
echo -e "\033[1;36m----------------------------------------------------------------\033[0m"

exec "$COLIBRI_DIR/c/coli" "$COMMAND" \
    --model "$MODEL_PATH" \
    --ram "$RAM_GB" \
    "$@"
