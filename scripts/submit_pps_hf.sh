#!/usr/bin/env bash
# Submit script for PPS-only Stage 2 training (Ablation C) on HF-backed C3VDv2.
# Skips Stage 1 and loads the best Stage 1 weights produced by submit_finetune_hf.sh.
#
# Usage:
#   bash scripts/submit_pps_hf.sh
#   PPS_SUP_WEIGHT=0.005 S1_FINAL_WEIGHTS=/custom/path bash scripts/submit_pps_hf.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# =============================================================================
# CONFIGURE THESE
# =============================================================================

export DATA_SOURCE="${DATA_SOURCE:-hf}"
export HF_REPO_ID="${HF_REPO_ID:-SmartWhatt/c3vdv2-SfM}"
export HF_STREAMING="${HF_STREAMING:-0}"         # 0 = local HF cache, 1 = stream from HF
export HF_CACHE_DIR="${HF_CACHE_DIR:-}"          # empty uses HF default cache (~/.cache/huggingface)
export HF_SHUFFLE_BUFFER="${HF_SHUFFLE_BUFFER:-10000}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"    # set e.g. "500" to cap steps when streaming
export C3VDV2_VAL_BATCHES="${C3VDV2_VAL_BATCHES:-8}"
export C3VDV2_DEV_HF_SPLIT="${C3VDV2_DEV_HF_SPLIT:-}"

export LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/c3vdv2_hf}"
export GPU="${GPU:-0}"

# Stage 1 is skipped — reuse the best checkpoint from submit_finetune_hf.sh.
export SKIP_STAGE1=1
export S1_FINAL_WEIGHTS="${S1_FINAL_WEIGHTS:-$PROJECT_DIR/logs/c3vdv2_hf/stage1/models/weights_best/}"

# Stage 2
export S2_EPOCHS="${S2_EPOCHS:-20}"
export S2_LR="${S2_LR:-1e-4}"
export S2_BATCH="${S2_BATCH:-12}"
export S2_LR_STEP="${S2_LR_STEP:-10}"

# Supervised loss weight ramp
export W_DEPTH_START="${W_DEPTH_START:-1.0}"
export W_DEPTH_END="${W_DEPTH_END:-0.3}"
export W_POSE_START="${W_POSE_START:-0.5}"
export W_POSE_END="${W_POSE_END:-0.1}"
export RAMP_EPOCHS="${RAMP_EPOCHS:-8}"

# PPS-only — Ablation C
export USE_SPECULAR_NET="0"
export PPS_SUP_WEIGHT="${PPS_SUP_WEIGHT:-0.01}"
export PPS_SELF_WEIGHT="${PPS_SELF_WEIGHT:-0.0}"
export PPS_LIGHT_BASELINE_MM="${PPS_LIGHT_BASELINE_MM:-3.0}"

# =============================================================================
# RUN
# =============================================================================

mkdir -p "$LOG_DIR"

echo ">>> PPS-only Stage 2 (Ablation C)"
echo "    Stage 1 weights: $S1_FINAL_WEIGHTS"
echo "    Log dir:         $LOG_DIR"
echo "    Run dir:         $LOG_DIR/stage2_pps"
echo "    HF repo:         $HF_REPO_ID"
echo "    PPS sup weight:  $PPS_SUP_WEIGHT"
echo "    Light baseline:  $PPS_LIGHT_BASELINE_MM mm"
echo ""

bash scripts/run_finetune_c3vdv2.sh
