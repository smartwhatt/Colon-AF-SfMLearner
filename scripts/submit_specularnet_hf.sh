#!/usr/bin/env bash
# Submit script for SpecularNet Stage 2 training (Ablation B) on HF-backed C3VDv2.
# Skips Stage 1 and loads the best Stage 1 weights produced by submit_finetune_hf.sh.
#
# Usage:
#   bash scripts/submit_specularnet_hf.sh
#   S1_FINAL_WEIGHTS=/custom/path bash scripts/submit_specularnet_hf.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# =============================================================================
# CONFIGURE THESE
# =============================================================================

export DATA_SOURCE="hf"
export HF_REPO_ID="SmartWhatt/c3vdv2-SfM"
export HF_STREAMING="0"         # 0 = read through the local HF cache, 1 = stream from HF
export HF_CACHE_DIR=""          # empty uses HF default cache (~/.cache/huggingface)
export HF_SHUFFLE_BUFFER="10000"
export STEPS_PER_EPOCH=""       # set e.g. "500" to cap steps when streaming
export C3VDV2_VAL_BATCHES="8"    # fixed batches per validation log for metric depth/pose tags
export C3VDV2_DEV_HF_SPLIT=""    # optional in-domain HF dev split, when the repo provides one

export LOG_DIR="$PROJECT_DIR/logs/c3vdv2_hf"
export GPU="0"

# Stage 1 is skipped — reuse the best checkpoint from submit_finetune_hf.sh.
export SKIP_STAGE1=1
export S1_FINAL_WEIGHTS="${S1_FINAL_WEIGHTS:-$PROJECT_DIR/logs/c3vdv2_hf/stage1/models/weights_best/}"

# Stage 2 (same hyperparams as submit_finetune_hf.sh)
export S2_EPOCHS="20"
export S2_LR="1e-4"
export S2_BATCH="12"
export S2_LR_STEP="10"

# Supervised loss weight ramp
export W_DEPTH_START="1.0"
export W_DEPTH_END="0.3"
export W_POSE_START="0.5"
export W_POSE_END="0.1"
export RAMP_EPOCHS="8"

# SpecularNet — the whole point of this script
export USE_SPECULAR_NET="1"
export SPECULAR_LOSS_WEIGHT="0.05"
export SPECULAR_THRESHOLD="0.98"

# =============================================================================
# RUN
# =============================================================================

mkdir -p "$LOG_DIR"

echo ">>> SpecularNet Stage 2 (Ablation B)"
echo "    Stage 1 weights: $S1_FINAL_WEIGHTS"
echo "    Log dir:         $LOG_DIR"
echo "    HF repo:         $HF_REPO_ID"
echo ""

bash scripts/run_finetune_c3vdv2.sh
