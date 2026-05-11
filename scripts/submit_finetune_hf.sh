#!/usr/bin/env bash
# Submit script for HF-backed C3VDv2 fine-tuning (both stages).
# No SLURM — just sets environment variables and delegates to run_finetune_c3vdv2.sh.
#
# Usage:
#   bash scripts/submit_finetune_hf.sh
#   SKIP_STAGE1=1 S1_FINAL_WEIGHTS=/path/to/weights bash scripts/submit_finetune_hf.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# =============================================================================
# CONFIGURE THESE
# =============================================================================

export DATA_SOURCE="hf"
export HF_REPO_ID="SmartWhatt/c3vdv2-SfM"
export HF_STREAMING="0"         # 0 = cache locally, 1 = stream from HF
export HF_CACHE_DIR="/media/data01/smartwatt/hf"          # leave empty to use HF default cache (~/.cache/huggingface)
export HF_SHUFFLE_BUFFER="10000"
export STEPS_PER_EPOCH=""       # set e.g. "500" to cap steps when streaming

export LOG_DIR="$PROJECT_DIR/logs/c3vdv2_hf"
export GPU="0"

# Optional: SCARED Stage 1 pretrained weights for warm-start.
# Set to "" to train Stage 1 from scratch.
export SCARED_STAGE1_WEIGHTS="$PROJECT_DIR/model/pretrained/scared_stage1/"

# Stage 1
export S1_EPOCHS="20"
export S1_LR="1e-4"
export S1_BATCH="12"
export S1_LR_STEP="10"

# Stage 2
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

# =============================================================================
# RUN
# =============================================================================

mkdir -p "$LOG_DIR"

bash scripts/run_finetune_c3vdv2.sh
