#!/usr/bin/env bash
# Submit script for HF-backed REAL-Colon self-supervised fine-tuning.
# No SLURM - just sets environment variables and delegates to
# run_finetune_real_colon.sh.
#
# Defaults are intentionally diagnostic-sized. Run B + pose_transform is the
# frozen-depth recovery diagnostic; set ADAPTATION_PRESET=full to keep the
# previous full Stage 2 adaptation behavior.
#
# Usage:
#   bash scripts/submit_real_colon_hf.sh
#   ADAPTATION_PRESET=full START_RUN=run_d bash scripts/submit_real_colon_hf.sh
#   ADAPTATION_PRESET=teacher_pose_transform START_RUN=run_b bash scripts/submit_real_colon_hf.sh
#   EPOCHS=20 STEPS_PER_EPOCH="" MODEL_NAME=realcolon_run_d_full bash scripts/submit_real_colon_hf.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# =============================================================================
# CONFIGURE THESE
# =============================================================================

export DATA_SOURCE="${DATA_SOURCE:-hf}"
export HF_REPO_ID="${HF_REPO_ID:-SmartWhatt/real-colon-SfM}"
export HF_STREAMING="${HF_STREAMING:-1}"         # 1 = stream from HF; 0 = local HF cache
export HF_CACHE_DIR="${HF_CACHE_DIR:-}"          # empty uses HF default cache (~/.cache/huggingface)
export HF_SHUFFLE_BUFFER="${HF_SHUFFLE_BUFFER:-10000}"
if [ -z "${STEPS_PER_EPOCH+x}" ]; then
    export STEPS_PER_EPOCH="500"                 # diagnostic cap; set "" for full pass
else
    export STEPS_PER_EPOCH
fi

export LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/real_colon_hf}"
export GPU="${GPU:-0}"
export ADAPTATION_PRESET="${ADAPTATION_PRESET:-pose_transform}"

case "$ADAPTATION_PRESET" in
    full)
        ADAPTATION_TRAINABLE_MODULES="default Stage 2 modules"
        ;;
    pose_transform)
        ADAPTATION_TRAINABLE_MODULES="pose_encoder pose transform_encoder transform"
        ;;
    teacher_pose_transform)
        ADAPTATION_TRAINABLE_MODULES="pose_encoder pose transform_encoder transform"
        ;;
    *)
        echo "ERROR: ADAPTATION_PRESET must be one of: full, pose_transform, teacher_pose_transform"
        exit 1
        ;;
esac

# run_d = SpecularNet + PPS best aggregate C3VDv2 checkpoint.
# run_b = SpecularNet-only fallback from the stratified audit.
START_RUN="${START_RUN:-run_b}"
C3VDV2_LOG_DIR="${C3VDV2_LOG_DIR:-$PROJECT_DIR/logs/c3vdv2_hf}"

case "$START_RUN" in
    run_d)
        DEFAULT_START_WEIGHTS="$C3VDV2_LOG_DIR/stage2_specular_pps/models/weights_best/"
        if [ "$ADAPTATION_PRESET" = "pose_transform" ]; then
            DEFAULT_MODEL_NAME="realcolon_run_d_pose_transform"
        elif [ "$ADAPTATION_PRESET" = "teacher_pose_transform" ]; then
            DEFAULT_MODEL_NAME="realcolon_run_d_teacher_pose"
        else
            DEFAULT_MODEL_NAME="realcolon_run_d_diag"
        fi
        DEFAULT_USE_SPECULAR_NET="1"
        ;;
    run_b)
        DEFAULT_START_WEIGHTS="$C3VDV2_LOG_DIR/stage2_specular/models/weights_best/"
        if [ "$ADAPTATION_PRESET" = "pose_transform" ]; then
            DEFAULT_MODEL_NAME="realcolon_run_b_pose_transform"
        elif [ "$ADAPTATION_PRESET" = "teacher_pose_transform" ]; then
            DEFAULT_MODEL_NAME="realcolon_run_b_teacher_pose"
        else
            DEFAULT_MODEL_NAME="realcolon_run_b_diag"
        fi
        DEFAULT_USE_SPECULAR_NET="1"
        ;;
    run_a)
        DEFAULT_START_WEIGHTS="$C3VDV2_LOG_DIR/stage2/models/weights_best/"
        if [ "$ADAPTATION_PRESET" = "pose_transform" ]; then
            DEFAULT_MODEL_NAME="realcolon_run_a_pose_transform"
        elif [ "$ADAPTATION_PRESET" = "teacher_pose_transform" ]; then
            DEFAULT_MODEL_NAME="realcolon_run_a_teacher_pose"
        else
            DEFAULT_MODEL_NAME="realcolon_run_a_diag"
        fi
        DEFAULT_USE_SPECULAR_NET="0"
        ;;
    *)
        echo "ERROR: START_RUN must be one of: run_d, run_b, run_a"
        exit 1
        ;;
esac

export LOAD_WEIGHTS_FOLDER="${LOAD_WEIGHTS_FOLDER:-$DEFAULT_START_WEIGHTS}"
export LOAD_TRAINABLE_WEIGHTS_FOLDER="${LOAD_TRAINABLE_WEIGHTS_FOLDER:-$LOAD_WEIGHTS_FOLDER}"
if [ "$ADAPTATION_PRESET" = "teacher_pose_transform" ]; then
    export TEACHER_WEIGHTS_FOLDER="${TEACHER_WEIGHTS_FOLDER:-$LOAD_WEIGHTS_FOLDER}"
    export TEACHER_CONSISTENCY_WEIGHT="${TEACHER_CONSISTENCY_WEIGHT:-0.5}"
fi
export MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"

# Diagnostic defaults. Override for a full campaign.
export EPOCHS="${EPOCHS:-1}"
export LR="${LR:-1e-5}"
export BATCH="${BATCH:-12}"
export LR_STEP="${LR_STEP:-10}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export SAVE_FREQUENCY="${SAVE_FREQUENCY:-1}"
export LOG_FREQUENCY="${LOG_FREQUENCY:-50}"

# Match the C3VDv2 checkpoint depth range.
export MIN_DEPTH="${MIN_DEPTH:-0.001}"
export MAX_DEPTH="${MAX_DEPTH:-0.1}"

export USE_SPECULAR_NET="${USE_SPECULAR_NET:-$DEFAULT_USE_SPECULAR_NET}"
export SPECULAR_LOSS_WEIGHT="${SPECULAR_LOSS_WEIGHT:-0.05}"
export SPECULAR_THRESHOLD="${SPECULAR_THRESHOLD:-0.98}"

# =============================================================================
# RUN
# =============================================================================

mkdir -p "$LOG_DIR"

echo ">>> REAL-Colon HF self-supervised fine-tuning"
echo "    Start run:      $START_RUN"
echo "    Start weights:  $LOAD_WEIGHTS_FOLDER"
echo "    Preset:         $ADAPTATION_PRESET"
echo "    Trainable:      $ADAPTATION_TRAINABLE_MODULES"
echo "    Log dir:        $LOG_DIR"
echo "    Run dir:        $LOG_DIR/$MODEL_NAME"
echo "    HF repo:        $HF_REPO_ID"
echo "    HF streaming:   $HF_STREAMING"
echo "    Steps/epoch:    ${STEPS_PER_EPOCH:-full split}"
echo "    Epochs:         $EPOCHS"
echo "    SpecularNet:    $USE_SPECULAR_NET"
echo "    Teacher:        ${TEACHER_WEIGHTS_FOLDER:-off}"
echo "    Teacher weight: ${TEACHER_CONSISTENCY_WEIGHT:-0.0}"
echo ""

bash scripts/run_finetune_real_colon.sh
