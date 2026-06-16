#!/usr/bin/env bash
# Submit script for HF-backed C3VDv2 Stage 2 SSL diagnostics.
#
# This is intentionally a thin preset wrapper over run_finetune_c3vdv2.sh.
# Training uses SSL/teacher losses only for the ssl_* presets; C3VDv2 GT is
# still used by the trainer for validation and best-checkpoint selection.
#
# Usage:
#   bash scripts/submit_c3vdv2_ssl_hf.sh
#   START_RUN=run_d C3VDV2_STAGE2_PRESET=ssl_teacher_depth bash scripts/submit_c3vdv2_ssl_hf.sh
#   START_RUN=run_b C3VDV2_STAGE2_PRESET=ssl_teacher_depth EPOCHS=1 STEPS_PER_EPOCH=500 bash scripts/submit_c3vdv2_ssl_hf.sh

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
if [ -z "${STEPS_PER_EPOCH+x}" ]; then
    export STEPS_PER_EPOCH="500"                 # diagnostic cap; set "" for full pass
else
    export STEPS_PER_EPOCH
fi
export C3VDV2_VAL_BATCHES="${C3VDV2_VAL_BATCHES:-8}"
export C3VDV2_DEV_HF_SPLIT="${C3VDV2_DEV_HF_SPLIT:-}"

export LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/c3vdv2_hf_ssl}"
export GPU="${GPU:-0}"
export C3VDV2_STAGE2_PRESET="${C3VDV2_STAGE2_PRESET:-ssl_full}"

case "$C3VDV2_STAGE2_PRESET" in
    supervised)
        PRESET_TRAINABLE="default Stage 2 modules"
        ;;
    ssl_full)
        PRESET_TRAINABLE="default Stage 2 modules"
        ;;
    ssl_depth_only)
        PRESET_TRAINABLE="depth"
        ;;
    ssl_teacher_depth)
        PRESET_TRAINABLE="depth"
        ;;
    ssl_pose_transform_control)
        PRESET_TRAINABLE="pose_encoder pose transform_encoder transform"
        ;;
    *)
        echo "ERROR: C3VDV2_STAGE2_PRESET must be one of: supervised, ssl_full, ssl_depth_only, ssl_teacher_depth, ssl_pose_transform_control"
        exit 1
        ;;
esac

# run_d = SpecularNet + PPS best aggregate C3VDv2 checkpoint.
# run_b = SpecularNet-only fallback and best raw PolypSense3D MAE reference.
# run_a = plain fine-tuned AF-SFM baseline.
START_RUN="${START_RUN:-run_d}"
C3VDV2_LOG_DIR="${C3VDV2_LOG_DIR:-$PROJECT_DIR/logs/c3vdv2_hf}"

case "$START_RUN" in
    run_d)
        DEFAULT_START_WEIGHTS="$C3VDV2_LOG_DIR/stage2_specular_pps/models/weights_best/"
        DEFAULT_USE_SPECULAR_NET="1"
        ;;
    run_b)
        DEFAULT_START_WEIGHTS="$C3VDV2_LOG_DIR/stage2_specular/models/weights_best/"
        DEFAULT_USE_SPECULAR_NET="1"
        ;;
    run_a)
        DEFAULT_START_WEIGHTS="$C3VDV2_LOG_DIR/stage2/models/weights_best/"
        DEFAULT_USE_SPECULAR_NET="0"
        ;;
    *)
        echo "ERROR: START_RUN must be one of: run_d, run_b, run_a"
        exit 1
        ;;
esac

export SKIP_STAGE1=1
export S1_FINAL_WEIGHTS="${S1_FINAL_WEIGHTS:-$DEFAULT_START_WEIGHTS}"
export LOAD_TRAINABLE_WEIGHTS_FOLDER="${LOAD_TRAINABLE_WEIGHTS_FOLDER:-$S1_FINAL_WEIGHTS}"
if [ "$C3VDV2_STAGE2_PRESET" = "ssl_teacher_depth" ]; then
    export TEACHER_WEIGHTS_FOLDER="${TEACHER_WEIGHTS_FOLDER:-$LOAD_TRAINABLE_WEIGHTS_FOLDER}"
    export TEACHER_CONSISTENCY_WEIGHT="${TEACHER_CONSISTENCY_WEIGHT:-0.5}"
fi

RUN_STEPS_LABEL="${STEPS_PER_EPOCH:-full}"
DEFAULT_MODEL_NAME="c3vdv2_${START_RUN}_${C3VDV2_STAGE2_PRESET}_${RUN_STEPS_LABEL}"
export S2_MODEL_NAME="${S2_MODEL_NAME:-${MODEL_NAME:-$DEFAULT_MODEL_NAME}}"

# Diagnostic defaults. Override for a full campaign.
export S2_EPOCHS="${S2_EPOCHS:-${EPOCHS:-1}}"
export S2_LR="${S2_LR:-${LR:-1e-5}}"
export S2_BATCH="${S2_BATCH:-${BATCH:-12}}"
export S2_LR_STEP="${S2_LR_STEP:-${LR_STEP:-10}}"
export S2_SAVE_FREQUENCY="${S2_SAVE_FREQUENCY:-${SAVE_FREQUENCY:-1}}"
export S2_LOG_FREQUENCY="${S2_LOG_FREQUENCY:-${LOG_FREQUENCY:-50}}"

# Match the C3VDv2 checkpoint depth range.
export MIN_DEPTH="${MIN_DEPTH:-0.001}"
export MAX_DEPTH="${MAX_DEPTH:-0.1}"

export USE_SPECULAR_NET="${USE_SPECULAR_NET:-$DEFAULT_USE_SPECULAR_NET}"
export SPECULAR_THRESHOLD="${SPECULAR_THRESHOLD:-0.98}"

# SSL presets choose default supervised/PPS/specular weights inside
# run_finetune_c3vdv2.sh. Keep them unset here unless the caller explicitly
# exports them.
export PPS_SELF_WEIGHT="${PPS_SELF_WEIGHT:-0.0}"
export PPS_LIGHT_BASELINE_MM="${PPS_LIGHT_BASELINE_MM:-3.0}"

# =============================================================================
# RUN
# =============================================================================

mkdir -p "$LOG_DIR"

echo ">>> C3VDv2 HF Stage 2 SSL diagnostic"
echo "    Start run:      $START_RUN"
echo "    Start weights:  $S1_FINAL_WEIGHTS"
echo "    Warm start:     $LOAD_TRAINABLE_WEIGHTS_FOLDER"
echo "    Preset:         $C3VDV2_STAGE2_PRESET"
echo "    Trainable:      $PRESET_TRAINABLE"
echo "    Log dir:        $LOG_DIR"
echo "    Run dir:        $LOG_DIR/$S2_MODEL_NAME"
echo "    HF repo:        $HF_REPO_ID"
echo "    HF streaming:   $HF_STREAMING"
echo "    Steps/epoch:    ${STEPS_PER_EPOCH:-full split}"
echo "    Epochs:         $S2_EPOCHS"
echo "    LR:             $S2_LR"
echo "    SpecularNet:    $USE_SPECULAR_NET"
echo "    Teacher:        ${TEACHER_WEIGHTS_FOLDER:-off}"
echo "    Teacher weight: ${TEACHER_CONSISTENCY_WEIGHT:-0.0}"
echo "    GT supervision: off for ssl_* presets; GT remains validation-only"
echo ""

bash scripts/run_finetune_c3vdv2.sh
