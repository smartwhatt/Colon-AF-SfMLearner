#!/usr/bin/env bash
# =============================================================================
# run_finetune_c3vdv2.sh
# =============================================================================
# Full end-to-end fine-tuning pipeline for AF-SfMLearner on C3VDv2.
#
# Stages:
#   Stage 1 — Fine-tune registration/correspondence net.
#             Purely self-supervised. ~20 epochs.
#   Stage 2 — Fine-tune depth + pose + transform with frozen Stage 1 weights.
#             Supervised depth and pose losses ramp high → low.
#
# SpecularNet/PPS ablations (Phase 1):
#   USE_SPECULAR_NET=0  →  Ablation A: baseline (default)
#   USE_SPECULAR_NET=1  →  Ablation B: specular gating only (no PPS)
#   PPS_SUP_WEIGHT>0    →  Ablation C: PPS-only, or D with USE_SPECULAR_NET=1
#
# Usage:
#   bash scripts/run_finetune_c3vdv2.sh
#   DATA_PATH=/my/data LOG_DIR=/my/logs bash scripts/run_finetune_c3vdv2.sh
#   USE_SPECULAR_NET=1 DATA_PATH=/my/data LOG_DIR=/my/logs bash scripts/run_finetune_c3vdv2.sh
#
# To skip Stage 1 (use existing weights):
#   SKIP_STAGE1=1 S1_FINAL_WEIGHTS=/path/to/weights bash scripts/run_finetune_c3vdv2.sh
# =============================================================================

set -e

# =============================================================================
# CONFIGURE THESE
# =============================================================================

DATA_PATH="${DATA_PATH:-/path/to/C3VDv2}"
UNDISTORT_MAPS="${UNDISTORT_MAPS:-undistort_maps/}"
LOG_DIR="${LOG_DIR:-logs/c3vdv2}"
SPLIT="${SPLIT:-c3vdv2}"
GPU="${GPU:-0}"
DATA_SOURCE="${DATA_SOURCE:-local}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_STREAMING="${HF_STREAMING:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
HF_SHUFFLE_BUFFER="${HF_SHUFFLE_BUFFER:-10000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"
C3VDV2_VAL_BATCHES="${C3VDV2_VAL_BATCHES:-8}"
C3VDV2_DEV_SPLIT_FILE="${C3VDV2_DEV_SPLIT_FILE:-}"
C3VDV2_DEV_HF_SPLIT="${C3VDV2_DEV_HF_SPLIT:-}"

# Optional: path to SCARED pretrained weights to warm-start Stage 1.
# Set to "" to train Stage 1 from scratch.
SCARED_STAGE1_WEIGHTS="${SCARED_STAGE1_WEIGHTS:-./model/pretrained/scared_stage1/}"

# Stage 1 settings
S1_EPOCHS="${S1_EPOCHS:-20}"
S1_LR="${S1_LR:-1e-4}"
S1_BATCH="${S1_BATCH:-12}"
S1_LR_STEP="${S1_LR_STEP:-10}"

# Stage 2 settings
S2_EPOCHS="${S2_EPOCHS:-20}"
S2_LR="${S2_LR:-1e-4}"
S2_BATCH="${S2_BATCH:-12}"
S2_LR_STEP="${S2_LR_STEP:-10}"

# Loss weight ramping
W_DEPTH_START="${W_DEPTH_START:-1.0}"
W_DEPTH_END="${W_DEPTH_END:-0.3}"
W_POSE_START="${W_POSE_START:-0.5}"
W_POSE_END="${W_POSE_END:-0.1}"
RAMP_EPOCHS="${RAMP_EPOCHS:-8}"

# Depth range — must match evaluate_c3vdv2.py (MIN_DEPTH=0.001, MAX_DEPTH=0.100)
# Default options.py values (0.1 / 150) are KITTI-scale; C3VDv2 depth is 0-100mm in metres.
MIN_DEPTH="${MIN_DEPTH:-0.001}"
MAX_DEPTH="${MAX_DEPTH:-0.1}"

# ── SpecularNet (Phase 1 PR 1) ────────────────────────────────────────────────
# USE_SPECULAR_NET=0  →  Ablation A: baseline (no change to existing training)
# USE_SPECULAR_NET=1  →  Ablation B: specular gating + BCE weak supervision
USE_SPECULAR_NET="${USE_SPECULAR_NET:-0}"
SPECULAR_LOSS_WEIGHT="${SPECULAR_LOSS_WEIGHT:-0.05}"
SPECULAR_THRESHOLD="${SPECULAR_THRESHOLD:-0.98}"

# ── PPS loss (Phase 1 PR 2) ───────────────────────────────────────────────────
# PPS_SUP_WEIGHT=0     →  disabled (default)
# PPS_SUP_WEIGHT>0     →  supervised PPS shading consistency on C3VDv2
PPS_SUP_WEIGHT="${PPS_SUP_WEIGHT:-0.0}"
PPS_SELF_WEIGHT="${PPS_SELF_WEIGHT:-0.0}"
PPS_LIGHT_BASELINE_MM="${PPS_LIGHT_BASELINE_MM:-3.0}"

# ── Uncertainty-weighted pose loss (B.1, Kendall & Cipolla CVPR 2017) ─────────
# USE_UNCERTAINTY_POSE=0  →  disabled (default, fixed rot/trans weights)
# USE_UNCERTAINTY_POSE=1  →  learn log-variance scalars for rot and trans
USE_UNCERTAINTY_POSE="${USE_UNCERTAINTY_POSE:-0}"

# =============================================================================
# DERIVED PATHS
# =============================================================================

S1_LOG_NAME="stage1"

case "$PPS_SUP_WEIGHT" in
    0|0.0|0.00|0.000|0.0000) PPS_ENABLED=0 ;;
    *) PPS_ENABLED=1 ;;
esac

# Stage 2 log name encodes the ablation variant for clean separation in TensorBoard
if [ "$USE_SPECULAR_NET" = "1" ] && [ "$PPS_ENABLED" = "1" ]; then
    S2_LOG_NAME="stage2_specular_pps"
elif [ "$USE_SPECULAR_NET" = "1" ]; then
    S2_LOG_NAME="stage2_specular"
elif [ "$PPS_ENABLED" = "1" ]; then
    S2_LOG_NAME="stage2_pps"
else
    S2_LOG_NAME="stage2"
fi
if [ "$USE_UNCERTAINTY_POSE" = "1" ]; then
    S2_LOG_NAME="${S2_LOG_NAME}_uncertainty"
fi

SKIP_STAGE1="${SKIP_STAGE1:-0}"

# S1_FINAL_WEIGHTS can be overridden externally (e.g. SKIP_STAGE1=1 with existing ckpt)
S1_FINAL_WEIGHTS_DEFAULT="${LOG_DIR}/${S1_LOG_NAME}/models/weights_$((S1_EPOCHS - 1))/"
S1_BEST_WEIGHTS="${LOG_DIR}/${S1_LOG_NAME}/models/weights_best/"
S1_FINAL_WEIGHTS="${S1_FINAL_WEIGHTS:-$S1_BEST_WEIGHTS}"

# =============================================================================
# HELPERS
# =============================================================================

log() { echo -e "\n\033[1;36m>>> $*\033[0m\n"; }

HF_ARGS=(--data_source "$DATA_SOURCE")
if [ "$DATA_SOURCE" = "hf" ]; then
    HF_ARGS+=(--hf_repo_id "$HF_REPO_ID")
    HF_ARGS+=(--hf_shuffle_buffer "$HF_SHUFFLE_BUFFER")
    if [ "$HF_STREAMING" = "1" ]; then
        HF_ARGS+=(--hf_streaming)
    fi
    if [ -n "$HF_CACHE_DIR" ]; then
        HF_ARGS+=(--hf_cache_dir "$HF_CACHE_DIR")
    fi
    if [ -n "$STEPS_PER_EPOCH" ]; then
        HF_ARGS+=(--steps_per_epoch "$STEPS_PER_EPOCH")
    fi
    if [ -n "$C3VDV2_DEV_HF_SPLIT" ]; then
        HF_ARGS+=(--c3vdv2_dev_hf_split "$C3VDV2_DEV_HF_SPLIT")
    fi
fi

C3VDV2_ARGS=(--c3vdv2_val_batches "$C3VDV2_VAL_BATCHES")
if [ -n "$C3VDV2_DEV_SPLIT_FILE" ]; then
    C3VDV2_ARGS+=(--c3vdv2_dev_split_file "$C3VDV2_DEV_SPLIT_FILE")
fi

check_data() {
    if [ "$DATA_SOURCE" = "hf" ]; then
        if [ -z "$HF_REPO_ID" ]; then
            echo "ERROR: HF_REPO_ID is required when DATA_SOURCE=hf"
            exit 1
        fi
        return
    fi

    if [ ! -d "$DATA_PATH" ]; then
        echo "ERROR: DATA_PATH does not exist: $DATA_PATH"
        echo "Set DATA_PATH=/path/to/C3VDv2 before running."
        exit 1
    fi
    if [ ! -d "$UNDISTORT_MAPS" ]; then
        echo "ERROR: UNDISTORT_MAPS directory not found: $UNDISTORT_MAPS"
        echo "Run: uv run python -m scripts.compute_undistort_map --output_dir $UNDISTORT_MAPS"
        exit 1
    fi
    if [ ! -f "splits/${SPLIT}/train.txt" ]; then
        echo "ERROR: Split file not found: splits/${SPLIT}/train.txt"
        echo "Run: uv run python scripts/generate_splits.py --data_root \"$DATA_PATH\" --output_dir \"splits/${SPLIT}\""
        exit 1
    fi
}

# =============================================================================
# STAGE 1 — Registration net fine-tuning
# =============================================================================

run_stage1() {
    log "STAGE 1: Fine-tuning registration/correspondence net on C3VDv2"
    echo "  What:    position_encoder + position (only)"
    echo "  Loss:    photometric + smoothness  (no GT supervision)"
    echo "  Epochs:  $S1_EPOCHS  LR: $S1_LR  Batch: $S1_BATCH"
    echo "  Output:  $S1_FINAL_WEIGHTS_DEFAULT"
    echo "  Best:    $S1_BEST_WEIGHTS"

    LOAD_WEIGHTS_ARG=""
    if [ -n "$SCARED_STAGE1_WEIGHTS" ] && [ -d "$SCARED_STAGE1_WEIGHTS" ]; then
        LOAD_WEIGHTS_ARG="--load_weights_folder $SCARED_STAGE1_WEIGHTS"
        echo "  Init:    $SCARED_STAGE1_WEIGHTS  (SCARED Stage 1 pretrained)"
    else
        echo "  Init:    random (no pretrained Stage 1 weights found)"
    fi

    CUDA_VISIBLE_DEVICES=$GPU uv run python train.py --mode c3vdv2_stage1 \
        --dataset             c3vdv2 \
        --data_path           "$DATA_PATH" \
        --split               "$SPLIT" \
        --undistort_maps      "$UNDISTORT_MAPS" \
        --log_dir             "$LOG_DIR" \
        --model_name          "$S1_LOG_NAME" \
        --num_epochs          "$S1_EPOCHS" \
        --learning_rate       "$S1_LR" \
        --batch_size          "$S1_BATCH" \
        --scheduler_step_size "$S1_LR_STEP" \
        --height              256 \
        --width               320 \
        --scales              0 1 2 3 \
        --min_depth           "$MIN_DEPTH" \
        --max_depth           "$MAX_DEPTH" \
        --save_frequency      5 \
        "${HF_ARGS[@]}" \
        "${C3VDV2_ARGS[@]}" \
        $LOAD_WEIGHTS_ARG

    log "Stage 1 complete. Final weights: $S1_FINAL_WEIGHTS_DEFAULT"
    echo "  Best weights:  $S1_BEST_WEIGHTS"
}

# =============================================================================
# STAGE 2 — Depth + pose + transform fine-tuning
# =============================================================================

run_stage2() {
    log "STAGE 2: Fine-tuning depth + pose + transform on C3VDv2"
    echo "  What:    encoder+depth  transform_encoder+transform  pose_encoder+pose"
    echo "  Frozen:  position_encoder + position  (Stage 1 C3VDv2 weights)"
    echo "  Epochs:  $S2_EPOCHS  LR: $S2_LR  Batch: $S2_BATCH"
    echo "  w_depth: $W_DEPTH_START → $W_DEPTH_END  w_pose: $W_POSE_START → $W_POSE_END"
    echo "  Ramp:    fixed for $RAMP_EPOCHS epochs, then linear decay"

    if [ "$USE_SPECULAR_NET" = "1" ]; then
        echo "  SpecularNet: ON  (weight=$SPECULAR_LOSS_WEIGHT  threshold=$SPECULAR_THRESHOLD)"
    else
        echo "  SpecularNet: OFF"
    fi
    if [ "$PPS_ENABLED" = "1" ]; then
        echo "  PPS:         ON  (sup_weight=$PPS_SUP_WEIGHT  baseline_mm=$PPS_LIGHT_BASELINE_MM)"
    else
        echo "  PPS:         OFF"
    fi
    if [ "$USE_UNCERTAINTY_POSE" = "1" ]; then
        echo "  UncertPose:  ON  (Kendall & Cipolla learnable rot/trans scales)"
    else
        echo "  UncertPose:  OFF"
    fi
    if [ "$USE_SPECULAR_NET" = "1" ] && [ "$PPS_ENABLED" = "1" ]; then
        echo "  Ablation:    D  [SpecularNet + PPS]"
    elif [ "$USE_SPECULAR_NET" = "1" ]; then
        echo "  Ablation:    B  [SpecularNet gating only]"
    elif [ "$PPS_ENABLED" = "1" ]; then
        echo "  Ablation:    C  [PPS only]"
    else
        echo "  Ablation:    A  [baseline]"
    fi

    if [ ! -d "$S1_FINAL_WEIGHTS" ]; then
        echo "ERROR: Stage 1 output not found: $S1_FINAL_WEIGHTS"
        echo "Did Stage 1 complete? Or set S1_FINAL_WEIGHTS=/path/to/weights to override."
        exit 1
    fi

    # Build optional flag arrays
    SPECULAR_ARGS=""
    if [ "$USE_SPECULAR_NET" = "1" ]; then
        SPECULAR_ARGS="--use_specular_net --specular_loss_weight $SPECULAR_LOSS_WEIGHT --specular_threshold $SPECULAR_THRESHOLD"
    fi
    PPS_ARGS=(
        --pps_sup_weight "$PPS_SUP_WEIGHT"
        --pps_self_weight "$PPS_SELF_WEIGHT"
        --pps_light_baseline_mm "$PPS_LIGHT_BASELINE_MM"
    )
    UNCERTAINTY_POSE_ARGS=""
    if [ "$USE_UNCERTAINTY_POSE" = "1" ]; then
        UNCERTAINTY_POSE_ARGS="--use_uncertainty_pose"
    fi

    CUDA_VISIBLE_DEVICES=$GPU uv run python train.py --mode c3vdv2_stage2 \
        --dataset             c3vdv2 \
        --data_path           "$DATA_PATH" \
        --split               "$SPLIT" \
        --undistort_maps      "$UNDISTORT_MAPS" \
        --log_dir             "$LOG_DIR" \
        --model_name          "$S2_LOG_NAME" \
        --load_weights_folder "$S1_FINAL_WEIGHTS" \
        --num_epochs          "$S2_EPOCHS" \
        --learning_rate       "$S2_LR" \
        --batch_size          "$S2_BATCH" \
        --scheduler_step_size "$S2_LR_STEP" \
        --height              256 \
        --width               320 \
        --scales              0 1 2 3 \
        --min_depth           "$MIN_DEPTH" \
        --max_depth           "$MAX_DEPTH" \
        --save_frequency      5 \
        --w_depth_start       "$W_DEPTH_START" \
        --w_depth_end         "$W_DEPTH_END" \
        --w_pose_start        "$W_POSE_START" \
        --w_pose_end          "$W_POSE_END" \
        --ramp_epochs         "$RAMP_EPOCHS" \
        "${HF_ARGS[@]}" \
        "${C3VDV2_ARGS[@]}" \
        "${PPS_ARGS[@]}" \
        $SPECULAR_ARGS \
        $UNCERTAINTY_POSE_ARGS

    log "Stage 2 complete."
    echo "  Final weights: ${LOG_DIR}/${S2_LOG_NAME}/models/weights_$((S2_EPOCHS - 1))/"
    echo "  Best weights:  ${LOG_DIR}/${S2_LOG_NAME}/models/weights_best/"
}

# =============================================================================
# MAIN
# =============================================================================

log "AF-SfMLearner × C3VDv2 Fine-Tuning Pipeline"
echo "  GPU:          $GPU"
echo "  Data source:  $DATA_SOURCE"
echo "  Data:         $DATA_PATH"
echo "  HF repo:      ${HF_REPO_ID:-none}"
if [ "$DATA_SOURCE" = "hf" ]; then
    if [ "$HF_STREAMING" = "1" ]; then
        echo "  HF mode:      streaming"
    else
        echo "  HF mode:      cached"
        echo "  HF cache:     ${HF_CACHE_DIR:-~/.cache/huggingface}"
    fi
fi
echo "  Logs:         $LOG_DIR"
echo "  Split:        $SPLIT"
echo "  Val batches:  $C3VDV2_VAL_BATCHES"
echo "  SpecularNet:  $USE_SPECULAR_NET"

check_data

if [ "$SKIP_STAGE1" = "1" ]; then
    log "Skipping Stage 1 (SKIP_STAGE1=1). Using: $S1_FINAL_WEIGHTS"
else
    run_stage1
fi

run_stage2

S2_FINAL_WEIGHTS="${LOG_DIR}/${S2_LOG_NAME}/models/weights_$((S2_EPOCHS - 1))/"
S2_BEST_WEIGHTS="${LOG_DIR}/${S2_LOG_NAME}/models/weights_best/"

log "ALL DONE"
echo ""
if [ "$DATA_SOURCE" = "hf" ]; then
    echo "  Evaluate best checkpoint with:"
    echo "  uv run python tools/evaluate_c3vdv2.py \\"
    echo "      --data_source hf \\"
    echo "      --hf_repo_id \"$HF_REPO_ID\" \\"
    echo "      --hf_split test \\"
    echo "      --weights_folder $S2_BEST_WEIGHTS"
    echo ""
    echo "  Evaluate final checkpoint with:"
    echo "  uv run python tools/evaluate_c3vdv2.py \\"
    echo "      --data_source hf \\"
    echo "      --hf_repo_id \"$HF_REPO_ID\" \\"
    echo "      --hf_split test \\"
    echo "      --weights_folder $S2_FINAL_WEIGHTS"
else
    echo "  Evaluate best checkpoint with:"
    echo "  uv run python tools/evaluate_c3vdv2.py \\"
    echo "      --data_root $DATA_PATH \\"
    echo "      --weights_folder $S2_BEST_WEIGHTS \\"
    echo "      --split $SPLIT"
    echo ""
    echo "  Evaluate final checkpoint with:"
    echo "  uv run python tools/evaluate_c3vdv2.py \\"
    echo "      --data_root $DATA_PATH \\"
    echo "      --weights_folder $S2_FINAL_WEIGHTS \\"
    echo "      --split $SPLIT"
fi
echo ""
echo "  Monitor training:"
echo "  uv run tensorboard --logdir $LOG_DIR"
echo ""
echo "  Run ablation B (SpecularNet on, same Stage 1 weights):"
echo "  USE_SPECULAR_NET=1 SKIP_STAGE1=1 S1_FINAL_WEIGHTS=$S1_FINAL_WEIGHTS \\"
echo "    DATA_PATH=$DATA_PATH LOG_DIR=$LOG_DIR bash scripts/run_finetune_c3vdv2.sh"
