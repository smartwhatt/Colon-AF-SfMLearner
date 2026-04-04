#!/usr/bin/env bash
# =============================================================================
# run_finetune_c3vdv2.sh
# =============================================================================
# Full end-to-end fine-tuning pipeline for AF-SfMLearner on C3VDv2.
#
# What this script does:
#   Stage 1 — Fine-tune the registration/correspondence net on C3VDv2.
#             Purely self-supervised. ~20 epochs.
#             Teaches the net colonoscopy tissue appearance and scope motion.
#
#   Stage 2 — Fine-tune depth + pose + transform on C3VDv2, with frozen
#             Stage 1 registration weights. Supervised depth and pose losses
#             ramp from high weight → low weight as the self-supervised signal
#             becomes reliable.
#
# Prerequisites:
#   - AF-SfMLearner repo cloned and set up
#   - Python env with PyTorch, OpenCV, tensorboardX, etc.
#   - C3VDv2 dataset downloaded and organised
#   - Undistortion maps computed: python compute_undistort_map.py ...
#   - Split files generated:      python generate_splits.py ...
#   - AF-SfMLearner repo cloned and set up (train.py entry point)
#
# Usage:
#   bash run_finetune_c3vdv2.sh
#   or with custom paths:
#   DATA_PATH=/my/data LOG_DIR=/my/logs bash run_finetune_c3vdv2.sh
#
# =============================================================================

set -e  # exit immediately on any error

# =============================================================================
# CONFIGURE THESE
# =============================================================================

DATA_PATH="${DATA_PATH:-/path/to/C3VDv2}"
UNDISTORT_MAPS="${UNDISTORT_MAPS:-undistort_maps/}"
LOG_DIR="${LOG_DIR:-logs/c3vdv2}"
SPLIT="${SPLIT:-c3vdv2}"
GPU="${GPU:-0}"

# Optional: path to SCARED pretrained weights to warm-start Stage 1.
# Set to "" to train Stage 1 from scratch (takes longer).
SCARED_STAGE1_WEIGHTS="${SCARED_STAGE1_WEIGHTS:-./model/pretrained/scared_stage1/}" 

# Stage 1 settings (paper Section 4.2)
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
# Phase 1 (epochs 0 → ramp_epochs-1): fixed at _start values
#   High GT weight anchors depth+pose while registration/transform adapt
# Phase 2 (epochs ramp_epochs → end): linearly decay to _end values
#   Self-supervised signal now more reliable, reduce GT dependency
W_DEPTH_START="${W_DEPTH_START:-1.0}"
W_DEPTH_END="${W_DEPTH_END:-0.3}"
W_POSE_START="${W_POSE_START:-0.5}"
W_POSE_END="${W_POSE_END:-0.1}"
RAMP_EPOCHS="${RAMP_EPOCHS:-8}"

# =============================================================================
# DERIVED PATHS
# =============================================================================

S1_LOG_NAME="stage1"
S2_LOG_NAME="stage2"

# Stage 1 saves checkpoints to: $LOG_DIR/$S1_LOG_NAME/models/weights_N/
# We pass the final epoch checkpoint to Stage 2
S1_FINAL_WEIGHTS="${LOG_DIR}/${S1_LOG_NAME}/models/weights_$((S1_EPOCHS - 1))/"

# =============================================================================
# HELPERS
# =============================================================================

log() { echo -e "\n\033[1;36m>>> $*\033[0m\n"; }

check_data() {
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
        echo "Run: uv run python -m scripts.generate_splits --data_path $DATA_PATH --split $SPLIT"
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
    echo "  Purpose: learn colonoscopy tissue appearance and scope motion"
    echo "  Epochs:  $S1_EPOCHS  LR: $S1_LR  Batch: $S1_BATCH"
    echo "  Output:  $S1_FINAL_WEIGHTS"

    # Build load_weights_folder arg — only if SCARED weights exist
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
        --save_frequency      5 \
        $LOAD_WEIGHTS_ARG

    log "Stage 1 complete. Final weights: $S1_FINAL_WEIGHTS"
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

    if [ ! -d "$S1_FINAL_WEIGHTS" ]; then
        echo "ERROR: Stage 1 output not found: $S1_FINAL_WEIGHTS"
        echo "Did Stage 1 complete successfully?"
        exit 1
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
        --save_frequency      5 \
        --w_depth_start       "$W_DEPTH_START" \
        --w_depth_end         "$W_DEPTH_END" \
        --w_pose_start        "$W_POSE_START" \
        --w_pose_end          "$W_POSE_END" \
        --ramp_epochs         "$RAMP_EPOCHS"

    log "Stage 2 complete."
    echo "  Final weights: ${LOG_DIR}/${S2_LOG_NAME}/models/weights_$((S2_EPOCHS - 1))/"
}

# =============================================================================
# MAIN
# =============================================================================

log "AF-SfMLearner × C3VDv2 Fine-Tuning Pipeline"
echo "  GPU:          $GPU"
echo "  Data:         $DATA_PATH"
echo "  Logs:         $LOG_DIR"
echo "  Split:        $SPLIT"

check_data
run_stage1
run_stage2

log "ALL DONE"
echo ""
echo "  Evaluate with:"
echo "  uv run python tools/evaluate_c3vdv2.py \\"
echo "      --data_path $DATA_PATH \\"
echo "      --load_weights_folder ${LOG_DIR}/${S2_LOG_NAME}/models/weights_$((S2_EPOCHS - 1))/ \\"
echo "      --split $SPLIT"
echo ""
echo "  Monitor training:"
echo "  uv run tensorboard --logdir $LOG_DIR"