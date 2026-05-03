#!/usr/bin/env bash
# =============================================================================
# run_train_real_colon.sh
# =============================================================================
# Self-supervised Stage 2 adaptation on REAL-Colon.
#
# The frozen registration network is loaded from LOAD_WEIGHTS_FOLDER. If that
# folder also contains Stage 2 weights, depth/pose/transform are warm-started
# without freezing through --load_trainable_weights_folder.
#
# Usage:
#   DATA_PATH=/data/REAL-Colon \
#   LOAD_WEIGHTS_FOLDER=logs/c3vdv2/stage2/models/weights_19 \
#   bash scripts/run_train_real_colon.sh
# =============================================================================

set -e

DATA_PATH="${DATA_PATH:-/path/to/REAL-Colon}"
LOG_DIR="${LOG_DIR:-logs/real_colon}"
SPLIT="${SPLIT:-real_colon}"
GPU="${GPU:-0}"
MODEL_NAME="${MODEL_NAME:-stage2_adapt}"

# Required: source for frozen position_encoder + position.
LOAD_WEIGHTS_FOLDER="${LOAD_WEIGHTS_FOLDER:-}"

# Optional: warm-start trainable Stage 2 networks. Defaults to the same folder;
# missing model files are skipped by the trainer, so Stage 1-only folders work.
LOAD_TRAINABLE_WEIGHTS_FOLDER="${LOAD_TRAINABLE_WEIGHTS_FOLDER:-$LOAD_WEIGHTS_FOLDER}"

EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-12}"
LR_STEP="${LR_STEP:-10}"
NUM_WORKERS="${NUM_WORKERS:-12}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-320}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-5}"

USE_SPECULAR_NET="${USE_SPECULAR_NET:-0}"
SPECULAR_LOSS_WEIGHT="${SPECULAR_LOSS_WEIGHT:-0.05}"
SPECULAR_THRESHOLD="${SPECULAR_THRESHOLD:-0.98}"

log() { echo -e "\n\033[1;36m>>> $*\033[0m\n"; }

check_data() {
    if [ ! -d "$DATA_PATH" ]; then
        echo "ERROR: DATA_PATH does not exist: $DATA_PATH"
        echo "Set DATA_PATH=/path/to/REAL-Colon before running."
        exit 1
    fi
    if [ ! -f "$DATA_PATH/video_info.csv" ]; then
        echo "ERROR: video_info.csv not found under DATA_PATH: $DATA_PATH"
        exit 1
    fi
    if [ ! -f "splits/${SPLIT}/train_files.txt" ]; then
        echo "ERROR: Split file not found: splits/${SPLIT}/train_files.txt"
        echo "Run: uv run python scripts/generate_splits_realcolon.py --data_root \"$DATA_PATH\" --output_dir \"splits/${SPLIT}\""
        exit 1
    fi
    if [ ! -f "splits/${SPLIT}/val_files.txt" ]; then
        echo "ERROR: Split file not found: splits/${SPLIT}/val_files.txt"
        echo "Run: uv run python scripts/generate_splits_realcolon.py --data_root \"$DATA_PATH\" --output_dir \"splits/${SPLIT}\""
        exit 1
    fi
    if [ -z "$LOAD_WEIGHTS_FOLDER" ]; then
        echo "ERROR: LOAD_WEIGHTS_FOLDER is required for the frozen registration net."
        exit 1
    fi
    if [ ! -d "$LOAD_WEIGHTS_FOLDER" ]; then
        echo "ERROR: LOAD_WEIGHTS_FOLDER does not exist: $LOAD_WEIGHTS_FOLDER"
        exit 1
    fi
}

run_stage2_adaptation() {
    log "REAL-Colon self-supervised Stage 2 adaptation"
    echo "  Data:       $DATA_PATH"
    echo "  Split:      $SPLIT"
    echo "  Output:     ${LOG_DIR}/${MODEL_NAME}"
    echo "  Frozen:     position_encoder + position from $LOAD_WEIGHTS_FOLDER"
    echo "  Warm start: ${LOAD_TRAINABLE_WEIGHTS_FOLDER:-none}"
    echo "  Epochs:     $EPOCHS  LR: $LR  Batch: $BATCH"

    cmd=(
        uv run python train.py
        --mode stage2
        --dataset real_colon
        --data_path "$DATA_PATH"
        --split "$SPLIT"
        --log_dir "$LOG_DIR"
        --model_name "$MODEL_NAME"
        --load_weights_folder "$LOAD_WEIGHTS_FOLDER"
        --num_epochs "$EPOCHS"
        --learning_rate "$LR"
        --batch_size "$BATCH"
        --scheduler_step_size "$LR_STEP"
        --num_workers "$NUM_WORKERS"
        --height "$HEIGHT"
        --width "$WIDTH"
        --scales 0 1 2 3
        --save_frequency "$SAVE_FREQUENCY"
    )

    if [ -n "$LOAD_TRAINABLE_WEIGHTS_FOLDER" ]; then
        cmd+=(--load_trainable_weights_folder "$LOAD_TRAINABLE_WEIGHTS_FOLDER")
    fi

    if [ "$USE_SPECULAR_NET" = "1" ]; then
        cmd+=(
            --use_specular_net
            --specular_loss_weight "$SPECULAR_LOSS_WEIGHT"
            --specular_threshold "$SPECULAR_THRESHOLD"
        )
    fi

    CUDA_VISIBLE_DEVICES=$GPU "${cmd[@]}"
}

log "AF-SfMLearner x REAL-Colon"
check_data
run_stage2_adaptation

log "Done"
echo "  Final weights: ${LOG_DIR}/${MODEL_NAME}/models/weights_$((EPOCHS - 1))/"
echo "  Monitor: uv run tensorboard --logdir $LOG_DIR"
