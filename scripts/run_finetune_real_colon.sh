#!/usr/bin/env bash
# =============================================================================
# run_finetune_real_colon.sh
# =============================================================================
# Self-supervised Stage 2 fine-tuning/adaptation on REAL-Colon.
#
# This reuses a C3VDv2 Stage 2 checkpoint as both:
#   1. The frozen registration source for position_encoder + position.
#   2. The trainable warm start for depth, pose, transform, and SpecularNet.
#
# The default depth range intentionally matches the C3VDv2 training/evaluation
# scripts because the starting checkpoint was trained in metres over 0-100 mm.
#
# Usage:
#   DATA_SOURCE=hf HF_REPO_ID=SmartWhatt/real-colon-SfM \
#   LOAD_WEIGHTS_FOLDER=logs/c3vdv2_hf/stage2_specular_pps/models/weights_best \
#   bash scripts/run_finetune_real_colon.sh
#
# For a short diagnostic:
#   HF_STREAMING=1 STEPS_PER_EPOCH=500 EPOCHS=1 SAVE_FREQUENCY=1 \
#   bash scripts/run_finetune_real_colon.sh
#
# For frozen-depth Run B adaptation:
#   ADAPTATION_PRESET=pose_transform bash scripts/run_finetune_real_colon.sh
#
# For frozen-depth adaptation with teacher relative-depth anchoring:
#   ADAPTATION_PRESET=teacher_pose_transform bash scripts/run_finetune_real_colon.sh
# =============================================================================

set -e

# =============================================================================
# CONFIGURE THESE
# =============================================================================

DATA_PATH="${DATA_PATH:-/path/to/REAL-Colon}"
LOG_DIR="${LOG_DIR:-logs/real_colon}"
SPLIT="${SPLIT:-real_colon}"
GPU="${GPU:-0}"
MODEL_NAME="${MODEL_NAME:-stage2_adapt}"
DATA_SOURCE="${DATA_SOURCE:-local}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_STREAMING="${HF_STREAMING:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
HF_SHUFFLE_BUFFER="${HF_SHUFFLE_BUFFER:-10000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"

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
LOG_FREQUENCY="${LOG_FREQUENCY:-200}"
WEIGHTS_INIT="${WEIGHTS_INIT:-pretrained}"

# Keep REAL-Colon adaptation numerically aligned with C3VDv2 checkpoints.
MIN_DEPTH="${MIN_DEPTH:-0.001}"
MAX_DEPTH="${MAX_DEPTH:-0.1}"

USE_SPECULAR_NET="${USE_SPECULAR_NET:-0}"
SPECULAR_LOSS_WEIGHT="${SPECULAR_LOSS_WEIGHT:-0.05}"
SPECULAR_THRESHOLD="${SPECULAR_THRESHOLD:-0.98}"

ADAPTATION_PRESET="${ADAPTATION_PRESET:-full}"
TRAINABLE_MODULES_ENV="${TRAINABLE_MODULES:-}"
TRAINABLE_MODULES=()
case "$ADAPTATION_PRESET" in
    full)
        ;;
    pose_transform)
        TRAINABLE_MODULES=(pose_encoder pose transform_encoder transform)
        ;;
    teacher_pose_transform)
        TRAINABLE_MODULES=(pose_encoder pose transform_encoder transform)
        : "${TEACHER_WEIGHTS_FOLDER:=$LOAD_WEIGHTS_FOLDER}"
        : "${TEACHER_CONSISTENCY_WEIGHT:=0.5}"
        ;;
    *)
        echo "ERROR: ADAPTATION_PRESET must be one of: full, pose_transform, teacher_pose_transform"
        exit 1
        ;;
esac
if [ -n "$TRAINABLE_MODULES_ENV" ]; then
    read -r -a TRAINABLE_MODULES <<< "$TRAINABLE_MODULES_ENV"
fi
TEACHER_WEIGHTS_FOLDER="${TEACHER_WEIGHTS_FOLDER:-}"
TEACHER_CONSISTENCY_WEIGHT="${TEACHER_CONSISTENCY_WEIGHT:-0.0}"
TEACHER_CONSISTENCY_MODE="${TEACHER_CONSISTENCY_MODE:-relative_log_median}"

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
fi

check_data() {
    if [ "$DATA_SOURCE" = "hf" ]; then
        if [ -z "$HF_REPO_ID" ]; then
            echo "ERROR: HF_REPO_ID is required when DATA_SOURCE=hf"
            exit 1
        fi
    else
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
    fi

    if [ -z "$LOAD_WEIGHTS_FOLDER" ]; then
        echo "ERROR: LOAD_WEIGHTS_FOLDER is required for the frozen registration net."
        exit 1
    fi
    if [ ! -d "$LOAD_WEIGHTS_FOLDER" ]; then
        echo "ERROR: LOAD_WEIGHTS_FOLDER does not exist: $LOAD_WEIGHTS_FOLDER"
        exit 1
    fi
    if [ -n "$LOAD_TRAINABLE_WEIGHTS_FOLDER" ] && [ ! -d "$LOAD_TRAINABLE_WEIGHTS_FOLDER" ]; then
        echo "ERROR: LOAD_TRAINABLE_WEIGHTS_FOLDER does not exist: $LOAD_TRAINABLE_WEIGHTS_FOLDER"
        exit 1
    fi
    if [ -n "$TEACHER_WEIGHTS_FOLDER" ] && [ ! -d "$TEACHER_WEIGHTS_FOLDER" ]; then
        echo "ERROR: TEACHER_WEIGHTS_FOLDER does not exist: $TEACHER_WEIGHTS_FOLDER"
        exit 1
    fi
}

run_stage2_adaptation() {
    log "REAL-Colon self-supervised Stage 2 fine-tuning"
    echo "  Source:       $DATA_SOURCE"
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
    echo "  Split:        $SPLIT"
    echo "  Output:       ${LOG_DIR}/${MODEL_NAME}"
    echo "  Frozen:       position_encoder + position from $LOAD_WEIGHTS_FOLDER"
    echo "  Warm start:   ${LOAD_TRAINABLE_WEIGHTS_FOLDER:-none}"
    echo "  Preset:       $ADAPTATION_PRESET"
    if [ "${#TRAINABLE_MODULES[@]}" -gt 0 ]; then
        echo "  Trainable:    ${TRAINABLE_MODULES[*]}"
    else
        echo "  Trainable:    default Stage 2 modules"
    fi
    echo "  Epochs:       $EPOCHS  LR: $LR  Batch: $BATCH"
    echo "  Weights init: $WEIGHTS_INIT"
    echo "  Depth range:  $MIN_DEPTH to $MAX_DEPTH"
    if [ -n "$STEPS_PER_EPOCH" ]; then
        echo "  Step cap:     $STEPS_PER_EPOCH batches/epoch"
    fi
    if [ "$USE_SPECULAR_NET" = "1" ]; then
        echo "  SpecularNet:  ON  (weight=$SPECULAR_LOSS_WEIGHT threshold=$SPECULAR_THRESHOLD)"
    else
        echo "  SpecularNet:  OFF"
    fi
    if [ -n "$TEACHER_WEIGHTS_FOLDER" ]; then
        echo "  Teacher:      $TEACHER_WEIGHTS_FOLDER"
        echo "  Teacher loss: $TEACHER_CONSISTENCY_MODE weight=$TEACHER_CONSISTENCY_WEIGHT"
    else
        echo "  Teacher:      OFF"
    fi

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
        --weights_init "$WEIGHTS_INIT"
        --height "$HEIGHT"
        --width "$WIDTH"
        --scales 0 1 2 3
        --min_depth "$MIN_DEPTH"
        --max_depth "$MAX_DEPTH"
        --save_frequency "$SAVE_FREQUENCY"
        --log_frequency "$LOG_FREQUENCY"
        "${HF_ARGS[@]}"
    )

    if [ -n "$LOAD_TRAINABLE_WEIGHTS_FOLDER" ]; then
        cmd+=(--load_trainable_weights_folder "$LOAD_TRAINABLE_WEIGHTS_FOLDER")
    fi

    if [ "${#TRAINABLE_MODULES[@]}" -gt 0 ]; then
        cmd+=(--trainable_modules "${TRAINABLE_MODULES[@]}")
    fi

    if [ -n "$TEACHER_WEIGHTS_FOLDER" ]; then
        cmd+=(
            --teacher_weights_folder "$TEACHER_WEIGHTS_FOLDER"
            --teacher_consistency_weight "$TEACHER_CONSISTENCY_WEIGHT"
            --teacher_consistency_mode "$TEACHER_CONSISTENCY_MODE"
        )
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

log "AF-SfMLearner x REAL-Colon Fine-Tuning Pipeline"
check_data
mkdir -p "$LOG_DIR"
run_stage2_adaptation

FINAL_WEIGHTS="${LOG_DIR}/${MODEL_NAME}/models/weights_$((EPOCHS - 1))/"
BEST_WEIGHTS="${LOG_DIR}/${MODEL_NAME}/models/weights_best/"

log "Done"
echo "  Final weights: $FINAL_WEIGHTS"
echo "  Best weights:  $BEST_WEIGHTS"
echo ""
echo "  C3VDv2 no-collapse check (no median scaling):"
if [ "$USE_SPECULAR_NET" = "1" ]; then
    echo "  uv run python tools/evaluate_c3vdv2.py \\"
    echo "      --data_source hf \\"
    echo "      --hf_repo_id SmartWhatt/c3vdv2-SfM \\"
    echo "      --hf_split test \\"
    echo "      --weights_folder $FINAL_WEIGHTS \\"
    echo "      --load_specular_weights"
else
    echo "  uv run python tools/evaluate_c3vdv2.py \\"
    echo "      --data_source hf \\"
    echo "      --hf_repo_id SmartWhatt/c3vdv2-SfM \\"
    echo "      --hf_split test \\"
    echo "      --weights_folder $FINAL_WEIGHTS"
fi
echo ""
echo "  C3VDv2 relative-depth check (median scaling on):"
if [ "$USE_SPECULAR_NET" = "1" ]; then
    echo "  uv run python tools/evaluate_c3vdv2.py \\"
    echo "      --data_source hf \\"
    echo "      --hf_repo_id SmartWhatt/c3vdv2-SfM \\"
    echo "      --hf_split test \\"
    echo "      --weights_folder $FINAL_WEIGHTS \\"
    echo "      --median_scaling \\"
    echo "      --load_specular_weights"
else
    echo "  uv run python tools/evaluate_c3vdv2.py \\"
    echo "      --data_source hf \\"
    echo "      --hf_repo_id SmartWhatt/c3vdv2-SfM \\"
    echo "      --hf_split test \\"
    echo "      --weights_folder $FINAL_WEIGHTS \\"
    echo "      --median_scaling"
fi
echo ""
echo "  Monitor training:"
echo "  uv run tensorboard --logdir $LOG_DIR"
