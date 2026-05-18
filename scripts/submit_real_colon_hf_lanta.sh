#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -A lt200353
#SBATCH -J real_colon_hf
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err

# Lanta self-submitting wrapper for HF-backed REAL-Colon fine-tuning.
#
# Run this with bash on a login/submit node, not with sbatch directly:
#   bash scripts/submit_real_colon_hf_lanta.sh
#
# The login/submit node prefetches the Hugging Face dataset into a shared cache.
# The SLURM job then runs with HF offline flags and DATA_SOURCE=hf,
# HF_STREAMING=0, so compute nodes do not need internet access.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# =============================================================================
# LANTA PATHS AND RUNTIME
# =============================================================================

export PROJECT_DIR="${PROJECT_DIR:-/lustrefs/disk/project/lt200353-pcllm/swattana/Colon-AF-SfMLearner}"
export MODULE_NAME="${MODULE_NAME:-Mamba/23.11.0-0}"
export CONDA_ENV="${CONDA_ENV:-/lustrefs/disk/project/lt200353-pcllm/swattana/conda}"

export HF_CACHE_DIR="${HF_CACHE_DIR:-/scratch/lt200353-pcllm/swattana/hf_cache/datasets}"
export HF_HOME="${HF_HOME:-/scratch/lt200353-pcllm/swattana/hf_cache/home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_DIR}"
export CACHE_READY_MARKER="${CACHE_READY_MARKER:-$HF_CACHE_DIR/.real_colon_sfm_ready}"

# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

export DATA_SOURCE="hf"
export HF_REPO_ID="${HF_REPO_ID:-SmartWhatt/real-colon-SfM}"
export HF_STREAMING="0"
export HF_SHUFFLE_BUFFER="${HF_SHUFFLE_BUFFER:-10000}"
if [ -z "${STEPS_PER_EPOCH+x}" ]; then
    export STEPS_PER_EPOCH="500"
else
    export STEPS_PER_EPOCH
fi

export LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/real_colon_hf}"
export GPU="${GPU:-0}"

# run_d = SpecularNet + PPS best aggregate C3VDv2 checkpoint.
# run_b = SpecularNet-only fallback from the stratified audit.
export START_RUN="${START_RUN:-run_d}"
export C3VDV2_LOG_DIR="${C3VDV2_LOG_DIR:-$PROJECT_DIR/logs/c3vdv2_hf}"

case "$START_RUN" in
    run_d)
        DEFAULT_START_WEIGHTS="$C3VDV2_LOG_DIR/stage2_specular_pps/models/weights_best/"
        DEFAULT_MODEL_NAME="realcolon_run_d_diag"
        DEFAULT_USE_SPECULAR_NET="1"
        ;;
    run_b)
        DEFAULT_START_WEIGHTS="$C3VDV2_LOG_DIR/stage2_specular/models/weights_best/"
        DEFAULT_MODEL_NAME="realcolon_run_b_diag"
        DEFAULT_USE_SPECULAR_NET="1"
        ;;
    run_a)
        DEFAULT_START_WEIGHTS="$C3VDV2_LOG_DIR/stage2/models/weights_best/"
        DEFAULT_MODEL_NAME="realcolon_run_a_diag"
        DEFAULT_USE_SPECULAR_NET="0"
        ;;
    *)
        echo "ERROR: START_RUN must be one of: run_d, run_b, run_a"
        exit 1
        ;;
esac

export LOAD_WEIGHTS_FOLDER="${LOAD_WEIGHTS_FOLDER:-$DEFAULT_START_WEIGHTS}"
export LOAD_TRAINABLE_WEIGHTS_FOLDER="${LOAD_TRAINABLE_WEIGHTS_FOLDER:-$LOAD_WEIGHTS_FOLDER}"
export MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"

export EPOCHS="${EPOCHS:-1}"
export LR="${LR:-1e-4}"
export BATCH="${BATCH:-12}"
export LR_STEP="${LR_STEP:-10}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export SAVE_FREQUENCY="${SAVE_FREQUENCY:-1}"
export LOG_FREQUENCY="${LOG_FREQUENCY:-50}"

export MIN_DEPTH="${MIN_DEPTH:-0.001}"
export MAX_DEPTH="${MAX_DEPTH:-0.1}"

export USE_SPECULAR_NET="${USE_SPECULAR_NET:-$DEFAULT_USE_SPECULAR_NET}"
export SPECULAR_LOSS_WEIGHT="${SPECULAR_LOSS_WEIGHT:-0.05}"
export SPECULAR_THRESHOLD="${SPECULAR_THRESHOLD:-0.98}"

# Splits required by training. Add "test" if you also want it cached for later
# ad-hoc inspection from compute nodes.
export PREFETCH_SPLITS="${PREFETCH_SPLITS:-train val}"
export PREFETCH_HF="${PREFETCH_HF:-1}"

log() { echo -e "\n\033[1;36m>>> $*\033[0m\n"; }

setup_runtime() {
    if [ -n "$MODULE_NAME" ]; then
        if command -v module >/dev/null 2>&1; then
            module load "$MODULE_NAME"
        else
            echo "WARNING: module command not found; assuming runtime is already configured."
        fi
    fi

    if [ -n "$CONDA_ENV" ]; then
        conda activate "$CONDA_ENV"
    fi
}

prefetch_hf_dataset() {
    if [ "$PREFETCH_HF" != "1" ]; then
        log "Skipping HF prefetch (PREFETCH_HF=$PREFETCH_HF)"
        return
    fi

    log "Prefetching Hugging Face dataset on submit node"
    echo "  Repo:        $HF_REPO_ID"
    echo "  Splits:      $PREFETCH_SPLITS"
    echo "  HF cache:    $HF_CACHE_DIR"
    echo "  HF_HOME:     $HF_HOME"
    mkdir -p "$HF_CACHE_DIR" "$HF_HOME"

    cd "$PROJECT_DIR"
    uv run python - "$HF_REPO_ID" "$HF_CACHE_DIR" "$PREFETCH_SPLITS" <<'PY'
import sys
from datasets import VerificationMode, load_dataset

repo_id, cache_dir, split_text = sys.argv[1:4]
for split in split_text.split():
    print(f"Loading {repo_id} split={split} into cache={cache_dir}", flush=True)
    ds = load_dataset(
        repo_id,
        split=split,
        streaming=False,
        cache_dir=cache_dir,
        verification_mode=VerificationMode.NO_CHECKS,
    )
    try:
        n_rows = len(ds)
    except TypeError:
        n_rows = "unknown"
    if n_rows != 0:
        _ = ds[0]
    print(f"Cached {repo_id} split={split} rows={n_rows}", flush=True)
PY

    touch "$CACHE_READY_MARKER"
}

submit_job() {
    cd "$PROJECT_DIR"
    mkdir -p logs/slurm "$LOG_DIR"

    log "Submitting REAL-Colon SLURM job"
    echo "  Project:      $PROJECT_DIR"
    echo "  Start run:    $START_RUN"
    echo "  Start ckpt:   $LOAD_WEIGHTS_FOLDER"
    echo "  Run dir:      $LOG_DIR/$MODEL_NAME"
    echo "  Cache marker: $CACHE_READY_MARKER"
    echo ""

    sbatch --export=ALL "$SCRIPT_PATH"
}

run_slurm_job() {
    log "Running REAL-Colon HF fine-tuning on SLURM compute node"
    setup_runtime
    cd "$PROJECT_DIR"

    if [ ! -f "$CACHE_READY_MARKER" ]; then
        echo "ERROR: HF cache marker not found: $CACHE_READY_MARKER"
        echo "Run this script with bash on a login/submit node before using sbatch directly."
        exit 1
    fi

    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export DATA_SOURCE=hf
    export HF_STREAMING=0
    export HF_CACHE_DIR
    export HF_HOME
    export HF_DATASETS_CACHE

    echo "  Job ID:       ${SLURM_JOB_ID:-unknown}"
    echo "  Node:         ${SLURMD_NODENAME:-unknown}"
    echo "  HF repo:      $HF_REPO_ID"
    echo "  HF cache:     $HF_CACHE_DIR"
    echo "  Offline:      HF_HUB_OFFLINE=$HF_HUB_OFFLINE HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
    echo "  Steps/epoch:  ${STEPS_PER_EPOCH:-full split}"
    echo ""

    bash scripts/run_finetune_real_colon.sh
}

if [ -n "${SLURM_JOB_ID:-}" ]; then
    run_slurm_job
else
    setup_runtime
    prefetch_hf_dataset
    submit_job
fi
