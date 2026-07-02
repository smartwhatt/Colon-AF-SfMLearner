#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -N 1 -c 8
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A lt200353
#SBATCH -J dav2_c3vdv2_eval
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err

# Lanta self-submitting wrapper for standalone Depth Anything V2 evaluation
# on C3VDv2 dense depth (no-median + median-scaled, the gating protocol).
#
# Run this with bash on a login/submit node:
#   bash scripts/submit_dav2_c3vdv2_lanta.sh
#
# DA-v2 is loaded from Hugging Face via transformers; no --weights_folder is
# required. The job evaluates both raw (no-median) and median-scaled depth
# metrics against the C3VDv2 HF test split.
#
# The login/submit node prefetches both the DA-v2 model snapshot and the
# C3VDv2 HF dataset into shared caches; the compute node then runs offline
# (HF_HUB_OFFLINE=1) so nothing is re-downloaded and the job works on nodes
# without internet.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

export PROJECT_DIR="${PROJECT_DIR:-/lustrefs/disk/project/lt200353-pcllm/swattana/Colon-AF-SfMLearner}"
export MODULE_NAME="${MODULE_NAME:-Mamba/23.11.0-0}"
export CONDA_ENV="${CONDA_ENV:-/lustrefs/disk/project/lt200353-pcllm/swattana/conda}"

export HF_REPO_ID="${HF_REPO_ID:-SmartWhatt/c3vdv2-SfM}"
export HF_SPLIT="${HF_SPLIT:-test}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-/scratch/lt200353-pcllm/swattana/hf_cache/datasets}"
export HF_HOME="${HF_HOME:-/scratch/lt200353-pcllm/swattana/hf_cache/home}"
export DEPTH_ANYTHING_REPO="${DEPTH_ANYTHING_REPO:-depth-anything/Depth-Anything-V2-Metric-Hypersim-Large}"
export RUN_ID="${RUN_ID:-dav2_c3vdv2}"
export OUT_DIR="${OUT_DIR:-$PROJECT_DIR/reports/c3vdv2/$RUN_ID}"

# Ready markers (mirrors submit_real_colon_hf_lanta.sh).
export DAV2_CACHE_READY_MARKER="${DAV2_CACHE_READY_MARKER:-$HF_HOME/.dav2_${DEPTH_ANYTHING_REPO//\//_}_ready}"
export C3VDV2_CACHE_READY_MARKER="${C3VDV2_CACHE_READY_MARKER:-$HF_CACHE_DIR/.c3vdv2_${HF_REPO_ID//\//_}_${HF_SPLIT}_ready}"
export PREFETCH_DAV2="${PREFETCH_DAV2:-1}"
export PREFETCH_C3VDV2="${PREFETCH_C3VDV2:-1}"

export HEIGHT="${HEIGHT:-256}"
export WIDTH="${WIDTH:-320}"

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

sync_env() {
    # Sync the uv venv on the submit node (has internet) so the offline
    # compute node never tries to reach PyPI for missing deps like
    # transformers/tokenizers.
    log "Syncing uv environment on submit node"
    cd "$PROJECT_DIR"
    uv sync
}

check_inputs() {
    local missing=0
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "ERROR: PROJECT_DIR does not exist: $PROJECT_DIR"
        missing=1
    fi
    if [ -z "$DEPTH_ANYTHING_REPO" ]; then
        echo "ERROR: DEPTH_ANYTHING_REPO must be set"
        missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        exit 1
    fi
}

prefetch_dav2_model() {
    if [ "$PREFETCH_DAV2" != "1" ]; then
        log "Skipping DA-v2 prefetch (PREFETCH_DAV2=$PREFETCH_DAV2)"
        return
    fi

    log "Prefetching DA-v2 model on submit node"
    echo "  Repo:    $DEPTH_ANYTHING_REPO"
    echo "  HF_HOME: $HF_HOME"
    mkdir -p "$HF_HOME"

    cd "$PROJECT_DIR"
    uv run python - "$DEPTH_ANYTHING_REPO" "$HF_HOME" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

repo_id, hf_home = sys.argv[1], sys.argv[2]
print(f"Downloading DA-v2 snapshot: {repo_id}", flush=True)
path = snapshot_download(
    repo_id=repo_id,
    cache_dir=os.path.join(hf_home, "hub"),
)
print(f"Cached DA-v2 snapshot at: {path}", flush=True)
PY

    touch "$DAV2_CACHE_READY_MARKER"
}

prefetch_c3vdv2_dataset() {
    if [ "$PREFETCH_C3VDV2" != "1" ]; then
        log "Skipping C3VDv2 prefetch (PREFETCH_C3VDV2=$PREFETCH_C3VDV2)"
        return
    fi

    log "Prefetching C3VDv2 HF dataset on submit node"
    echo "  Repo:      $HF_REPO_ID"
    echo "  Split:     $HF_SPLIT"
    echo "  HF cache:  $HF_CACHE_DIR"
    mkdir -p "$HF_CACHE_DIR"

    cd "$PROJECT_DIR"
    uv run python - "$HF_REPO_ID" "$HF_CACHE_DIR" "$HF_SPLIT" <<'PY'
import sys
from datasets import VerificationMode, load_dataset

repo_id, cache_dir, split = sys.argv[1:4]
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

    touch "$C3VDV2_CACHE_READY_MARKER"
}

submit_job() {
    cd "$PROJECT_DIR"
    mkdir -p logs/slurm "$OUT_DIR"
    check_inputs

    log "Submitting DA-v2 C3VDv2 dense-depth evaluation"
    echo "  Project:   $PROJECT_DIR"
    echo "  HF repo:   $HF_REPO_ID (split=$HF_SPLIT)"
    echo "  DA-v2 repo: $DEPTH_ANYTHING_REPO"
    echo "  Run ID:    $RUN_ID"
    echo "  Output:    $OUT_DIR"
    echo ""

    sbatch --export=ALL "$SCRIPT_PATH"
}

run_eval() {
    local median_flag=("$1")
    uv run python tools/evaluate_c3vdv2.py \
        --data_source hf \
        --hf_repo_id "$HF_REPO_ID" \
        --hf_split "$HF_SPLIT" \
        --hf_cache_dir "$HF_CACHE_DIR" \
        --depth_backend depth_anything \
        --depth_anything_repo "$DEPTH_ANYTHING_REPO" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --run_id "${RUN_ID}$2" \
        --metrics_out "$OUT_DIR/metrics$2.jsonl" \
        --summary_out "$OUT_DIR/summary$2.md" \
        "${median_flag[@]}"
}

run_slurm_job() {
    log "Running DA-v2 C3VDv2 dense-depth evaluation on SLURM compute node"
    setup_runtime
    cd "$PROJECT_DIR"
    check_inputs
    mkdir -p "$OUT_DIR"

    if [ ! -f "$DAV2_CACHE_READY_MARKER" ]; then
        echo "ERROR: DA-v2 cache marker not found: $DAV2_CACHE_READY_MARKER"
        echo "Run this script with bash on a login/submit node before using sbatch directly."
        exit 1
    fi
    if [ ! -f "$C3VDV2_CACHE_READY_MARKER" ]; then
        echo "ERROR: C3VDv2 cache marker not found: $C3VDV2_CACHE_READY_MARKER"
        echo "Run this script with bash on a login/submit node before using sbatch directly."
        exit 1
    fi

    # Run offline so the compute node never re-downloads anything.
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export UV_OFFLINE=1
    export HF_CACHE_DIR
    export HF_HOME

    echo "  Job ID:    ${SLURM_JOB_ID:-unknown}"
    echo "  Node:      ${SLURMD_NODENAME:-unknown}"
    echo "  HF repo:   $HF_REPO_ID (split=$HF_SPLIT)"
    echo "  DA-v2 repo: $DEPTH_ANYTHING_REPO"
    echo "  HF_HOME:   $HF_HOME"
    echo "  HF cache:  $HF_CACHE_DIR"
    echo "  Offline:   HF_HUB_OFFLINE=$HF_HUB_OFFLINE HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
    echo "  Output:    $OUT_DIR"
    echo ""

    # No-median (absolute metric depth — the gating check for scale drift).
    log "No-median evaluation"
    run_eval "" "_nomedian"

    # Median-scaled (relative shape — the gating check for relative depth).
    log "Median-scaled evaluation"
    run_eval "--median_scaling" "_median"

    log "Done"
    echo "  No-median:  $OUT_DIR/summary_nomedian.md"
    echo "  Median:     $OUT_DIR/summary_median.md"
}

if [ -n "${SLURM_JOB_ID:-}" ]; then
    run_slurm_job
else
    setup_runtime
    sync_env
    prefetch_dav2_model
    prefetch_c3vdv2_dataset
    submit_job
fi
