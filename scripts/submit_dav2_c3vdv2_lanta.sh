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

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

export PROJECT_DIR="${PROJECT_DIR:-/lustrefs/disk/project/lt200353-pcllm/swattana/Colon-AF-SfMLearner}"
export MODULE_NAME="${MODULE_NAME:-Mamba/23.11.0-0}"
export CONDA_ENV="${CONDA_ENV:-/lustrefs/disk/project/lt200353-pcllm/swattana/conda}"

export HF_REPO_ID="${HF_REPO_ID:-SmartWhatt/c3vdv2-SfM}"
export HF_SPLIT="${HF_SPLIT:-test}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$PROJECT_DIR/.hf_cache/datasets}"
export DEPTH_ANYTHING_REPO="${DEPTH_ANYTHING_REPO:-depth-anything/Depth-Anything-V2-Metric-Hypersim-Large}"
export RUN_ID="${RUN_ID:-dav2_c3vdv2}"
export OUT_DIR="${OUT_DIR:-$PROJECT_DIR/reports/c3vdv2/$RUN_ID}"

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

    echo "  Job ID:    ${SLURM_JOB_ID:-unknown}"
    echo "  Node:      ${SLURMD_NODENAME:-unknown}"
    echo "  HF repo:   $HF_REPO_ID (split=$HF_SPLIT)"
    echo "  DA-v2 repo: $DEPTH_ANYTHING_REPO"
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
    submit_job
fi
