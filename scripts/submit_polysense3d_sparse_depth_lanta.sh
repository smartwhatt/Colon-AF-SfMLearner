#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -N 1 -c 8
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt200353
#SBATCH -J polysense_sparse_eval
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err

# Lanta self-submitting wrapper for PolypSense3D clinical sparse-depth eval.
#
# Run this with bash on a login/submit node:
#   bash scripts/submit_polysense3d_sparse_depth_lanta.sh

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

export PROJECT_DIR="${PROJECT_DIR:-/lustrefs/disk/project/lt200353-pcllm/swattana/Colon-AF-SfMLearner}"
export MODULE_NAME="${MODULE_NAME:-Mamba/23.11.0-0}"
export CONDA_ENV="${CONDA_ENV:-/lustrefs/disk/project/lt200353-pcllm/swattana/conda}"

export DATA_ROOT="${DATA_ROOT:-/project/lt200353-pcllm/swattana/PolypSense3D/Clinical-Dataset-For-PolypSense3D}"
export C3VDV2_LOG_DIR="${C3VDV2_LOG_DIR:-$PROJECT_DIR/logs/c3vdv2_hf}"
export WEIGHTS_FOLDER="${WEIGHTS_FOLDER:-$C3VDV2_LOG_DIR/stage2_specular_pps/models/weights_best}"
export RUN_ID="${RUN_ID:-run_d_polysense_sparse}"
export OUT_DIR="${OUT_DIR:-$PROJECT_DIR/reports/polysense3d/$RUN_ID}"

export HEIGHT="${HEIGHT:-256}"
export WIDTH="${WIDTH:-320}"
export MIN_DEPTH="${MIN_DEPTH:-0.001}"
export MAX_DEPTH="${MAX_DEPTH:-0.1}"
export SAMPLE_RADIUS_PX="${SAMPLE_RADIUS_PX:-0}"
export TOPK="${TOPK:-25}"
export DEPTH_VIS_MIN="${DEPTH_VIS_MIN:-0.001}"
export DEPTH_VIS_MAX="${DEPTH_VIS_MAX:-0.1}"
export CPU="${CPU:-0}"

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
    if [ ! -d "$DATA_ROOT" ]; then
        echo "ERROR: DATA_ROOT does not exist: $DATA_ROOT"
        missing=1
    fi
    if [ ! -d "$WEIGHTS_FOLDER" ]; then
        echo "ERROR: WEIGHTS_FOLDER does not exist: $WEIGHTS_FOLDER"
        missing=1
    else
        for file in encoder.pth depth.pth; do
            if [ ! -f "$WEIGHTS_FOLDER/$file" ]; then
                echo "ERROR: missing $file under $WEIGHTS_FOLDER"
                missing=1
            fi
        done
    fi
    if [ "$missing" -ne 0 ]; then
        exit 1
    fi
}

submit_job() {
    cd "$PROJECT_DIR"
    mkdir -p logs/slurm "$OUT_DIR"
    check_inputs

    log "Submitting PolypSense3D sparse-depth evaluation"
    echo "  Project:   $PROJECT_DIR"
    echo "  Dataset:   $DATA_ROOT"
    echo "  Weights:   $WEIGHTS_FOLDER"
    echo "  Run ID:    $RUN_ID"
    echo "  Output:    $OUT_DIR"
    echo ""

    sbatch --export=ALL "$SCRIPT_PATH"
}

run_slurm_job() {
    log "Running PolypSense3D sparse-depth evaluation on SLURM compute node"
    setup_runtime
    cd "$PROJECT_DIR"
    check_inputs
    mkdir -p "$OUT_DIR"

    echo "  Job ID:    ${SLURM_JOB_ID:-unknown}"
    echo "  Node:      ${SLURMD_NODENAME:-unknown}"
    echo "  Dataset:   $DATA_ROOT"
    echo "  Weights:   $WEIGHTS_FOLDER"
    echo "  Output:    $OUT_DIR"
    echo ""

    cmd=(
        uv run python tools/evaluate_polysense3d_sparse_depth.py
        --data_root "$DATA_ROOT"
        --weights_folder "$WEIGHTS_FOLDER"
        --run_id "$RUN_ID"
        --out_dir "$OUT_DIR"
        --height "$HEIGHT"
        --width "$WIDTH"
        --min_depth "$MIN_DEPTH"
        --max_depth "$MAX_DEPTH"
        --sample_radius_px "$SAMPLE_RADIUS_PX"
        --topk "$TOPK"
        --depth_vis_min "$DEPTH_VIS_MIN"
        --depth_vis_max "$DEPTH_VIS_MAX"
    )
    if [ "$CPU" = "1" ]; then
        cmd+=(--cpu)
    fi

    "${cmd[@]}"

    log "Done"
    echo "  Summary:   $OUT_DIR/summary.md"
    echo "  JSON:      $OUT_DIR/summary.json"
    echo "  Top-K:     $OUT_DIR/topk"
}

if [ -n "${SLURM_JOB_ID:-}" ]; then
    run_slurm_job
else
    submit_job
fi
