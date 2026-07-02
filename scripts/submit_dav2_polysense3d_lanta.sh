#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -N 1 -c 8
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A lt200353
#SBATCH -J dav2_polysense_eval
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err

# Lanta self-submitting wrapper for standalone Depth Anything V2 evaluation
# on PolypSense3D clinical sparse depth (the "DA-v2 vs paper" comparison).
#
# Run this with bash on a login/submit node:
#   bash scripts/submit_dav2_polysense3d_lanta.sh
#
# DA-v2 is loaded from Hugging Face via transformers; no --weights_folder is
# required. Override DEPTH_ANYTHING_REPO to compare DA-v2 variants.
#
# The login/submit node prefetches the DA-v2 model snapshot into a shared HF
# cache; the compute node then runs offline (HF_HUB_OFFLINE=1) so the model
# is never re-downloaded and the job works on nodes without internet.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

export PROJECT_DIR="${PROJECT_DIR:-/lustrefs/disk/project/lt200353-pcllm/swattana/Colon-AF-SfMLearner}"
export MODULE_NAME="${MODULE_NAME:-Mamba/23.11.0-0}"
export CONDA_ENV="${CONDA_ENV:-/lustrefs/disk/project/lt200353-pcllm/swattana/conda}"

export DATA_ROOT="${DATA_ROOT:-/project/lt200353-pcllm/swattana/PolypSense3D/Clinical-Dataset-For-PolypSense3D}"
export DEPTH_ANYTHING_VARIANT="${DEPTH_ANYTHING_VARIANT:-metric_hypersim}"
export DEPTH_ANYTHING_ENCODER_SIZE="${DEPTH_ANYTHING_ENCODER_SIZE:-vitl}"
# Default repo per variant (override DEPTH_ANYTHING_REPO to compare variants).
if [ -z "${DEPTH_ANYTHING_REPO:-}" ]; then
    case "$DEPTH_ANYTHING_VARIANT" in
        metric_hypersim)
            case "$DEPTH_ANYTHING_ENCODER_SIZE" in
                vits) DEPTH_ANYTHING_REPO="depth-anything/Depth-Anything-V2-Metric-Hypersim-Small" ;;
                vitb) DEPTH_ANYTHING_REPO="depth-anything/Depth-Anything-V2-Metric-Hypersim-Base" ;;
                *)    DEPTH_ANYTHING_REPO="depth-anything/Depth-Anything-V2-Metric-Hypersim-Large" ;;
            esac
            ;;
        relative_hf)
            case "$DEPTH_ANYTHING_ENCODER_SIZE" in
                vits) DEPTH_ANYTHING_REPO="depth-anything/Depth-Anything-V2-Small-hf" ;;
                vitb) DEPTH_ANYTHING_REPO="depth-anything/Depth-Anything-V2-Base-hf" ;;
                *)    DEPTH_ANYTHING_REPO="depth-anything/Depth-Anything-V2-Large-hf" ;;
            esac
            ;;
        *)
            echo "ERROR: DEPTH_ANYTHING_VARIANT must be metric_hypersim or relative_hf"
            exit 1
            ;;
    esac
fi
export DEPTH_ANYTHING_REPO
export RUN_ID="${RUN_ID:-dav2_${DEPTH_ANYTHING_VARIANT}_polysense_sparse}"
export OUT_DIR="${OUT_DIR:-$PROJECT_DIR/reports/polysense3d/$RUN_ID}"

# Shared HF cache (mirrors submit_real_colon_hf_lanta.sh layout).
export HF_HOME="${HF_HOME:-/scratch/lt200353-pcllm/swattana/hf_cache/home}"
export DAV2_CACHE_READY_MARKER="${DAV2_CACHE_READY_MARKER:-$HF_HOME/.dav2_${DEPTH_ANYTHING_REPO//\//_}_ready}"
export PREFETCH_DAV2="${PREFETCH_DAV2:-1}"

export HEIGHT="${HEIGHT:-256}"
export WIDTH="${WIDTH:-320}"
export MIN_DEPTH="${MIN_DEPTH:-0.001}"
export MAX_DEPTH="${MAX_DEPTH:-0.1}"
export SAMPLE_RADIUS_PX="${SAMPLE_RADIUS_PX:-0}"
export TOPK="${TOPK:-25}"
# Median scaling is required for the relative_hf variant (uncalibrated scale)
# and useful as a diagnostic for metric_hypersim. Set MEDIAN_SCALING=1 to enable.
export MEDIAN_SCALING="${MEDIAN_SCALING:-0}"
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
    if [ ! -d "$DATA_ROOT" ]; then
        echo "ERROR: DATA_ROOT does not exist: $DATA_ROOT"
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
    echo "  Variant: $DEPTH_ANYTHING_VARIANT"
    echo "  Size:    $DEPTH_ANYTHING_ENCODER_SIZE"
    echo "  HF_HOME: $HF_HOME"
    mkdir -p "$HF_HOME"

    cd "$PROJECT_DIR"
    uv run python - "$DEPTH_ANYTHING_REPO" "$DEPTH_ANYTHING_VARIANT" "$DEPTH_ANYTHING_ENCODER_SIZE" <<'PY'
import os
import sys

repo_id, variant, encoder_size = sys.argv[1:4]
print(f"Pre-caching DA-v2: {repo_id} variant={variant} size={encoder_size}", flush=True)

if variant == "metric_hypersim":
    # The official metric .pth: download via hf_hub_download so it lands in
    # HF_HOME/hub where the adapter's hf_hub_download(repo_id, filename)
    # call finds it offline.
    from huggingface_hub import hf_hub_download
    fname = f"depth_anything_v2_metric_hypersim_{encoder_size}.pth"
    path = hf_hub_download(repo_id=repo_id, filename=fname)
    print(f"Cached metric .pth at: {path}", flush=True)
else:  # relative_hf
    # transformers -hf: warm via from_pretrained so the safetensors +
    # config land where the offline from_pretrained call expects.
    from transformers import AutoModelForDepthEstimation
    _ = AutoModelForDepthEstimation.from_pretrained(repo_id)
    print(f"Cached relative-hf model for offline reuse: {repo_id}", flush=True)
PY

    touch "$DAV2_CACHE_READY_MARKER"
}

submit_job() {
    cd "$PROJECT_DIR"
    mkdir -p logs/slurm "$OUT_DIR"
    check_inputs

    log "Submitting DA-v2 PolypSense3D sparse-depth evaluation"
    echo "  Project:   $PROJECT_DIR"
    echo "  Dataset:   $DATA_ROOT"
    echo "  DA-v2 repo: $DEPTH_ANYTHING_REPO"
    echo "  Run ID:    $RUN_ID"
    echo "  Output:    $OUT_DIR"
    echo ""

    sbatch --export=ALL "$SCRIPT_PATH"
}

run_slurm_job() {
    log "Running DA-v2 PolypSense3D sparse-depth evaluation on SLURM compute node"
    setup_runtime
    cd "$PROJECT_DIR"
    check_inputs
    mkdir -p "$OUT_DIR"

    if [ ! -f "$DAV2_CACHE_READY_MARKER" ]; then
        echo "ERROR: DA-v2 cache marker not found: $DAV2_CACHE_READY_MARKER"
        echo "Run this script with bash on a login/submit node before using sbatch directly."
        exit 1
    fi

    # Run offline so the compute node never re-downloads the model.
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export UV_OFFLINE=1
    export HF_HOME

    echo "  Job ID:    ${SLURM_JOB_ID:-unknown}"
    echo "  Node:      ${SLURMD_NODENAME:-unknown}"
    echo "  Dataset:   $DATA_ROOT"
    echo "  DA-v2 repo: $DEPTH_ANYTHING_REPO"
    echo "  HF_HOME:   $HF_HOME"
    echo "  Offline:   HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
    echo "  Output:    $OUT_DIR"
    echo ""

    cmd=(
        uv run python tools/evaluate_polysense3d_sparse_depth.py
        --data_root "$DATA_ROOT"
        --depth_backend depth_anything
        --depth_anything_repo "$DEPTH_ANYTHING_REPO"
        --depth_anything_variant "$DEPTH_ANYTHING_VARIANT"
        --depth_anything_encoder_size "$DEPTH_ANYTHING_ENCODER_SIZE"
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
    if [ "$MEDIAN_SCALING" = "1" ]; then
        cmd+=(--median_scaling)
    fi
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
    setup_runtime
    sync_env
    prefetch_dav2_model
    submit_job
fi
