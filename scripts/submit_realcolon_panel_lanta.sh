#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -N 1 -c 8
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -A lt200353
#SBATCH -J realcolon_panel
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err

# Lanta self-submitting wrapper for the REALColon qualitative depth panel.
#
# Run this with bash on a login/submit node, not with sbatch directly:
#   bash scripts/submit_realcolon_panel_lanta.sh
#
# The login/submit node prefetches the REALColon HF split into a shared cache.
# The SLURM job then runs with HF offline flags, hashes the C3VD warm-start and
# pose_transform checkpoints, and writes side-by-side panel PNGs plus metrics.

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
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"

# =============================================================================
# PANEL DEFAULTS
# =============================================================================

export HF_REPO_ID="${HF_REPO_ID:-SmartWhatt/real-colon-SfM}"
export HF_SPLIT="${HF_SPLIT:-val}"
export PREFETCH_HF="${PREFETCH_HF:-1}"
export CACHE_READY_MARKER="${CACHE_READY_MARKER:-$HF_CACHE_DIR/.real_colon_sfm_panel_${HF_SPLIT}_ready}"

export C3VDV2_LOG_DIR="${C3VDV2_LOG_DIR:-$PROJECT_DIR/logs/c3vdv2_hf}"
export REALCOLON_LOG_DIR="${REALCOLON_LOG_DIR:-$PROJECT_DIR/logs/real_colon_hf}"

export BASE_B="${BASE_B:-$C3VDV2_LOG_DIR/stage2_specular/models/weights_best}"
export BASE_D="${BASE_D:-$C3VDV2_LOG_DIR/stage2_specular_pps/models/weights_best}"

if [ -z "${POSE_B+x}" ]; then
    export POSE_B="$REALCOLON_LOG_DIR/realcolon_run_b_pose_transform/models/weights_0"
else
    export POSE_B
fi

if [ -z "${POSE_D+x}" ]; then
    if [ -d "$REALCOLON_LOG_DIR/realcolon_run_d_pose_transform/models/weights_0" ]; then
        export POSE_D="$REALCOLON_LOG_DIR/realcolon_run_d_pose_transform/models/weights_0"
    else
        export POSE_D="$REALCOLON_LOG_DIR/realcolon_run_d_diag/models/weights_0"
    fi
else
    export POSE_D
fi

export OUT_DIR="${OUT_DIR:-$PROJECT_DIR/reports/realcolon/pose_transform_panel}"
export PANEL_INDICES="${PANEL_INDICES:-0:16000:1000}"
export HEIGHT="${HEIGHT:-256}"
export WIDTH="${WIDTH:-320}"
export MIN_DEPTH="${MIN_DEPTH:-0.001}"
export MAX_DEPTH="${MAX_DEPTH:-0.1}"
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

prefetch_hf_dataset() {
    if [ "$PREFETCH_HF" != "1" ]; then
        log "Skipping HF prefetch (PREFETCH_HF=$PREFETCH_HF)"
        return
    fi

    log "Prefetching REALColon split on submit node"
    echo "  Repo:        $HF_REPO_ID"
    echo "  Split:       $HF_SPLIT"
    echo "  HF cache:    $HF_CACHE_DIR"
    echo "  HF_HOME:     $HF_HOME"
    mkdir -p "$HF_CACHE_DIR" "$HF_HOME"

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

    touch "$CACHE_READY_MARKER"
}

check_weights() {
    local missing=0
    for dir in "$BASE_B" "$POSE_B" "$BASE_D" "$POSE_D"; do
        if [ ! -d "$dir" ]; then
            echo "ERROR: weights folder does not exist: $dir"
            missing=1
            continue
        fi
        for file in encoder.pth depth.pth specular.pth; do
            if [ ! -f "$dir/$file" ]; then
                echo "ERROR: missing $file under $dir"
                missing=1
            fi
        done
    done
    if [ "$missing" -ne 0 ]; then
        exit 1
    fi
}

submit_job() {
    cd "$PROJECT_DIR"
    mkdir -p logs/slurm "$OUT_DIR"
    check_weights

    log "Submitting REALColon qualitative panel SLURM job"
    echo "  Project:      $PROJECT_DIR"
    echo "  HF repo:      $HF_REPO_ID"
    echo "  HF split:     $HF_SPLIT"
    echo "  Cache marker: $CACHE_READY_MARKER"
    echo "  Output:       $OUT_DIR"
    echo "  Indices:      $PANEL_INDICES"
    echo "  Run B base:   $BASE_B"
    echo "  Run B pose:   $POSE_B"
    echo "  Run D base:   $BASE_D"
    echo "  Run D pose:   $POSE_D"
    echo ""

    sbatch --export=ALL "$SCRIPT_PATH"
}

run_slurm_job() {
    log "Running REALColon qualitative panel on SLURM compute node"
    setup_runtime
    cd "$PROJECT_DIR"

    if [ ! -f "$CACHE_READY_MARKER" ]; then
        echo "ERROR: HF cache marker not found: $CACHE_READY_MARKER"
        echo "Run this script with bash on a login/submit node before using sbatch directly."
        exit 1
    fi
    check_weights

    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_CACHE_DIR
    export HF_HOME
    export HF_DATASETS_CACHE
    export TORCH_HOME

    mkdir -p "$OUT_DIR"

    echo "  Job ID:       ${SLURM_JOB_ID:-unknown}"
    echo "  Node:         ${SLURMD_NODENAME:-unknown}"
    echo "  HF repo:      $HF_REPO_ID"
    echo "  HF split:     $HF_SPLIT"
    echo "  HF cache:     $HF_CACHE_DIR"
    echo "  Offline:      HF_HUB_OFFLINE=$HF_HUB_OFFLINE HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
    echo "  Output:       $OUT_DIR"
    echo "  Indices:      $PANEL_INDICES"
    echo ""

    log "Writing checkpoint hashes"
    {
        echo "# REALColon pose_transform panel checkpoint hashes"
        echo "# Generated: $(date -Is)"
        echo "# BASE_B=$BASE_B"
        echo "# POSE_B=$POSE_B"
        echo "# BASE_D=$BASE_D"
        echo "# POSE_D=$POSE_D"
        echo ""
        for file in encoder.pth depth.pth specular.pth; do
            sha256sum "$BASE_B/$file" "$POSE_B/$file" "$BASE_D/$file" "$POSE_D/$file"
        done
    } | tee "$OUT_DIR/weight_hashes.txt"

    log "Rendering qualitative panel"
    panel_cmd=(
        uv run python tools/qualitative_realcolon_panel.py
        --hf_repo_id "$HF_REPO_ID"
        --hf_split "$HF_SPLIT"
        --hf_cache_dir "$HF_CACHE_DIR"
        --weights "run_b_base=$BASE_B"
        --weights "run_b_pose=$POSE_B"
        --weights "run_d_base=$BASE_D"
        --weights "run_d_pose=$POSE_D"
        --indices "$PANEL_INDICES"
        --out_dir "$OUT_DIR"
        --height "$HEIGHT"
        --width "$WIDTH"
        --min_depth "$MIN_DEPTH"
        --max_depth "$MAX_DEPTH"
        --depth_vis_min "$DEPTH_VIS_MIN"
        --depth_vis_max "$DEPTH_VIS_MAX"
    )
    if [ "$CPU" = "1" ]; then
        panel_cmd+=(--cpu)
    fi

    "${panel_cmd[@]}"

    log "Done"
    echo "  Panels:       $OUT_DIR"
    echo "  Hashes:       $OUT_DIR/weight_hashes.txt"
    echo "  Metrics CSV:  $OUT_DIR/panel_metrics.csv"
    echo "  Summary JSON: $OUT_DIR/summary.json"
}

if [ -n "${SLURM_JOB_ID:-}" ]; then
    run_slurm_job
else
    setup_runtime
    prefetch_hf_dataset
    submit_job
fi
