#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16                  # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=1           # Specify number of GPU per task
#SBATCH --ntasks-per-node=1         # Specify tasks per node
#SBATCH -t 24:00:00                 # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200353                 # Specify project name
#SBATCH -J fine_tune_c3vdv2         # Specify job name
#SBATCH -o logs/slurm/%x_%j.out    # stdout log
#SBATCH -e logs/slurm/%x_%j.err    # stderr log

module load Mamba/23.11.0-0        # Load the conda module

conda activate /lustrefs/disk/project/lt200353-pcllm/swattana/conda   # Activate the conda environment

# =============================================================================
# PATHS — adjust to your LANTA project layout
# =============================================================================

PROJECT_DIR="/lustrefs/disk/project/lt200353-pcllm/swattana/Colon-AF-SfMLearner"
DATA_PATH="/lustrefs/disk/project/lt200353-pcllm/swattana/data/C3VDv2"
LOG_DIR="$PROJECT_DIR/logs/c3vdv2"
UNDISTORT_MAPS="$PROJECT_DIR/undistort_maps/"

# Optional: SCARED Stage 1 pretrained weights for warm-start
SCARED_STAGE1_WEIGHTS="$PROJECT_DIR/model/pretrained/scared_stage1/"

# =============================================================================
# RUN
# =============================================================================

cd "$PROJECT_DIR"
mkdir -p logs/slurm

export DATA_PATH LOG_DIR UNDISTORT_MAPS SCARED_STAGE1_WEIGHTS
export GPU=0

bash scripts/run_finetune_c3vdv2.sh