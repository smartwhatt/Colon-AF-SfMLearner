#!/bin/bash
#SBATCH -p memory                   # CPU/memory partition (no GPU needed)
#SBATCH -N 1 -c 4
#SBATCH --ntasks-per-node=1
#SBATCH -t 8:00:00                  # Adjust to how long you want it running
#SBATCH -A lt200353
#SBATCH -J tensorboard
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err

module load Mamba/23.11.0-0
conda activate /lustrefs/disk/project/lt200353-pcllm/swattana/conda

PROJECT_DIR="/lustrefs/disk/project/lt200353-pcllm/swattana/Colon-AF-SfMLearner"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/c3vdv2}"
PORT="${PORT:-6006}"

cd "$PROJECT_DIR"
mkdir -p logs/slurm

NODE=$(hostname)
echo "========================================"
echo "TensorBoard running on: $NODE:$PORT"
echo "Log dir: $LOG_DIR"
echo ""
echo "To access locally, run this SSH tunnel from your machine:"
echo "  ssh -N -L $PORT:$NODE:$PORT <your-lanta-login>@lanta.nstda.or.th"
echo ""
echo "Then open: http://localhost:$PORT"
echo "========================================"

uv run tensorboard --logdir "$LOG_DIR" --port "$PORT" --bind_all
