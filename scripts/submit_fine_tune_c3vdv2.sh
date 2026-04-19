#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16   			    # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=1		    # Specify number of GPU per task
#SBATCH --ntasks-per-node=4		    # Specify tasks per node
#SBATCH -t 1:00:00                  # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200353                 # Specify project name
#SBATCH -J fine_tune_c3vdv2         # Specify job name

module load Mamba        # Load the conda module