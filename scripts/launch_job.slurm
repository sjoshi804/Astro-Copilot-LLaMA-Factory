#!/bin/bash
#SBATCH --job-name=qwen_training         # Job name
#SBATCH --nodes=8                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks per node (1 per GPU)
#SBATCH --time=1:59:59                     # Max job run time
#SBATCH --output=/work/10450/sjoshi804/vista/logs/qwen_train_%j.out     # Output log file
#SBATCH --error=/work/10450/sjoshi804/vista/logs/qwen_train_%j.err      # Error log file
#SBATCH --partition=gh-dev                     # Partition name

# Load modules
module load gcc cuda

source ~/.bashrc
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate lf

# Print current conda environment
echo "Current Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"

srun $WORK/LLaMA-Factory/scripts/multinode_qwen_coder_astro_sft.sh
