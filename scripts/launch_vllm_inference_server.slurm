#!/bin/bash
#SBATCH --job-name=vllm_inference_server
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node
#SBATCH --time=1:59:59            # Job time limit
#SBATCH --output=logs/inference_server_%j.out
#SBATCH --error=logs/inference_server_%j.err
#SBATCH --partition=gh-dev

# Load modules
module load gcc/13 cuda/12.4 

source ~/.bashrc
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate lf

# Print current conda environment
echo "Current Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Model to serve 
export MODEL_NAME="$WORK/checkpoints/qwen_coder"

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Running on master node: $MASTER_ADDR"

# Run vLLM using torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m vllm.entrypoints.api_server \
    --model $MODEL_NAME \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000
