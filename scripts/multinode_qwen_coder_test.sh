#!/bin/bash
HOST=$1
NODES=$2

# Enable NCCL debugging for distributed training
export NCCL_DEBUG=INFO
export NODENAME=$(hostname -s)

# Set SLURM distributed training variables
export RANK=$SLURM_PROCID
export FS_LOCAL_RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=1
export LOCAL_RANK=0
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))

# Bash RC And env
source ~/.bashrc
conda activate lf

# Logging SLURM environment variables
echo "=============================================="
echo "SLURM Job Details:"
echo "Nodelist       : $SLURM_JOB_NODELIST"
echo "Total Nodes    : $SLURM_JOB_NUM_NODES"
echo "Tasks per Node : $SLURM_NTASKS_PER_NODE"
echo "NODE_RANK      : $NODE_RANK"
echo "WORLD_SIZE     : $WORLD_SIZE"
echo "=============================================="

# Set distributed environment variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export SLURM_GPUS_PER_NODE=1
TOTAL_GPUS=$(($SLURM_NTASKS * SLURM_GPUS_PER_NODE))
export WORLD_SIZE=$TOTAL_GPUS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Get master node address and confirm connectivity
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

cd $WORK/LLaMA-Factory

echo "=============================================="
echo "Distributed Training Configuration:"
echo "MASTER_ADDR    : $MASTER_ADDR"
echo "MASTER_PORT    : $MASTER_PORT"
echo "WORLD_SIZE     : $WORLD_SIZE"
echo "LOCAL_RANK     : $LOCAL_RANK"
echo "RANK           : $RANK"
echo "CWD            : $(pwd)"
echo "=============================================="

# Check master node connectivity
echo "Pinging master node to confirm connectivity..."
ping -c 5 $MASTER_ADDR

# Verify if the master port is accessible
echo "Checking connectivity to master node port..."
nc -zv $MASTER_ADDR $MASTER_PORT

# NCCL performance tuning (recommended settings)
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

# Run torch distributed training
torchrun --nnodes=$SLURM_NNODES \
         --nproc_per_node=$SLURM_GPUS_PER_NODE \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         --rdzv_id=12349 \
         --rdzv_conf 'read_timeout=420' \
         $WORK/LLaMA-Factory/src/train.py \
         --deepspeed $WORK/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
         --stage sft \
         --do_train \
         --use_fast_tokenizer \
         --flash_attn auto \
         --model_name_or_path Qwen/Qwen2.5-Coder-3B \
         --dataset qwen_coder_demo \
         --template qwen \
         --finetuning_type lora \
         --output_dir $WORK/checkpoints/qwen_coder \
         --overwrite_cache \
         --overwrite_output_dir \
         --weight_decay 0.1 \
         --per_device_train_batch_size 3 \
         --gradient_accumulation_steps 1 \
         --learning_rate 5e-5 \
         --lr_scheduler_type cosine \
         --logging_steps 1 \
         --cutoff_len 4096 \
         --save_steps 1000 \
         --plot_loss \
         --num_train_epochs 3

echo "Job completed successfully!"
