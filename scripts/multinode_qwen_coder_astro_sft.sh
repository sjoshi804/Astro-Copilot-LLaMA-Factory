#!/bin/bash
export WANDB_PROJECT="astro_llm"
export WANDB_LOG_SYSTEM="all"

# Set configurable training parameters
BASE_MODEL="Qwen/Qwen2.5-Coder-3B"
DATASET="astro_sft_train"
EVAL_DATASET="astro_sft_test"
NUM_EPOCHS=5
LEARNING_RATE=5e-6
BATCH_SIZE=256
WEIGHT_DECAY=0.1
GRAD_ACCUMULATION_STEPS=1
LR_SCHEDULER_TYPE="cosine"
CUTOFF_LEN=2048
FINETUNING_TYPE="full"
WARMUP_STEPS=100

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

# Logging SLURM environment variables and training parameters
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

echo "=============================================="
echo "               Hyperparameters                "
echo "=============================================="
echo "Base Model     : $BASE_MODEL"
echo "Dataset Path   : $DATASET"
echo "Num Epochs     : $NUM_EPOCHS"
echo "Learning Rate  : $LEARNING_RATE"
echo "Batch Size     : $BATCH_SIZE"
echo "Weight Decay   : $WEIGHT_DECAY"
echo "Gradient Accum : $GRAD_ACCUMULATION_STEPS"
echo "LR Scheduler   : $LR_SCHEDULER_TYPE"
echo "Cutoff Len     : $CUTOFF_LEN"
echo "Finetuning Type: $FINETUNING_TYPE"
echo "Warmup Steps   : $WARMUP_STEPS"
echo "=============================================="

# NCCL performance tuning (recommended settings)
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

PER_DEVICE_BATCH_SIZE=$(($BATCH_SIZE / $TOTAL_GPUS))
echo "=============================================="
echo "Computed Per Device Batch Size: $PER_DEVICE_BATCH_SIZE"
echo "=============================================="


# Wandb
echo "=============================================="
echo "                    WANDB                     "
echo "=============================================="
wandb login $WANDB_API_KEY
output=$(wandb login)
echo "$output"
output=$(wandb status)
echo "$output"
echo "=============================================="

# Run training
torchrun --nnodes=$SLURM_NNODES \
         --nproc_per_node=$SLURM_GPUS_PER_NODE \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         --rdzv_id=12349 \
         --rdzv_conf 'read_timeout=420' \
         $WORK/LLaMA-Factory/src/train.py \
         --deepspeed $WORK/LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
         --stage sft \
         --do_train \
         --use_fast_tokenizer \
         --flash_attn auto \
         --model_name_or_path $BASE_MODEL \
         --dataset $DATASET \
         --eval_dataset $EVAL_DATASET \
         --template qwen \
         --finetuning_type $FINETUNING_TYPE \
         --output_dir $WORK/checkpoints/qwen_coder/astro_sft_train \
         --overwrite_cache \
         --overwrite_output_dir \
         --num_train_epochs $NUM_EPOCHS \
         --weight_decay $WEIGHT_DECAY \
         --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
         --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \
         --learning_rate $LEARNING_RATE \
         --lr_scheduler_type $LR_SCHEDULER_TYPE \
         --cutoff_len $CUTOFF_LEN \
         --warmup_steps $WARMUP_STEPS \
         --save_steps 1000 \
         --logging_steps 1 \
         --evaluation_strategy "epoch" \
         --report_to wandb 

echo "Job completed successfully!"
