#!/bin/bash

# Set environment variables for GPU and other configurations
export CUDA_VISIBLE_DEVICES=0,1

MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=1
NUM_PROCESSES=2
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4

# Run the training script using the "accelerate" launcher
accelerate launch \
    --mixed_precision bf16 \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_PROCESSES \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --use_slow_tokenizer False \
    --use_flash_attn False \
    --max_seq_length 1024 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 3 \
    --output_dir output/sft_8b \
    --with_tracking \
    --report_to wandb \
    --wandb_entity autox-tech \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-hard-coded-10x 100 \
    --checkpointing_steps epoch \
    --dataset_mix_dir output/sft_8b \
    --exp_name self-recognition-llama \
    --seed 123
