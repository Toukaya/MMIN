#!/bin/bash
dataset=$1
gpu_ids=$2

# 训练模型的简化脚本
for mask_rate in 0.0 0.1 0.2 0.3 0.4 0.5; do
    python -u train_miss.py \
        --mask_rate=${mask_rate} \
        --dataset_mode=${dataset}_miss \
        --model=mmin \
        --log_dir=./logs \
        --checkpoints_dir=./checkpoints \
        --print_freq=10 \
        --gpu_ids=$gpu_ids \
        --input_dim_a=512 \
        --embd_size_a=128 \
        --input_dim_v=1024 \
        --embd_size_v=128 \
        --input_dim_l=1024 \
        --embd_size_l=128 \
        --AE_layers=256,128,64 \
        --n_blocks=5 \
        --num_thread=0 \
        --pretrained_path=checkpoints/utt_fusion_${dataset}_AVL \
        --ce_weight=1.0 \
        --mse_weight=0.2 \
        --cycle_weight=1.0 \
        --cls_layers=128,128 \
        --dropout_rate=0.5 \
        --niter=20 \
        --niter_decay=80 \
        --init_type normal \
        --batch_size=256 \
        --lr=1e-3 \
        --run_idx=1 \
        --name=mmin \
        --suffix=${dataset}_MMINV
done