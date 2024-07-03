#!/bin/bash
dataset=$1
gpu_ids=$2
modalities=('A' 'V' 'L' 'AV' 'AL' 'VL')

for modality in "${modalities[@]}"; do
  python train_baseline.py --dataset_mode=${dataset}_multimodal \
  --model=utt_fusion --gpu_ids=$gpu_ids --modality=$modality \
  --log_dir=./logs --checkpoints_dir=./checkpoints \
  --print_freq=10 --input_dim_a=512 --embd_size_a=128 \
  --input_dim_v=1024 --embd_size_v=128 --input_dim_l=1024 \
  --embd_size_l=128 --cls_layers=128,128 --dropout_rate=0.3 \
  --niter=20 --niter_decay=80 --beta1=0.9 --init_type=kaiming \
  --batch_size=256 --lr=1e-3 --run_idx=6 --name=utt_fusion \
  --suffix=${dataset}_AVL --output_dim=1
done