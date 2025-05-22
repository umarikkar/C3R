#!/bin/bash

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate c3r


# Baseline
torchrun --nproc-per-node=8 train_scripts/train_baseline.py --arch 'vit_small' --batch_size_per_gpu 10 --data_path './sample_data' \
                    --pred_ratio 0.0 0.3 --pred_ratio_var 0.0 0.2 \
                    --in_chans 4 --channels all

# SingleChan
torchrun --nproc-per-node=8 train_scripts/train_baseline.py --arch 'vit_small' --batch_size_per_gpu 10 --data_path './sample_data' \
                    --pred_ratio 0.0 0.3 --pred_ratio_var 0.0 0.2 \
                    --in_chans 1 --channels random

# ChannelViT
torchrun --nproc-per-node=8 train_scripts/train_channelvit.py --arch 'vit_small' --batch_size_per_gpu 4 --data_path './sample_data' \
                    --pred_ratio 0.0 0.3 --pred_ratio_var 0.0 0.2 \
                    --channel_budget 2

# ChannelViT
torchrun --nproc-per-node=8 train_scripts/train_dichavit.py --arch 'vit_small' --batch_size_per_gpu 4 --data_path './sample_data' \
                    --pred_ratio 0.0 0.3 --pred_ratio_var 0.0 0.2 \
                    --channel_budget 2

# CCE
torchrun --nproc-per-node=8 train_scripts/train_c3r.py --arch 'vit_small' --batch_size_per_gpu 10 --data_path './sample_data' \
                    --pred_ratio 0.0 0.3 --pred_ratio_var 0.0 0.2 \
                    --student_keep_count 3 --teacher_keep_count 3 --aggregation pre \
                    --pre_layers 0 1 --post_layer_count -1 

# CCE + MCD
torchrun --nproc-per-node=8 train_scripts/train_c3r.py --arch 'vit_small' --batch_size_per_gpu 10 --data_path './sample_data' \
                    --pred_ratio 0.0 0.3 --pred_ratio_var 0.0 0.2 \
                    --student_keep_count 1 2 3 --teacher_keep_count 3 --aggregation post \
                    --pre_layers 0 1 --post_layer_count -1 

