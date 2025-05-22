# C3R: Channel-Conditioned Cell Representations for unified evaluation in microscopy imaging

## Environment Setup

```bash
# Create conda environment with base dependencies
conda create -n c3r python=3.9 numpy=1.26 pandas=2.2 scipy=1.13 scikit-learn=1.6 h5py=3.12 -y
conda activate c3r

# Install additional packages via pip
pip install matplotlib==3.9.4 numpy==2.0.2 torch==2.6.0 torchvision==0.21.0 \
            timm==1.0.14 einops==0.8.0 wandb==0.19.6
```
---
## HPA sample dataset download

The sample dataset can be found at https://doi.org/10.5281/zenodo.15489894 .
Each sample is an .h5 file, that contains approx. 24 cell images. During training, we sample 8 images at random per iteration. 

Save the .h5 files into ./sample_data/train_balanced/*.h5 .

---
## Training

### ViT Baseline: 
```bash
torchrun --nproc-per-node=8 train_scripts/train_baseline.py --arch 'vit_small' --batch_size_per_gpu 10 --data_path './sample_data' --in_chans 4 --channels all
```
### SingleChan: 
```bash
torchrun --nproc-per-node=8 train_scripts/train_baseline.py --arch 'vit_small' --batch_size_per_gpu 10 --data_path './sample_data' --in_chans 1 --channels random
```
### ChannelViT: 
```bash
torchrun --nproc-per-node=8 train_scripts/train_channelvit.py --arch 'vit_small' --batch_size_per_gpu 4 --data_path './sample_data' --channel_budget 2
```
### DiChaViT: 
```bash
torchrun --nproc-per-node=8 train_scripts/train_dichavit.py --arch 'vit_small' --batch_size_per_gpu 4 --data_path './sample_data' --channel_budget 2
```
### CCE: 
```bash
torchrun --nproc-per-node=8 train_scripts/train_c3r.py --arch 'vit_small' --batch_size_per_gpu 10 --data_path './sample_data' --student_keep_count 3 --teacher_keep_count 3 --aggregation pre --pre_layers 0 1 --post_layer_count -1 
```
### CCE + MCD
```bash
torchrun --nproc-per-node=8 train_scripts/train_c3r.py --arch 'vit_small' --batch_size_per_gpu 10 --data_path './sample_data' --student_keep_count 1 2 3 --teacher_keep_count 3 --aggregation post --pre_layers 0 1 --post_layer_count -1 
```

---
## Pre-trained checkpoints

The pre-trained checkpoints can be found at https://doi.org/10.5281/zenodo.15490165

---
## Evaluation

We follow the same evaluation protocol as SubCell and use their code. Our exact implementation will be uploaded shortly. 
