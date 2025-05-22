# C3R: Channel-Conditioned Cell Representations for unified evaluation in microscopy imaging

## Environment Setup

```bash
# Create conda environment with base dependencies
conda create -n c3r python=3.9 numpy=1.26 pandas=2.2 scipy=1.13 scikit-learn=1.6 h5py=3.12 -y
conda activate c3r

# Install additional packages via pip
pip install -r requirements.txt

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
torchrun --nproc-per-node=8 train_scripts/main_baseline.py --arch 'vit_small' --num_workers 12 --batch_size_per_gpu 10 --data_path './sample_data' --in_chans 4 --channels all
```
### SingleChan: 
```bash
torchrun --nproc-per-node=8 train_scripts/main_baseline.py --arch 'vit_small' --num_workers 12 --batch_size_per_gpu 10 --data_path './sample_data' --in_chans 1 --channels random
```
### ChannelViT: 
```bash
torchrun --nproc-per-node=8 train_scripts/main_channelvit.py --arch 'vit_small' --num_workers 12 --batch_size_per_gpu 4 --data_path './sample_data' --channel_budget 2
```
### DiChaViT: 
```bash
torchrun --nproc-per-node=8 train_scripts/main_dichavit.py --arch 'vit_small' --num_workers 12 --batch_size_per_gpu 4 --data_path './sample_data' --channel_budget 2
```
### CCE: 
```bash
torchrun --nproc-per-node=8 train_scripts/main_c3r.py --arch 'vit_small' --num_workers 12 --batch_size_per_gpu 10 --data_path './sample_data' --student_keep_count 3 --teacher_keep_count 3 --aggregation pre --pre_layer_count 0 1 --post_layer_count -1 
```
### CCE + MCD
```bash
torchrun --nproc-per-node=8 train_scripts/main_c3r.py --arch 'vit_small' --num_workers 12 --batch_size_per_gpu 10 --data_path './sample_data' --student_keep_count 1 2 3 --teacher_keep_count 3 --aggregation post --pre_layer_count 0 1 --post_layer_count -1 
```

---
## Pre-trained checkpoints

The pre-trained checkpoints, and pre-computed features can be found at https://doi.org/10.5281/zenodo.15491064

---
## Evaluation

Use eval_scripts/train_hpa_19.py and train_hpa_31.py for 19 and 31-class protein localization with the argument --features_folder as your pre-computed features. 
For JUMP-CP, we use the exact subcell script to run OOD evaluation. 
