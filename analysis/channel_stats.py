# Standard Library
import os
import sys

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

# Third-Party Libraries
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import torch.nn as nn

import cv2

# Local Imports
from data.hpa_dataset_single import HPASubCellDataset

import glob


def parity_entropy_per_class(true_labels, cluster_preds, num_classes):
    # Confusion matrix: rows=true classes, cols=clusters
    cm = confusion_matrix(true_labels, cluster_preds)

    # Hungarian algorithm to align clusters to classes (maximize correct matches)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Map cluster -> class
    cluster_to_class = dict(zip(col_ind, row_ind))

    # Relabel cluster preds to classes based on assignment; unmatched -> -1
    aligned_preds = np.array([cluster_to_class.get(c, -1) for c in cluster_preds])

    parities = []
    entropies = []

    for c in range(num_classes):
        # Samples of class c
        idxs = np.where(true_labels == c)[0]
        aligned_c_preds = aligned_preds[idxs]

        # Parity per class: fraction correctly assigned
        parity_c = np.mean(aligned_c_preds == c)
        parities.append(parity_c)

        # Entropy per class: from confusion matrix row c
        cluster_counts = cm[c, :]
        if cluster_counts.sum() == 0:
            ent = 0.0
        else:
            probs = cluster_counts / cluster_counts.sum()

            ent = entropy(probs, base=2)
        entropies.append(ent)

    # Overall parity (accuracy)
    overall_parity = np.mean(aligned_preds == true_labels)

    return overall_parity, parities, entropies

def compute_kmeans_labels(tensor, n_clusters):
    """
    Args:
        tensor: numpy array or torch tensor of shape (N, C, D)
        n_clusters: int, number of clusters for KMeans

    Returns:
        true_labels: numpy array of shape (N*C,), true channel labels (0,...,C-1)
        cluster_preds: numpy array of shape (N*C,), cluster assignments
    """
    # Convert to numpy if tensor is torch tensor
    if 'torch' in str(type(tensor)):
        tensor = tensor.cpu().numpy()

    N, C, D = tensor.shape
    features = tensor.reshape(-1, D)  # (N*C, D)

    # True labels: for each vector, the channel index repeated N times
    true_labels = np.tile(np.arange(C), N)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_preds = kmeans.fit_predict(features)

    return true_labels, cluster_preds

class MinMaxNormalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min = torch.amin(x, dim=(1, 2), keepdim=True)
        max = torch.amax(x, dim=(1, 2), keepdim=True)

        x = (x - min) / (max - min + 1e-6)

        return x

class myDset(Dataset):

    def __init__(self, img_list, dtype='uint8'):
        super().__init__()

        self.img_list = glob.glob('/scratch1/test_data/antibody_cell_imgs/train-pretrain_uint8/*/*.png')
        self.dtype = dtype
        self.normalize = MinMaxNormalize()

    def __getitem__(self, index):

        img = cv2.imread(self.img_list[index], -1)

        if self.dtype == 'uint8' and img.dtype == np.uint16:
            img = (img/255).astype(np.uint8)
        elif self.dtype == 'uint16' and img.dtype == np.uint8:
            img = (img*255).astype(np.uint16)

        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)

        # normalize
        img = self.normalize(img)

        return img, -1
    
    def __len__(self):

        return len(self.img_list)
    

# # Set up dataset
# dset = HPASubCellDataset(root='/scratch1/test_data/antibody_cell_imgs', split='train', uint8=False, normalize=True, transform=None)

model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)

model = model.eval().cuda()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

pretrain_imgs = None

fname = 'dinov2_hpav3_uint8'

dset = myDset(pretrain_imgs, dtype=fname.split('_')[-1])
loader = DataLoader(dset, batch_size=1, num_workers=0, shuffle=False)

images = []
masks = []


with torch.no_grad():

    outs = []

    for idx, (imgss, labelss) in enumerate(tqdm(loader)):

        cell_imgs = imgss[0]

        # cell_imgs = cell_imgs * (mask/255)

        outs_img = []

        for img in cell_imgs:

            img = torch.stack([img for _ in range(3)]).cuda()
            img = transforms(img)

            out = model(img.unsqueeze(0)).cpu().detach()

            outs_img.append(out)

        outs.append(torch.stack(outs_img, dim=1))

        if idx ==999:
            break

tensor = torch.stack(outs, dim=1)
torch.save(tensor, f'{fname}.pt')

tensor_all = torch.load(f'{fname}.pt')

tensor = tensor_all[0]

N, C, D = tensor.shape
tensor_np = tensor.numpy().reshape(-1, D)  # (N*C, D)

# True string labels per channel
class_names = ['Microtubules', 'Nucleus', 'ER', 'Protein']
true_labels = np.tile(class_names, N)       # (N*C,)

# ---------- UMAP ----------
reducer = umap.UMAP(n_components=2)
reduced = reducer.fit_transform(tensor_np)  # (N*C, 2)

# ---------- DATAFRAME ----------
df = pd.DataFrame({
    'x': reduced[:, 0],
    'y': reduced[:, 1],
    'label': true_labels
})

# ---------- CUSTOM COLORS ----------
custom_palette = {
    'Microtubules': 'tab:red',
    'Nucleus': 'tab:blue',
    'ER': 'tab:orange',
    'Protein': 'tab:green'
}

# ---------- PLOT ----------
plt.figure(figsize=(5, 4))
plt.rcParams.update({'font.size': 11})
plt.grid(True, zorder=0)
ax = sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='label',
    palette=custom_palette,
    s=5,
    alpha=0.8,
    edgecolor=None
)
for collection in ax.collections:
    collection.set_zorder(3)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=4)
plt.xlabel("UMAP-1", fontsize=12)
plt.ylabel("UMAP-2", fontsize=12)
plt.legend(markerscale=2, fontsize=10)  # Increase markerscale for larger dots
sns.despine()
plt.tight_layout()
plt.savefig(f'{fname}.pdf')
plt.close()




true_labels, cluster_preds = compute_kmeans_labels(tensor, n_clusters=4)
overall_parity, parity_per_class, entropy_per_class = parity_entropy_per_class(true_labels, cluster_preds, 4)
print("Parity per class:", parity_per_class)
print("Entropy per class:", entropy_per_class)


