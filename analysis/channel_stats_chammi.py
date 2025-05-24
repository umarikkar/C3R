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
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

# Local Imports
from data.chammi.morphem70k import SingleCellDataset
from data.chammi.chammi_dataset_utils import get_data_transform

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

channel_dict = {
    "Allen":(3,  ["Nucleus", "Membrane", "Protein"]),
    "HPA": (4, ["Microtubules", "Protein", "Nucleus", "ER"]),
    "CP": (5,  ["Nucleus", "ER", "RNA", "Golgi", "Mito"]),
}

do_umap=False

for dataset_name, (num_channels, channel_names) in channel_dict.items():

    try:
        tensor_all = torch.load(f'analysis/chammi_datasets/dinov2_{dataset_name}.pt')
    except:
        _, transform_eval = get_data_transform(dataset_name, 256, 0, False)

        # Set up dataset
        dset = SingleCellDataset(csv_path='/work/um00109/CHAMMI/channel_adaptive_models/chammi_dataset/morphem70k_v2.csv', 
                                chunk=dataset_name,
                                root_dir='/work/um00109/CHAMMI/channel_adaptive_models/chammi_dataset',
                                is_train=False,
                                ssl_flag=None,
                                transform=None,
                                )

        loader = DataLoader(dset, shuffle=True, batch_size=1, num_workers=8)

        images = []
        masks = []

        model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)

        model = model.eval().cuda()

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        with torch.no_grad():

            outs = []

            for idx, imgss in enumerate(tqdm(loader)):

                img_all = imgss[0]

                outs_img = []

                for img in img_all:
                    

                    img = (img - img.min()) / (img.max() - img.min())

                    img = torch.stack([img for _ in range(3)]).cuda()
                    img = transforms(img)

                    out = model(img.unsqueeze(0)).cpu().detach()

                    outs_img.append(out)

                outs.append(torch.stack(outs_img, dim=1))

                if idx ==999:
                    break

        tensor_all = torch.stack(outs, dim=1)
        torch.save(tensor_all, f'analysis/chammi_datasets/dinov2_{dataset_name}.pt')

    tensor = tensor_all[0]

    if do_umap:

        N, C, D = tensor.shape

        tensor_np = tensor.numpy().reshape(-1, D)  # (N*C, D)

        # True string labels per channel
        # class_names = ['a', 'b', 'c']
        class_names = channel_names
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

        # ---------- PLOT ----------
        plt.figure(figsize=(5, 4))
        plt.rcParams.update({'font.size': 11})
        plt.grid(True, zorder=0)
        ax = sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='label',
            # palette=custom_palette,
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
        plt.savefig(f'analysis/chammi_datasets/umap_{dataset_name}.pdf')
        plt.close()


    true_labels, cluster_preds = compute_kmeans_labels(tensor, n_clusters=num_channels)
    overall_parity, parity_per_class, entropy_per_class = parity_entropy_per_class(true_labels, cluster_preds, num_channels)

    df = pd.DataFrame({
    'Class': channel_names,
    'Parity': parity_per_class,
    'Entropy': entropy_per_class
    })

    # Set class as index (optional, for better formatting)
    df.set_index('Class', inplace=True)
    print(df)



