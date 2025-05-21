# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
from scipy.ndimage import grey_dilation
import re
import einops


from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

from torchvision import transforms
from PIL import ImageFilter, ImageOps, Image, ImageDraw
import glob
import pandas as pd
import torch.nn as nn
import h5py
import random

import sys

COLORS = ["red", "yellow", "blue", "green"]
LOCATION_MAP = pd.read_csv("annotations/location_group_mapping.tsv", sep="\t")
UNIQUE_CATS = LOCATION_MAP["Original annotation"].unique().tolist()
UNIQUE_CATS.append("Negative")
NUM_CLASSES = len(UNIQUE_CATS)

# class MinMaxNormalize(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         min = torch.amin(x, dim=(1, 2, 3), keepdim=True)
#         max = torch.amax(x, dim=(1, 2, 3), keepdim=True)

#         return (x - min) / (max - min + 1e-6)
    


class MinMaxNormalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min = torch.amin(x, dim=(1, 2), keepdim=True)
        max = torch.amax(x, dim=(1, 2), keepdim=True)

        x = (x - min) / (max - min + 1e-6)

        return x
    
class MinMaxNormalizeWithSize(nn.Module):
    def __init__(self, sz=224):
        super().__init__()
        self.resize=transforms.RandomResizedCrop(size=(sz, sz), scale=(0.9, 1))

    def forward(self, x):
        min = torch.amin(x, dim=(1, 2), keepdim=True)
        max = torch.amax(x, dim=(1, 2), keepdim=True)

        x = (x - min) / (max - min + 1e-6)

        x = self.resize(x)

        return x

class MinMaxChannelNormalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min = torch.amin(x, dim=(2, 3), keepdim=True)
        max = torch.amax(x, dim=(2, 3), keepdim=True)

        return (x - min) / (max - min + 1e-6)

class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index
    
def shuffle_dict_keys(input_dict):
    keys = list(input_dict.keys())
    random.shuffle(keys)
    return {key: input_dict[key] for key in keys}

from collections import defaultdict

def get_stratified_idxs_v2(cell_lines, plate_positions, n_idxs):
    # Create a dictionary to group indices by (cell_line, plate_position)
    index_per_category = defaultdict(list)
    for idx, (cell_line, plate_position) in enumerate(zip(cell_lines, plate_positions)):
        index_per_category[(cell_line, plate_position)].append(idx)

    # Shuffle category keys
    category_keys = list(index_per_category.keys())
    random.shuffle(category_keys)

    sampled_idxs = []
    while len(sampled_idxs) < n_idxs:
        for category in category_keys:
            if index_per_category[category]:
                sampled_idxs.append(index_per_category[category].pop())
            if len(sampled_idxs) == n_idxs:
                return sampled_idxs  # Exit early when target count is reached

    return sampled_idxs


def get_stratified_idxs(df, n_idxs):
    df["basename"] = (
        df["cell_line"].astype(str) + "_" + df["plate_position"].astype(str)
    )
    index_per_category = (
        df.groupby("basename")
        .apply(lambda x: x.index.tolist(), include_groups=False)
        .to_dict()
    )

    sampled_idxs = []
    count=0
    while len(sampled_idxs) < n_idxs:
        for category, indexlist in shuffle_dict_keys(index_per_category).items():
            if len(indexlist) > 0:
                random.shuffle(indexlist)
                sampled_idxs.append(indexlist.pop())
            if len(sampled_idxs) == n_idxs:
                break
    assert (
        len(sampled_idxs) == n_idxs
    ), f"n_idxs: {n_idxs}, len(sampled_idxs): {len(sampled_idxs)}"
    return sampled_idxs
    

class HPASubCellDataset(VisionDataset):

    def __init__(
            self,
            root: str,
            split: str,
            data_prop=1.,

            color_channels=["red", "yellow", "blue", "green"],
            n_cells: int = -1,
            mask_prob: float = 0.0,
            normalize: str = "min_max",
            return_cell_mask: bool = False,
            finetuning=False,

            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,

            channels='all',

            
    ) -> None:
        super().__init__(root, transform=transform,
                                            target_transform=target_transform)
        
        data_folder = os.path.join(root, f"{split}-balanced")
        
        all_samples = pd.DataFrame({'filename':glob.glob(data_folder + '/*.h5'), 'antibody':''})
        all_samples['antibody'] = [f.split('/')[-1][:-5] for f in all_samples['filename']]
        with open(f"./annotations/splits/{split}_antibodies.txt") as file:
            antibodies = pd.DataFrame({'antibody':[line.rstrip() for line in file]})

        
        self.finetuning = finetuning
        # dropping the mitochondria channel
        self.channels = channels 

        self.df = all_samples.merge(antibodies, on='antibody').reset_index()
        if data_prop < 1:
            random_idxs = random.sample(list(range(len(self.df))), k=int(data_prop * len(self.df)))
            self.df = self.df.iloc[random_idxs].reset_index()

        print(f'created {split} dataset of {len(self.df)} samples.')

        self.color_channels = color_channels
        self.color_idxs = [i for i, c in enumerate(COLORS) if c in color_channels]
        self.n_cells = n_cells

        self.mask_prob = mask_prob
        self.epoch=0

        self.normalize = MinMaxNormalize()

        self.unique_cats = UNIQUE_CATS
        self.num_classes = NUM_CLASSES

        self.return_cell_mask = return_cell_mask
        self.split=split

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.df)
    
    def get_mask_tensor(self, masks):
        masks = (masks * 255).astype(np.uint8)
        masks = grey_dilation(grey_dilation(masks, size=(1, 5, 1)), size=(1, 1, 5))
        # masks = (masks / 255.0).astype(np.float32)
        masks = (masks / 255.0).astype(np.float16)
        masks = torch.from_numpy(masks[:, None, :, :])
        return masks

    def get_random_mask(self, mask_tensor):
        n_cells = mask_tensor.shape[0]
        n_non_masked = int(n_cells * (1 - self.mask_prob))
        mask_idxs = np.random.choice(n_cells, n_non_masked, replace=False)
        mask_tensor[mask_idxs] = 1.0
        return mask_tensor
    
    def get_select_idxs(self, cell_line, plate_postion, num_cells):
        if self.n_cells > 0 and num_cells > self.n_cells:
            df = pd.DataFrame(
                {
                    "cell_line": cell_line.split(";"),
                    "plate_position": plate_postion.split(";"),
                }
            )
            select_idxs = get_stratified_idxs(df, self.n_cells)
        elif self.n_cells > 0:
            # TODO: handle this such that we send the initial number + some randomly assigned n_cells. 
            select_idxs = np.arange(num_cells)
        else:
            select_idxs = np.arange(num_cells)
        return select_idxs

    def get_img_and_transform(self, img):

        # return torch.tensor(img).permute(2,0,1)

        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)

        # normalize
        img = self.normalize(img)

        # get the crops: will return a duo of images for each cell!
        if self.transform is not None:
            img = self.transform(img)

        return img
    

    def __getitem__(self, idx: int) -> Tuple:

        with h5py.File(self.df['filename'][idx], "r") as f:
            select_idxs = self.get_select_idxs(f.attrs["cell_line"], f.attrs["plate_position"], f.attrs["num_cells"])
            antibody_id = f.attrs["antibody"]

            """
            when using 448px, load with the mask. 
            Otherwise load wihtout the mask as it is already saved without!
            """
            valid_or_missing = self.split == 'valid' or 'imgs_256' not in self.df['filename'][idx]

            if self.channels == 'mt':
                slice_range = slice(1, -1 if valid_or_missing else None)
            elif self.channels == 'all':
                slice_range = slice(None, -1 if valid_or_missing else None)
            elif self.channels == 'random':
                r = torch.randint(0, 4, [])
                slice_range = slice(r, r + 1)

            images = []
            chans = []

            for i in select_idxs:
                img, chan = self.get_img_and_transform(f[f"img_{i}"][:, :, slice_range])
                images.append(img)
                chans.append(chan)
            
            if self.finetuning:
                target = np.array(f['target'])

        antibody_id_tensor = torch.tensor(
            [int(re.findall(r"\d+", antibody_id)[0])] * len(images)
        )

        if self.finetuning:
            target = torch.tensor(target)
            return torch.stack(images), torch.stack([target for _ in range(len(images))]).float()
        
        #  this is for pre-training only
        if self.transform is not None:
            flattened_images = [tensor for group in images for tensor in group]
            flattened_chans = [tensor for group in chans for tensor in group]
            
            return flattened_images, antibody_id_tensor, flattened_chans
        
        else:
            return torch.stack(images), antibody_id


class HPASubCellDatasetCVIT(VisionDataset):

    def __init__(
            self,
            root: str,
            split: str,
            data_prop=1.,

            color_channels=["red", "yellow", "blue", "green"],
            n_cells: int = -1,
            mask_prob: float = 0.0,
            normalize: str = "min_max",
            return_cell_mask: bool = False,
            finetuning=False,

            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,

            channels='all',

            
    ) -> None:
        super().__init__(root, transform=transform,
                                            target_transform=target_transform)
        
        data_folder = os.path.join(root, f"{split}-balanced")
        
        all_samples = pd.DataFrame({'filename':glob.glob(data_folder + '/*.h5'), 'antibody':''})
        all_samples['antibody'] = [f.split('/')[-1][:-5] for f in all_samples['filename']]
        with open(f"./annotations/splits/{split}_antibodies.txt") as file:
            antibodies = pd.DataFrame({'antibody':[line.rstrip() for line in file]})

        
        self.finetuning = finetuning
        # dropping the mitochondria channel
        self.channels = channels 

        self.df = all_samples.merge(antibodies, on='antibody').reset_index()
        if data_prop < 1:
            random_idxs = random.sample(list(range(len(self.df))), k=int(data_prop * len(self.df)))
            self.df = self.df.iloc[random_idxs].reset_index()

        print(f'created {split} dataset of {len(self.df)} samples.')

        self.color_channels = color_channels
        self.color_idxs = [i for i, c in enumerate(COLORS) if c in color_channels]
        self.n_cells = n_cells

        self.mask_prob = mask_prob
        self.epoch=0

        self.normalize = MinMaxNormalize()

        self.unique_cats = UNIQUE_CATS
        self.num_classes = NUM_CLASSES

        self.return_cell_mask = return_cell_mask
        self.split=split

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.df)
    
    def get_mask_tensor(self, masks):
        masks = (masks * 255).astype(np.uint8)
        masks = grey_dilation(grey_dilation(masks, size=(1, 5, 1)), size=(1, 1, 5))
        # masks = (masks / 255.0).astype(np.float32)
        masks = (masks / 255.0).astype(np.float16)
        masks = torch.from_numpy(masks[:, None, :, :])
        return masks

    def get_random_mask(self, mask_tensor):
        n_cells = mask_tensor.shape[0]
        n_non_masked = int(n_cells * (1 - self.mask_prob))
        mask_idxs = np.random.choice(n_cells, n_non_masked, replace=False)
        mask_tensor[mask_idxs] = 1.0
        return mask_tensor
    
    def get_select_idxs(self, cell_line, plate_postion, num_cells):
        if self.n_cells > 0 and num_cells > self.n_cells:
            df = pd.DataFrame(
                {
                    "cell_line": cell_line.split(";"),
                    "plate_position": plate_postion.split(";"),
                }
            )
            select_idxs = get_stratified_idxs(df, self.n_cells)
        elif self.n_cells > 0:
            # TODO: handle this such that we send the initial number + some randomly assigned n_cells. 
            select_idxs = np.arange(num_cells)
        else:
            select_idxs = np.arange(num_cells)
        return select_idxs

    def get_img_and_transform(self, img):

        # return torch.tensor(img).permute(2,0,1)

        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)

        # normalize
        img = self.normalize(img)

        # get the crops: will return a duo of images for each cell!
        if self.transform is not None:
            img = self.transform(img)

        return img
    

    def __getitem__(self, idx: int) -> Tuple:

        with h5py.File(self.df['filename'][idx], "r") as f:
            select_idxs = self.get_select_idxs(f.attrs["cell_line"], f.attrs["plate_position"], f.attrs["num_cells"])
            antibody_id = f.attrs["antibody"]

            """
            when using 448px, load with the mask. 
            Otherwise load wihtout the mask as it is already saved without!
            """
            valid_or_missing = self.split == 'valid' or 'imgs_256' not in self.df['filename'][idx]

            if self.channels == 'mt':
                slice_range = slice(1, -1 if valid_or_missing else None)
            elif self.channels == 'all':
                slice_range = slice(None, -1 if valid_or_missing else None)
            elif self.channels == 'random':
                r = torch.randint(0, 4, [])
                slice_range = slice(r, r + 1)
            
            images = [self.get_img_and_transform(f[f"img_{i}"][:, :, slice_range]) for i in select_idxs]

            if self.finetuning:
                target = np.array(f['target'])

        antibody_id_tensor = torch.tensor(
            [int(re.findall(r"\d+", antibody_id)[0])] * len(images)
        )

        if self.finetuning:
            target = torch.tensor(target)
            return torch.stack(images), torch.stack([target for _ in range(len(images))]).float()
        
        #  this is for pre-training only
        if self.transform is not None:
            flattened_images = [tensor for group in images for tensor in group]
            return flattened_images, antibody_id_tensor
        
        else:
            return torch.stack(images), antibody_id



class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class ImageFolderMask(HPASubCellDataset):

    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):

        super().__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)
                
        masks = []

        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)
    
class ImageFolderMaskCVIT(HPASubCellDatasetCVIT):

    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):

        super().__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)
                
        masks = []

        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)

