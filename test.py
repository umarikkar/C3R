import pandas as pd
import glob
from tqdm import tqdm
import os

from data.hpa_dataset import HPASubCellDataset_saver

import cv2
from torch.utils.data import DataLoader


dset = HPASubCellDataset_saver(root='/scratch1/test_data/antibody_cell_imgs', n_cells=-1, split='train')

loader = DataLoader(dset, batch_size=1, num_workers=8)

for data in tqdm(loader):
    pass


# dataset_dir = '/work/SAMEED/Protein/Subcell/Datasets/HPA_cell_crops'
# pretrain_images = pd.read_csv('data/pretrain_images.csv')

# png_files = []
# for idx, img_dir in enumerate(pretrain_images['img_dir']):
#     png_files.append(os.path.join(dataset_dir, img_dir))

#     if idx==0:
#         break

# from PIL import Image

# img = Image.open(png_files[0])
# img_cv = cv2.imread(png_files[0], -1)

# print('x')

# # Load antibody list
# with open("data/annotations/splits/train_antibodies.txt", "r") as f:
#     antibodies = f.read().splitlines()
# antibodies = pd.DataFrame({"antibody": antibodies})

# # Load and filter metadata
# metadata = pd.read_csv("data/HPA_metadata.csv")
# filtered_metadata = metadata[metadata['antibody'].isin(antibodies['antibody'])]

# # Prepare mapping key for merging
# filtered_metadata["merge_key"] = (
#     filtered_metadata["if_plate_id"].astype(str) + "_" +
#     filtered_metadata["position"] + "_" +
#     filtered_metadata["sample"].astype(str) + "_" +
#     filtered_metadata["cell_id"].astype(str)
# )

# # Create a mapping from merge_key to image path
# img_paths = []
# keys = []



# for img_dir in tqdm(glob.glob(f"{dataset_dir}/*/*_cell_image.png")):


#     img_name = img_dir.split('/')[-1]

#     dir_to_save = ('/').join(img_dir.split('/')[-2:])

#     plate_id, position, sample, cell_id, *_ = img_name.split('_')
#     key = f"{plate_id}_{position}_{sample}_{cell_id}"
#     keys.append(key)
#     img_paths.append(dir_to_save)

# img_df = pd.DataFrame({"merge_key": keys, "img_dir": img_paths})

# # Merge with metadata
# final_df = pd.merge(filtered_metadata, img_df, on="merge_key", how="inner")

# # Save to CSV
# final_df.to_csv("pretrain_images.csv", index=False)

    

