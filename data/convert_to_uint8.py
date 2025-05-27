import pandas as pd
import glob
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import os

class myDset(Dataset):

    def __init__(self, pretrain_imgs, dtype='uint8'):
        super().__init__()

        self.img_list = list(pretrain_imgs['img_dir'])
        self.dtype = dtype

        self.out_dir = f'/scratch1/test_data/antibody_cell_imgs/train-pretrain_{dtype}'

    def __getitem__(self, index):

        filename = pretrain_imgs['img_dir'][index]
        antibody = pretrain_imgs['antibody'][index]

        img_name = filename.split('/')[-1]

        out_path = os.path.join(self.out_dir, antibody, img_name)

        if not os.path.exists(out_path):

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            img = cv2.imread(filename, -1)
            img = cv2.resize(img, (448, 448), cv2.INTER_LINEAR)

            if self.dtype == 'uint8' and img.dtype == np.uint16:
                img = (img/255).astype(np.uint8)
            elif self.dtype == 'uint16' and img.dtype == np.uint8:
                img = (img*255).astype(np.uint16)

            cv2.imwrite(out_path, img)

        return -1, -1
    
    def __len__(self):

        return len(self.img_list)
    

pretrain_imgs = pd.read_csv("pretrain_images.csv")
dset = myDset(pretrain_imgs)
loader = DataLoader(dset, batch_size=12, shuffle=False, num_workers=14, pin_memory=True)

for idx, data in enumerate(tqdm(loader)):

    pass

    # if idx==10:
    #     break


