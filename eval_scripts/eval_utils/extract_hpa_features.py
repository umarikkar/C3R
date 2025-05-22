import argparse
import os
import shutil
from ast import literal_eval
from glob import glob
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from tqdm import tqdm
from models.vit_model import ViTPoolClassifier
import warnings
warnings.filterwarnings("ignore")

from utils.jump_utils import (
    elim_corr,
    interpolated_precision_recall_curve,
    match_matrix,
    nn_accuracy,
    normalize_step,
    sim_matrix,
)
from utils.load_model_util import get_dino_model

CHANNEL_DICT = {"nuc": 0, "rna": 1, "er": 2, "agp": 3, "mito": 4}
DINO_CHANNEL_ORDER = ["nuc", "rna", "er", "agp", "mito"]

def load_model_subcell(config):
    model = ViTPoolClassifier(config=config)
    encoder_path = config.get("vit_model")["encoder_path"]
    classifier_paths=config.get("vit_model")["classifier_paths"]
    model.load_model_dict(encoder_path, classifier_paths)
    return model


def load_image_tensor(input_path):
    cell_size = (448, 448) 
    img = cv2.imread(input_path, -1)
    img = cv2.resize(img, cell_size, cv2.INTER_LINEAR)
    img = (torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0))
    return img


def pc_min_max_standardize(im):
    min_val = torch.amin(im, dim=(2, 3), keepdim=True)
    max_val = torch.amax(im, dim=(2, 3), keepdim=True)

    im = (im - min_val) / (max_val - min_val + 1e-6)
    return im


def min_max_standardize(im):
    min_val = torch.amin(im, dim=(1, 2, 3), keepdim=True)
    max_val = torch.amax(im, dim=(1, 2, 3), keepdim=True)

    im = (im - min_val) / (max_val - min_val + 1e-6)
    return im


def self_normalize(im):
    mean = torch.mean(im, dim=(2, 3), keepdim=True)
    std = torch.std(im, dim=(2, 3), keepdim=True, unbiased=False)
    im = (im - mean) / (std + 1e-7)
    return im


def load_image_dino(input_path):
    img = load_image_tensor(input_path)
    img = img[:, [CHANNEL_DICT[c] for c in DINO_CHANNEL_ORDER], :, :]
    img = pc_min_max_standardize(img)
    img = self_normalize(img)
    return img


def load_image_subcell(input_path, normalize="all_chan"):
  
    imgs = load_image_tensor(input_path)
    
    if normalize == "per_chan":
        imgs = pc_min_max_standardize(imgs)
    elif normalize == "all_chan":
        imgs = min_max_standardize(imgs)
    else:
        imgs = imgs / 255.0
    return imgs


def extract_features(config, image_df, device):
    image_folder = config["image_folder"]
    normalize = "all_chan"
    image_names = []
    all_features = None
    running_index = 0    
    labels = []
    antibody_id = []

    with torch.no_grad():
        model_config = config["model_config"]
        model = load_model_subcell(model_config)
        model.to(device)
        model.eval()
        for i, row in tqdm(image_df.iterrows(), total=len(image_df)):
            bbox_plate_id = row["if_plate_id"]
            bbox_position = row["position"]
            bbox_sample = row["sample"]
            bbox_label = row["cell_id"]
            location = row["locations"]
            antibody=row['antibody']

            name_str = str(bbox_plate_id) + "_" + str(bbox_position) + "_" + str(bbox_sample) + "_" + str(bbox_label)
            cell_image = load_image_subcell(f"{image_folder}/{bbox_plate_id}/{name_str}_cell_image.png",normalize)

            output = model(cell_image.to(device))
            features = output.pool_op.reshape(1, -1)

            if all_features == None:
                all_features = torch.zeros(len(image_df), features.shape[1])
            all_features[running_index : running_index + len(features), :] = features.detach().cpu()
            running_index += len(features)
            del cell_image, features

            image_names.append(name_str)  # Ensures full names are stored correctly
            labels.append(location);  # Ensures full labels are stored correctly
            antibody_id.append(antibody);


    output_path = "extracted_features.pth"
    torch.save({
        'features': all_features,  # Feature vectors
        'image_names': image_names,  # Corresponding image names
        'labels': labels,  # Extracted labels
        'antibody_id': antibody_id  # Extract
        }, output_path)

    print(f"Feature extraction completed! Data saved to {output_path}")


def get_well_embeddings(plate_paths, agg="mean"):
    plates = [os.path.basename(p).split(".")[0] for p in plate_paths]

    all_plates = []
    all_wells = []
    all_fovs = []
    all_features = []

    for i, plate in tqdm(enumerate(plates), total=len(plates)):
        plate_df, plate_features = torch.load(plate_paths[i], map_location="cpu")
        well_fov_groups = plate_df.groupby(["well", "fov"]).groups
        for well_fov, idxs in tqdm(well_fov_groups.items(), total=len(well_fov_groups)):
            well = well_fov[0]
            fov = well_fov[1]
            fov_features = plate_features[idxs, :]
            if agg == "mean":
                fov_agg_feature = torch.mean(fov_features, dim=0, keepdim=True)
            elif agg == "median":
                fov_agg_feature = torch.median(fov_features, dim=0, keepdim=True)[0]
            all_plates.append(plate)
            all_wells.append(well)
            all_fovs.append(fov)
            all_features.append(fov_agg_feature)

    all_features = torch.cat(all_features, dim=0)
    features_df = pd.DataFrame(
        {
            "plate": all_plates,
            "well": all_wells,
            "fov": all_fovs,
        }
    )
    feat_cols = [f"feature_{i}" for i in range(all_features.shape[1])]
    features_df.loc[:, feat_cols] = all_features.numpy()
    features_df = features_df.groupby(["plate", "well"])[feat_cols].mean().reset_index()
    features_df.to_csv(f"{output_folder}/all_features_agg_{agg}.csv", index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="SearchFirst config file path")
    argparser.add_argument("-c", "--config", help="path to configuration file")
    args = argparser.parse_args(["-c", "configs/rybg/mae_contrast_supcon_model/model_config.yaml"])
    config = yaml.safe_load(open(args.config))
    n_cores = 1
    image_df = pd.read_csv(
        f"/work/SAMEED/Protein/Subcell/subcell-embed-main/HPA_metadata.csv",
        low_memory=False,
        index_col=0,
    )

    split_dfs = np.array_split(image_df, n_cores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    output_folder = f"{config['output_folder']}/{config['model_name']}"
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(f"{output_folder}/all_features_agg.csv"):
        plate_paths = glob(f"{output_folder}/*.pth")
        if len(plate_paths) == 7:
            print("All plate features already saved, Aggregating well features")
        else:
            print("Extracting features")
            joblib.Parallel(n_jobs=n_cores)(
                joblib.delayed(extract_features)(config, split_df, device)
                for split_df in split_dfs
            )
            print("All features extracted, ")
        