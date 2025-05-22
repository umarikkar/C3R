import argparse
import os
from tqdm import tqdm
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    coverage_error,
    label_ranking_average_precision_score,
    roc_auc_score,
)
from models.vit_model import ViTPoolClassifier
import yaml
import torch.nn.functional as F
from scipy.ndimage import grey_dilation



plt.switch_backend("agg")

from utils.train_mlp import eval_model

UNIQUE_CATS = np.array(
    [
        cat
        for cat in pd.read_csv("HPA_feat/location_mapping.csv")[
            "Original annotation"
        ]
        .unique()
        .tolist()
        if cat
        not in ["Cleavage furrow", "Midbody ring", "Rods & Rings", "Microtubule ends"]
    ]
    + ["Negative"]
)


CHALLENGE_CATS = [
    "Nucleoplasm",
    "Nuclear membrane",
    "Nucleoli",
    "Nucleoli fibrillar center",
    "Nuclear speckles",
    "Nuclear bodies",
    "Endoplasmic reticulum",
    "Golgi apparatus",
    "Intermediate filaments",
    "Actin filaments",
    "Microtubules",
    "Mitotic spindle",
    "Centrosome",
    "Plasma membrane",
    "Mitochondria",
    "Aggresome",
    "Cytosol",
    "Vesicles",
    "Negative",
]

def filter_classes(df, unique_cats=UNIQUE_CATS):
    locations_list = df["locations"].str.split(",").tolist()
    labels_onehot = np.array(
        [[1 if cat in x else 0 for cat in unique_cats] for x in locations_list]
    )

    keep_idx = np.where(labels_onehot.sum(axis=1) > 0)[0]
    df = df.iloc[keep_idx].reset_index(drop=True)
    df[unique_cats] = labels_onehot[keep_idx]
    return df

def get_train_val_test_idx(df, unique_cats=UNIQUE_CATS):

    test_antibodies = pd.read_csv(
        "annotations/splits/test_antibodies.txt", header=None
    )[0].to_list()

    test_idxs = df[df["antibody"].isin(test_antibodies)].index.to_list()

    test_df=df.iloc[test_idxs]
    test_y = torch.from_numpy(df[unique_cats].iloc[test_idxs].values)
    return test_df, test_y


def get_atlas_name_classes(df):
    cell_lines = df["atlas_name"].unique()
    labels_onehot = pd.get_dummies(df["atlas_name"]).values
    return labels_onehot, cell_lines



def get_multilabel_df(df_true, df_pred):
    cols = df_true.columns

    avg_precisions = []
    aucs = []
    all_categories = []
    all_counts = []
    for cat in cols:
        if len(np.unique(df_true[cat])) != 2:
            continue
        avg_precision = average_precision_score(df_true[cat], df_pred[cat])
        avg_precisions.append(avg_precision)
        all_categories.append(cat)
        all_counts.append(df_true[cat].sum())
        auc = roc_auc_score(df_true[cat], df_pred[cat])
        aucs.append(auc)

    avg_precisions.append(average_precision_score(df_true.values, df_pred.values))
    aucs.append(roc_auc_score(df_true.values, df_pred.values))
    all_categories.append("Overall")
    all_counts.append(len(df_true))
    df_multilabel = (
        pd.DataFrame(
            {
                "Category": all_categories,
                "Average Precision": avg_precisions,
                "AUC": aucs,
                "Count": all_counts,
            }
        )
        .sort_values(by="Count", ascending=False)
        .reset_index(drop=True)
    )
    return df_multilabel


def plot_multilabel_metrics(
    df, metric="Average Precision", label="valid", save_folder="./"
):
    n_cats = len(df)
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(1, figsize=(16, 10))
    sns.barplot(
        x="Category",
        y=metric,
        hue="Category",
        palette=sns.color_palette(cc.glasbey_dark, n_cats),
        data=df,
        ax=ax,
        orient="v",
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.savefig(
        f"{save_folder}/{label}_{metric}.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close()


def get_metrics(save_folder, df_test, tag="test", unique_cats=UNIQUE_CATS):
    df_true = df_test[[col + "_true" for col in unique_cats]]
    df_true = df_true.rename(
        columns={col: col.replace("_true", "") for col in df_true.columns}
    )
    df_pred = df_test[[col + "_pred" for col in unique_cats]]
    df_pred = df_pred.rename(
        columns={col: col.replace("_pred", "") for col in df_pred.columns}
    )

    non_zero_cats = [col for col in unique_cats if df_true[col].sum() > 0]
    df_true = df_true[non_zero_cats]
    df_pred = df_pred[non_zero_cats]

    label_ranking_ap = label_ranking_average_precision_score(
        df_true.values, df_pred.values
    )
    coverage = coverage_error(df_true.values, df_pred.values)
    micro_avg_precision = average_precision_score(
        df_true.values, df_pred.values, average="micro"
    )

    df_multilabel = get_multilabel_df(df_true, df_pred)
    df_multilabel["Coverage Error"] = coverage
    df_multilabel["Label Ranking Average Precision"] = label_ranking_ap
    df_multilabel["Micro Average Precision"] = micro_avg_precision
    df_multilabel.to_csv(f"{save_folder}/{tag}_metrics.csv", index=False)

    plot_multilabel_metrics(
        df_multilabel,
        metric="Average Precision",
        label=tag,
        save_folder=save_folder,
    )
    plot_multilabel_metrics(
        df_multilabel, metric="AUC", label=tag, save_folder=save_folder
    )


def str2bool(v):
    return v.lower() in ("True", "true", "1")

def load_model_subcell(config):
    model = ViTPoolClassifier(config=config)
    encoder_path = config.get("vit_model")["encoder_path"]
    classifier_paths=config.get("vit_model")["classifier_paths"]
    model.load_model_dict(encoder_path, classifier_paths)
    return model

def min_max_standardize(im):
    min_val = torch.amin(im, dim=(1, 2, 3), keepdim=True)
    max_val = torch.amax(im, dim=(1, 2, 3), keepdim=True)

    im = (im - min_val) / (max_val - min_val + 1e-6)
    return im

def extract_features(image_df, model,config,unique_cats):
    image_folder = config["image_folder"]
    image_names = []

    with torch.no_grad():
        y_pred = []
        y_true = []
        for i,row in tqdm(image_df.iterrows(), total=len(image_df)):
            bbox_plate_id = row["if_plate_id"]
            bbox_position = row["position"]
            bbox_sample = row["sample"]
            bbox_label = row["cell_id"]
            name_str = str(bbox_plate_id) + "_" + str(bbox_position) + "_" + str(bbox_sample) + "_" + str(int(bbox_label))
            img_str= str(bbox_plate_id) + "_" + str(bbox_position) + "_" + str(bbox_sample)
            
            
            label=(row[unique_cats].values)
            label=label.astype(float)
            label=torch.from_numpy(label).reshape(1, -1)

            input_path = (f"{image_folder}/{bbox_plate_id}/{name_str}_cell_image.png")
            input_msk = (f"{image_folder}/{bbox_plate_id}/{name_str}_cell_mask.png")
            cell_size = (512, 512) 
            img = cv2.imread(input_path, -1)
            mask = cv2.imread(input_msk, -1)

            img = cv2.resize(img, cell_size, cv2.INTER_LINEAR)
            mask = cv2.resize(mask, cell_size, cv2.INTER_NEAREST)
            mask = grey_dilation(mask, size=(7, 7))
            mask= (mask/ 255.0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)

            # Convert NumPy arrays to PyTorch tensors
            image_tensor = (torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0))


            # Apply the mask: Zero out pixels where mask != 255
            img_masked = image_tensor * mask_tensor

            img_masked = min_max_standardize(img_masked)

            output = model(img_masked.to(device))
            out = output.probabilities.reshape(1, -1)
            y_pred.append(out.cpu())
            y_true.append(label.cpu())
            image_names.append(img_str)  # Ensures full names are stored correctly


        y_pred = torch.cat(y_pred).cpu().numpy()
        y_true = torch.cat(y_true).cpu().numpy()

        mean_avg_precision = average_precision_score(y_true, y_pred, average="macro")
    
    print(f"Mean Avg Precision: {mean_avg_precision:.5f}")

    output_path = "img_features.pth"
    torch.save({
        'y_pred': y_pred,  # Feature vectors
        'y_true': y_true,  # Corresponding image names
        'image_names': image_names
        }, output_path)

    print(f"Feature extraction completed! Data saved to {output_path}")
    df_pred = pd.DataFrame(y_pred, columns=unique_cats)
    df_true = pd.DataFrame(y_true, columns=unique_cats)

    df = pd.merge(
        df_true,
        df_pred,
        suffixes=("_true", "_pred"),
        left_index=True,
        right_index=True,
    )
    return df
                

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", "--features_folder", type=str, default="HPA_feat/FULL_MAE/")
    argparser.add_argument("-cc", "--classification_cats", type=str, default="locations")
    argparser.add_argument("-uc", "--unique_cats", type=str, default="all_unique_cats")
    argparser.add_argument("-c", "--config", help="path to configuration file")
    args = argparser.parse_args()

    features_folder = 'HPA_test_img_mask'
    classification_cats = args.classification_cats
    unique_cats_name = (
        args.unique_cats if classification_cats == "locations" else "atlas_name"
    )

    print(f"Parameters: {args}")

    save_folder = (f"{features_folder}/test")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = yaml.safe_load(open('model_config_im.yaml', 'r'))

    model_config = config["model_config"]
    model = load_model_subcell(model_config)
    model.to(device)
    model.eval()

    df = pd.read_csv(
        f"/work/SAMEED/Protein/Subcell/Datasets/HPA_cell_crops/metadata.csv",
        low_memory=False,
        index_col=0,
    )

    df.loc[df["locations"].isna(), "locations"] = "Negative"
    unique_cats = (UNIQUE_CATS if unique_cats_name == "all_unique_cats" else CHALLENGE_CATS)
    df = filter_classes(df, unique_cats=unique_cats)

    print(f"Found {len(df)} samples with {len(unique_cats)} unique categories: {unique_cats}")

    test_df, test_y = get_train_val_test_idx(df, unique_cats)
    
    
    cls_save_folder = f"{save_folder}/multiclass_{unique_cats_name}"
    os.makedirs(cls_save_folder, exist_ok=True)
    test_y = torch.from_numpy(df[unique_cats].values)

    df_res=extract_features(test_df, model,config,unique_cats)
    test_results = df_res.reindex(sorted(df_res.columns), axis=1)

    get_metrics(cls_save_folder, test_results, tag="test", unique_cats=UNIQUE_CATS)

      