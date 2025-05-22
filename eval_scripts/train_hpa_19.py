import argparse
import os
import shutil
import random

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
# from harmony import harmonize
from sklearn.metrics import (
    average_precision_score,
    coverage_error,
    label_ranking_average_precision_score,
    roc_auc_score,
)

plt.switch_backend("agg")

from eval_scripts.eval_utils.train_mlp import train_mlp, eval_model

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

def filter_classes(df, feature_data, unique_cats=CHALLENGE_CATS):
    locations_list = df["locations"].str.split(",").tolist()
    labels_onehot = np.array(
        [[1 if cat in x else 0 for cat in unique_cats] for x in locations_list]
    )

    keep_idx = np.where(labels_onehot.sum(axis=1) > 0)[0]
    df = df.iloc[keep_idx].reset_index(drop=True)
    df[unique_cats] = labels_onehot[keep_idx]
    feature_data = feature_data[keep_idx]

    return df, feature_data


def get_atlas_name_classes(df):
    cell_lines = df["atlas_name"].unique()
    labels_onehot = pd.get_dummies(df["atlas_name"]).values
    return labels_onehot, cell_lines


def get_train_val_test_idx(df, feature_data, unique_cats=CHALLENGE_CATS):
    train_antibodies = pd.read_csv(
        "annotations/splits/train_antibodies.txt", header=None
    )[0].to_list()
    val_antibodies = pd.read_csv(
        "annotations/splits/valid_antibodies.txt", header=None
    )[0].to_list()
    test_antibodies = pd.read_csv(
        "annotations/splits/test_antibodies.txt", header=None
    )[0].to_list()
    train_idxs = df[df["antibody"].isin(train_antibodies)].index.to_list()
    val_idxs = df[df["antibody"].isin(val_antibodies)].index.to_list()
    test_idxs = df[df["antibody"].isin(test_antibodies)].index.to_list()

    train_x = feature_data[train_idxs]
    train_y = torch.from_numpy(df[unique_cats].iloc[train_idxs].values)

    val_x = feature_data[val_idxs]
    val_y = torch.from_numpy(df[unique_cats].iloc[val_idxs].values)

    test_x = feature_data[test_idxs]
    test_y = torch.from_numpy(df[unique_cats].iloc[test_idxs].values)
    return train_x, train_y, val_x, val_y, test_x, test_y


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


def get_metrics(save_folder, df_test, tag="test", unique_cats=CHALLENGE_CATS):
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



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", "--features_folder", type=str, default="HPA_feat/FULL_MAE/")
    argparser.add_argument("-hf", "--harmonize_features", type=str2bool, default=False)
    argparser.add_argument(
        "-cc", "--classification_cats", type=str, default="locations"  # "atlas_name"
    )
    argparser.add_argument("-uc", "--unique_cats", type=str, default="all_unique_cats")

    args = argparser.parse_args()

    features_folder = f"results/HPA_test_19classes/{args.features_folder.replace('.pth', '').split('/')[-1]}/Classifier"

    os.makedirs(features_folder, exist_ok=True)

    harmonize_features = args.harmonize_features
    classification_cats = args.classification_cats
    unique_cats_name = (
        args.unique_cats if classification_cats == "locations" else "atlas_name"
    )

    print(f"Parameters: {args}")

    save_folder = (
        f"{features_folder}/classification"
        if not harmonize_features
        else f"{features_folder}/classification_aligned"
    )
    # shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)

    data = torch.load(f"{args.features_folder}", map_location="cpu")
    feature_data = data['features']
    df = pd.DataFrame(
        {
            "images": data['image_names'],
            "antibody": data['antibody_id'],
            "locations": data['labels']
        }
    )
    df=df.replace(regex=['Cell Junctions'],value='Plasma membrane')
    df=df.replace(regex=['Centriolar satellite'],value='Centrosome')
    df=df.replace(regex=['Cytoplasmic bodies'],value='Cytosol')
    df=df.replace(regex=['Focal adhesion sites'],value='Actin filaments')
    df=df.replace(regex=['Nucleoli rim'],value='Nucleoli')
    df=df.replace(regex=['Endosomes','Lysosomes','Lipid droplets','Peroxisomes'],value='Vesicles')
    df.loc[df["locations"].isna(), "locations"] = "Negative"

    df, feature_data = filter_classes(df, feature_data, unique_cats=CHALLENGE_CATS)

    train_x, train_y, val_x, val_y, test_x, test_y = get_train_val_test_idx(
        df, feature_data, CHALLENGE_CATS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(5):
        np.random.seed(i)
        random.seed(i)
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)

        cls_save_folder = f"{save_folder}/multiclass_{unique_cats_name}/seed_{i}"
        os.makedirs(cls_save_folder, exist_ok=True)

        if not os.path.isfile(f"{save_folder}/test_preds.csv"):
            model = train_mlp(
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
                device,
                CHALLENGE_CATS,
                cls_save_folder,
            )
            val_results = eval_model(
                val_x, val_y, CHALLENGE_CATS, model, seed=i, device=device
            )
            val_results.to_csv(f"{cls_save_folder}/val_preds.csv", index=False)

            test_results = eval_model(
                test_x, test_y, CHALLENGE_CATS, model, seed=i, device=device
            )
            test_results.to_csv(f"{cls_save_folder}/test_preds.csv", index=False)

            get_metrics(
                cls_save_folder, val_results, tag="val", unique_cats=CHALLENGE_CATS
            )
            get_metrics(
                cls_save_folder, test_results, tag="test", unique_cats=CHALLENGE_CATS
            )
            
        else:
            val_results = pd.read_csv(f"{cls_save_folder}/val_preds.csv")
            test_results = pd.read_csv(f"{cls_save_folder}/test_preds.csv")

            get_metrics(
                cls_save_folder, val_results, tag="val", unique_cats=CHALLENGE_CATS
            )
            get_metrics(
                cls_save_folder, test_results, tag="test", unique_cats=CHALLENGE_CATS
            )
            