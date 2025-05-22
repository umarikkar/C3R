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
from sklearn.metrics.pairwise import cosine_similarity

rep_agg_df = pd.read_csv("Jump_eval/rep_agg_df.csv")

features1 = rep_agg_df[rep_agg_df.columns[3:4611]]
features2=features1.to_numpy()
sim_matrix = cosine_similarity(features2)
sim_matrix = sim_matrix.clip(-1, 1)
sim_matrix[np.diag_indices_from(sim_matrix)] = -1

similarities = sim_matrix
comp_match_matrix = match_matrix(rep_agg_df, "compound")
plate_block_matrix = ~match_matrix(rep_agg_df, "plate")

# Interpolated Mean Average Precision
__, comp_average_precision = interpolated_precision_recall_curve(
    comp_match_matrix, similarities
)
compound_map = np.mean(comp_average_precision)
print('compuound_map:',compound_map)               


