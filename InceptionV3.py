"""INCEPTIONV3 FEATURE EXTRACTION:

InceptionV3 is used here as a frozen feature extractor.
The final classification layer is replaced with nn.Identity(), meaning the
network outputs raw feature vectors instead of class predictions.

We later use these features to train simple models like Logistic Regression,
SVM, etc. in the notebook or benchmark scripts.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import models

from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, get_loader


# InceptionV3 Feature Extraction setup
inception_v3 = models.inception_v3(
    weights=models.Inception_V3_Weights.DEFAULT
)
inception_v3.aux_logits = False
inception_v3.AuxLogits = None
inception_v3.fc = nn.Identity()  # Replace the classifier head so we get raw features instead of class scores

inception_v3.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
inception_v3.to(device)


def extract_features_InceptionV3(paths=None, labels=None):
    # Get the data loader for the given paths and labels.
    loader = get_loader(paths=paths, labels=labels, image_size=299)
    all_feats = []
    all_y = []

    # We don't need gradients for feature extraction, so we wrap in torch.no_grad()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            feats = inception_v3(Xb)
            all_feats.append(feats.cpu().numpy())
            all_y.append(yb.numpy())

    # Concatenate all the features and labels into single arrays
    X_feat_inception = np.concatenate(all_feats, axis=0)
    y_np_inception = np.concatenate(all_y, axis=0)
    return X_feat_inception, y_np_inception


# Extract features and labels using InceptionV3
X_feat_inception, y_np_inception = extract_features_InceptionV3(
    DEFAULT_PATHS_SMALL,
    DEFAULT_Y_SMALL
)


# SPLITTING INTO TRAIN/TEST
def extract_subsets_InceptionV3(
    X_feat=X_feat_inception,
    y_np=y_np_inception,
    test_size=0.2,
    random_state=42
):
    return train_test_split(
        X_feat,
        y_np,
        test_size=test_size,
        random_state=random_state,
        stratify=y_np,
        shuffle=True
    )

# X_train, X_test, y_train, y_test = extract_subsets_InceptionV3(
#     X_feat_inception,
#     y_np_inception
# )
