"""EFFICIENTNET-B0 FEATURE EXTRACTION:

EfficientNet-B0 is used as a frozen feature extractor, similar to MobileNetV2. The final classification layer is replaced with nn.Identity(), allowing us to obtain raw feature vectors instead of class predictions. 
These features are then utilized to train simple models like Logistic Regression, SVM, etc. in the notebook.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import models

from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, get_loader


# EfficientNet-B0 Feature Extraction setup
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT) # pretrained model
efficientnet.classifier = nn.Identity() # This replaces the final classification layer with an identity function, so we get raw features instead of class scores

efficientnet.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
efficientnet.to(device)

# Function to extract features using EfficientNet-B0 for a given set of image paths and labels. 
def extract_features_EfficientNet(paths=None, labels=None):
    # Get the data loader for the given paths and labels
    loader = get_loader(paths=paths, labels=labels)
    all_feats = []
    all_y = []

    # We don't need gradients for feature extraction, so we wrap in torch.no_grad()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            feats = efficientnet(Xb)
            all_feats.append(feats.cpu().numpy())
            all_y.append(yb.numpy())

    # Concatenate all the features and labels into single arrays
    X_feat_efficientnet = np.concatenate(all_feats, axis=0)
    y_np_efficientnet = np.concatenate(all_y, axis=0)
    return X_feat_efficientnet, y_np_efficientnet


# Extract features and labels using EfficientNet-B0
X_feat_efficientnet, y_np_efficientnet = extract_features_EfficientNet(
    DEFAULT_PATHS_SMALL,
    DEFAULT_Y_SMALL
)

# SPLITTING INTO TRAIN/TEST
def extract_subsets_EfficientNet(
    X_feat=X_feat_efficientnet,
    y_np=y_np_efficientnet,
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


# X_train, X_test, y_train, y_test = extract_subsets_EfficientNet(
#     X_feat_efficientnet,
#     y_np_efficientnet
# )
