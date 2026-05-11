"""RESNET-18 FEATURE EXTRACTION:

ResNet-18 is used as a frozen feature extractor. 
The final classification layer is replaced with nn.Identity(), meaning the model outputs raw feature vectors instead of.

We Later use these features to train simple models like Logistic Regression, SVM, etc. in the notebook.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import models

from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, get_loader

# ResNet-18 Feature Extraction setup
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.fc = nn.Identity() # This replaces the final classification layer with an identity function, so we get raw features instead of class scores
resnet18.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
resnet18.to(device)

# Function to extract features using ResNet-18
def extract_features_ResNet18(paths=None, labels=None):
    # Get the data loader for the given paths and labels
    loader = get_loader(paths=paths, labels=labels)
    all_feats = []
    all_y = []

    # We don't need gradients for feature extraction, so we wrap in torch.no_grad()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            feats = resnet18(Xb)
            all_feats.append(feats.cpu().numpy())
            all_y.append(yb.numpy())

    # Concatenate all the features and labels into single arrays
    X_feat_18 = np.concatenate(all_feats, axis=0)
    y_np_18 = np.concatenate(all_y, axis=0)
    return X_feat_18, y_np_18

# Extract features and labels using ResNet-18
X_feat_18, y_np_18 = extract_features_ResNet18(DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL)

# SPLITTING INTO TRAIN/TEST
def extract_subsets_ResNet18(X_feat=X_feat_18, y_np=y_np_18, test_size=0.2, random_state=42):
    return train_test_split(
        X_feat,
        y_np,
        test_size=test_size,
        random_state=random_state,
        stratify=y_np,
        shuffle=True
    )


# X_train, X_test, y_train, y_test = extract_subsets_ResNet18(X_feat_18, y_np_18)