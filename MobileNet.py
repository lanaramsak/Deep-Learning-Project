"""
MOBILENETV2 FEATURE EXTRACTION:

MobileNet here is used as a frozen feature extractor. The classifier head is replaced with nn.Identity(), meaning the model just outputs raw feature vectors instead of class predictions. 
Those vectors are then fed into the shallow sklearn models in the notebook (Logistic Regression, SVM, etc.).

We Later use these features to train simple models like Logistic Regression, SVM, etc. in the notebook.
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.model_selection import train_test_split
from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, get_loader
from MobileNet_model import get_trained_MobileNet_model, get_loaders

# MobileNetV2 Feature Extraction setup
device = "cuda" if torch.cuda.is_available() else "cpu"

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT) # pretrained model
train_loader, test_loader = get_loaders()
# mobilenet = get_trained_MobileNet_model(train_loader, device) # Get the trained MobileNetV2 model (with frozen backbone and new head)
mobilenet.classifier = nn.Identity() # This replaces the final classification layer with an identity function, so we get raw features instead of class scores
mobilenet.eval()

mobilenet.to(device)

# Function to extract features using MobileNetV2
def extract_features_MobileNet(paths=None, labels=None):
    # Get the data loader for the given paths and labels
    loader = get_loader(paths=paths, labels=labels)
    all_feats = []
    all_y = []

    # We don't need gradients for feature extraction, so we wrap in torch.no_grad()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            feats = mobilenet(Xb)
            all_feats.append(feats.cpu().numpy())
            all_y.append(yb.numpy())

    # Concatenate all the features and labels into single arrays
    X_feat_mobile = np.concatenate(all_feats, axis=0)
    y_np_mobile = np.concatenate(all_y, axis=0)
    return X_feat_mobile, y_np_mobile

# Extract features and labels using MobileNetV2
X_feat_mobile, y_np_mobile = extract_features_MobileNet(DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL)

# SPLITTING INTO TRAIN/TEST
def extract_subsets_MobileNet(X_feat = X_feat_mobile, y_np = y_np_mobile, test_size=0.2, random_state=42):
    return train_test_split(
            X_feat,
            y_np,
            test_size=0.2,
            random_state=42,
            stratify=y_np,
            shuffle=True
        )
