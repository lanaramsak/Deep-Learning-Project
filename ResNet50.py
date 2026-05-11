"""RESNET-50 FEATURE EXTRACTION:

ResNet-50 is used as a frozen feature extractor. 
The final classification layer is replaced with nn.Identity(), meaning the model outputs raw feature vectors instead of class predictions.

We Later use these features to train simple models like Logistic Regression, SVM, etc. in the notebook.
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.model_selection import train_test_split
from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, get_loader

# RESNET-50 FEATURE EXTRACTION
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # pretrained model with default weights
resnet50.fc = nn.Identity() # This replaces the final classification layer with an identity function, so we get raw features instead of class scores
resnet50.eval() # Set the model to evaluation mode (important for layers like dropout or batchnorm)

device = "cuda" if torch.cuda.is_available() else "cpu"
resnet50.to(device)

# Function to extract features using ResNet-50
def extract_features_ResNet50(paths=None, labels=None):
    # Get the data loader for the given paths and labels
    loader = get_loader(paths=paths, labels=labels)
    all_feats = []
    all_y = []

    # We don't need gradients for feature extraction, so we wrap in torch.no_grad()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            feats = resnet50(Xb)
            all_feats.append(feats.cpu().numpy())
            all_y.append(yb.numpy())

    # Concatenate all the features and labels into single arrays
    X_feat_50 = np.concatenate(all_feats, axis=0)
    y_np_50 = np.concatenate(all_y, axis=0)
    return X_feat_50, y_np_50

# Extract features and labels using ResNet-50
X_feat_50, y_np_50 = extract_features_ResNet50(DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL)

# SPLITTING INTO TRAIN/TEST
def extract_subsets_ResNet50(X_feat = X_feat_50, y_np = y_np_50, test_size=0.2, random_state=42):
    return train_test_split(
            X_feat,
            y_np,
            test_size=0.2,
            random_state=42,
            stratify=y_np,
            shuffle=True
        )

# X_train, X_test, y_train, y_test = extract_subsets_ResNet50(X_feat_50, y_np_50)
