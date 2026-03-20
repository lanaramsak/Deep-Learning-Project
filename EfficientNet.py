import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import models

from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, get_loader


# EFFICIENTNET-B0 FEATURE EXTRACTION
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
efficientnet.classifier = nn.Identity()

efficientnet.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
efficientnet.to(device)


def extract_features_EfficientNet(paths=None, labels=None):
    loader = get_loader(paths=paths, labels=labels)
    all_feats = []
    all_y = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            feats = efficientnet(Xb)
            all_feats.append(feats.cpu().numpy())
            all_y.append(yb.numpy())

    X_feat_efficientnet = np.concatenate(all_feats, axis=0)
    y_np_efficientnet = np.concatenate(all_y, axis=0)
    return X_feat_efficientnet, y_np_efficientnet


X_feat_efficientnet, y_np_efficientnet = extract_features_EfficientNet(
    DEFAULT_PATHS_SMALL,
    DEFAULT_Y_SMALL
)


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


X_train, X_test, y_train, y_test = extract_subsets_EfficientNet(
    X_feat_efficientnet,
    y_np_efficientnet
)
