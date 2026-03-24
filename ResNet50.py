import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.model_selection import train_test_split
from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, get_loader

# RESNET-50 FEATURE EXTRACTION
# Using 'DEFAULT' weights ensures you get the best pre-trained version (ImageNet v2)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Remove the final classification layer
# In ResNet-50, the input to this layer is 2048 dimensions
resnet50.fc = nn.Identity() 

resnet50.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet50.to(device)


def extract_features_ResNet50(paths=None, labels=None):
    loader = get_loader(paths=paths, labels=labels)
    all_feats = []
    all_y = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            feats = resnet50(Xb)
            all_feats.append(feats.cpu().numpy())
            all_y.append(yb.numpy())

    X_feat_50 = np.concatenate(all_feats, axis=0)
    y_np_50 = np.concatenate(all_y, axis=0)
    return X_feat_50, y_np_50


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

X_train, X_test, y_train, y_test = extract_subsets_ResNet50(X_feat_50, y_np_50)
