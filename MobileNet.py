import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.model_selection import train_test_split
from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, get_loader

# MOBILENETV2 FEATURE EXTRACTION
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# In MobileNetV2, the "head" is called 'classifier' instead of 'fc'
# We replace the classifier with an Identity layer to get the 1280-dim features
mobilenet.classifier = nn.Identity()

mobilenet.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
mobilenet.to(device)


def extract_features_MobileNet(paths=None, labels=None):
    loader = get_loader(paths=paths, labels=labels)
    all_feats = []
    all_y = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            feats = mobilenet(Xb)
            all_feats.append(feats.cpu().numpy())
            all_y.append(yb.numpy())

    X_feat_mobile = np.concatenate(all_feats, axis=0)
    y_np_mobile = np.concatenate(all_y, axis=0)
    return X_feat_mobile, y_np_mobile


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
