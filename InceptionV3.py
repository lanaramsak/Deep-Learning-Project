import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import models

from import_data import get_loader


loader = get_loader(image_size=299)


# INCEPTION V3 FEATURE EXTRACTION
inception_v3 = models.inception_v3(
    weights=models.Inception_V3_Weights.DEFAULT
)
inception_v3.aux_logits = False
inception_v3.AuxLogits = None
inception_v3.fc = nn.Identity()

inception_v3.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
inception_v3.to(device)

all_feats = []
all_y = []

with torch.no_grad():
    for Xb, yb in loader:
        Xb = Xb.to(device)
        feats = inception_v3(Xb)
        all_feats.append(feats.cpu().numpy())
        all_y.append(yb.numpy())

X_feat_inception = np.concatenate(all_feats, axis=0)
y_np_inception = np.concatenate(all_y, axis=0)


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


X_train, X_test, y_train, y_test = extract_subsets_InceptionV3(
    X_feat_inception,
    y_np_inception
)
