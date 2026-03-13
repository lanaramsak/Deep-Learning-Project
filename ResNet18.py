import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.model_selection import train_test_split

from import_data import get_loader

loader = get_loader()

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()          # zdaj output = embedding (512 dim)
resnet.eval()
resnet.to("cpu")

all_feats = []
all_y = []

with torch.no_grad():
    for Xb, yb in loader:
        Xb = Xb.to("cpu")
        feats = resnet(Xb)              # (B, 512)
        all_feats.append(feats.cpu().numpy())
        all_y.append(yb.numpy())

X_feat = np.concatenate(all_feats, axis=0)
y_np   = np.concatenate(all_y, axis=0)

print(X_feat.shape, y_np.shape)

def extract_subsets_ResNet18(X_feat = X_feat, y_np = y_np, test_size=0.2, random_state=42):
    return train_test_split(
            X_feat,
            y_np,
            test_size=0.2,
            random_state=42,
            stratify=y_np,
            shuffle=True
        )

X_train, X_test, y_train, y_test = extract_subsets_ResNet18(X_feat, y_np)