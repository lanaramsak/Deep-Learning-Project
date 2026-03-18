import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class PathLabelDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def collect_paths(root, exts={".jpg", ".jpeg", ".png"}):
    root = Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


SCRIPT_DIR = Path(__file__).resolve().parent
paths_real = collect_paths(SCRIPT_DIR.parent / "wiki")
paths_fake = (
    collect_paths(SCRIPT_DIR.parent / "inpainting") +
    collect_paths(SCRIPT_DIR.parent / "insight") +
    collect_paths(SCRIPT_DIR.parent / "text2img")
)

paths = paths_real + paths_fake
y = [0] * len(paths_real) + [1] * len(paths_fake)
random.seed(42)
n = 500

paths_small = random.sample(paths_real, n) + random.sample(paths_fake, n)
y_small = [0] * n + [1] * n


def filter_valid_images(paths, labels):
    valid_paths = []
    valid_labels = []

    for p, lab in zip(paths, labels):
        try:
            img = Image.open(p)
            img.verify()
            valid_paths.append(p)
            valid_labels.append(lab)
        except (UnidentifiedImageError, OSError):
            print(f"Bad image removed: {p}")

    return valid_paths, valid_labels


paths_small, y_small = filter_valid_images(paths_small, y_small)


# EFFICIENTNET-B0 FEATURE EXTRACTION
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
efficientnet.classifier = nn.Identity()

efficientnet.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
efficientnet.to(device)

dataset = PathLabelDataset(paths_small, y_small, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

all_feats = []
all_y = []

with torch.no_grad():
    for Xb, yb in loader:
        Xb = Xb.to(device)
        feats = efficientnet(Xb)      # Output shape: (Batch_Size, 1280)
        all_feats.append(feats.cpu().numpy())
        all_y.append(yb.numpy())

X_feat_efficientnet = np.concatenate(all_feats, axis=0)
y_np_efficientnet = np.concatenate(all_y, axis=0)


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
