"""
VGGFACE2 FEATURE EXTRACTION:

VGGFace2 here is used as a frozen feature extractor through
`facenet_pytorch.InceptionResnetV1`. With `classify=False`, the network outputs
512-dimensional face embeddings instead of class predictions.

Those embeddings can then be used in shallow sklearn models such as Logistic
Regression or SVM, in the same spirit as the MobileNet 
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
import seaborn as sns
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent


# Collect all image paths from a directory recursively.
def collect_paths(root, exts={".jpg", ".jpeg", ".png"}):
    root = Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


# Build the real/fake image pools used in this experiment.
paths_real = collect_paths(SCRIPT_DIR / "wiki")
paths_fake = (
    collect_paths(SCRIPT_DIR / "inpainting") +
    collect_paths(SCRIPT_DIR / "insight") +
    collect_paths(SCRIPT_DIR / "text2img")
)

print(f"Real: {len(paths_real)}, Fake: {len(paths_fake)}, Total: {len(paths_real)+len(paths_fake)}")

# Balanced subsample: keep the same number of real and fake images.
random.seed(42)
n = 500
paths_small = random.sample(paths_real, n) + random.sample(paths_fake, n)
y_small = [0] * n + [1] * n  # 0=real, 1=fake


# Remove unreadable / corrupted files before feature extraction.
def filter_valid_images(paths, labels):
    valid_paths, valid_labels = [], []
    for p, lab in zip(paths, labels):
        try:
            img = Image.open(p)
            img.verify()
            valid_paths.append(p)
            valid_labels.append(lab)
        except (UnidentifiedImageError, OSError):
            print(f"Bad image skipped: {p}")
    return valid_paths, valid_labels

paths_small, y_small = filter_valid_images(paths_small, y_small)
print(f"Valid images: {len(paths_small)}")


# InceptionResnetV1 expects 160x160 RGB images normalized to [-1, 1].
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# VGGFace2 feature extraction setup.
extractor = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
extractor.eval()

dataset_full = PathLabelDataset(paths_small, y_small, transform=transform)
loader = DataLoader(dataset_full, batch_size=32, shuffle=False, num_workers=0)

all_embeddings = []
all_labels = []

# Extract 512-dimensional embeddings for all selected images.
print("Extracting VGGFace2 embeddings ...")
with torch.no_grad():
    for Xb, yb in loader:
        Xb = Xb.to(device)
        emb = extractor(Xb)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(yb.numpy())

# Concatenate all embeddings and labels into single arrays.
X = np.vstack(all_embeddings)
y = np.concatenate(all_labels)
print(f"Embedding matrix: {X.shape}")


# SPLITTING INTO TRAIN/TEST
def extract_subsets_VGGFace2(X_feat=X, y_np=y, test_size=0.2, random_state=42):
    return train_test_split(
            X_feat,
            y_np,
            test_size=test_size,
            random_state=random_state,
            stratify=y_np,
            shuffle=True
        )