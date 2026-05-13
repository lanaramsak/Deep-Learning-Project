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

from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, PathLabelDataset

SCRIPT_DIR = Path(__file__).resolve().parent

random.seed(42)

# InceptionResnetV1 expects 160x160 RGB images normalized to [-1, 1].
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGGFace2 feature extraction setup.
extractor = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
extractor.eval()

dataset_full = PathLabelDataset(DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, transform=transform)
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