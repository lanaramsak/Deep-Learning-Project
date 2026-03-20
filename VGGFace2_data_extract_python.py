from pathlib import Path
import random
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 1. DATA COLLECTION
# ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

def collect_paths(root, exts={".jpg", ".jpeg", ".png"}):
    root = Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

paths_real = collect_paths(SCRIPT_DIR / "wiki")
paths_fake = (
    collect_paths(SCRIPT_DIR / "inpainting") +
    collect_paths(SCRIPT_DIR / "insight") +
    collect_paths(SCRIPT_DIR / "text2img")
)

print(f"Real: {len(paths_real)}, Fake: {len(paths_fake)}, Total: {len(paths_real)+len(paths_fake)}")

# Balanced subsample
random.seed(42)
n = 500
paths_small = random.sample(paths_real, n) + random.sample(paths_fake, n)
y_small = [0] * n + [1] * n  # 0=real, 1=fake


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


# ─────────────────────────────────────────────
# 2. DATASET & TRANSFORMS
# ─────────────────────────────────────────────
# InceptionResnetV1 expects 160×160, normalised to [-1, 1]
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


# ─────────────────────────────────────────────
# 3. VGGFace2 FEATURE EXTRACTION
#    InceptionResnetV1 with classify=False outputs
#    512-dimensional face embeddings.
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

extractor = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
extractor.eval()

dataset_full = PathLabelDataset(paths_small, y_small, transform=transform)
loader = DataLoader(dataset_full, batch_size=32, shuffle=False, num_workers=0)

all_embeddings = []
all_labels = []

print("Extracting VGGFace2 embeddings …")
with torch.no_grad():
    for Xb, yb in loader:
        Xb = Xb.to(device)
        emb = extractor(Xb)           # shape: (B, 512)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(yb.numpy())

X = np.vstack(all_embeddings)   # (N, 512)
y = np.concatenate(all_labels)  # (N,)
print(f"Embedding matrix: {X.shape}")


# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Standardise (helps linear classifiers)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")


# ─────────────────────────────────────────────
# 5. SIMPLE CLASSIFICATION
# ─────────────────────────────────────────────
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "SVM (RBF)":           SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
}

results = {}
for name, clf in classifiers.items():
    print(f"\nTraining {name} …")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"acc": acc, "y_pred": y_pred, "clf": clf}
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
n_clf = len(classifiers)
fig, axes = plt.subplots(1, n_clf, figsize=(6 * n_clf, 5))
if n_clf == 1:
    axes = [axes]

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    ax.set_title(f"{name}\nAcc: {res['acc']:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.suptitle("VGGFace2 Embedding – Simple Classifiers", fontsize=14, y=1.02)
plt.tight_layout()
plot_path = SCRIPT_DIR / "vggface2_classification_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {plot_path}")
