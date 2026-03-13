from pathlib import Path
import random
from PIL import Image, UnidentifiedImageError

def collect_paths(root, exts={".jpg",".jpeg",".png"}):
    root = Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

# paths_real = collect_paths("data/wiki")
# paths_fake = (
#     collect_paths("data/inpainting") +
#     collect_paths("data/insight") +
#     collect_paths("data/text2img")
# )

# Maja path
SCRIPT_DIR = Path(__file__).resolve().parent
paths_real = collect_paths(SCRIPT_DIR.parent / "wiki")
paths_fake = (
    collect_paths(SCRIPT_DIR.parent / "inpainting") +
    collect_paths(SCRIPT_DIR.parent / "insight") +
    collect_paths(SCRIPT_DIR.parent / "text2img")
)

paths = paths_real + paths_fake
y = [0]*len(paths_real) + [1]*len(paths_fake)

print(len(paths_real), len(paths_fake), len(paths))

random.seed(42)

n = 500   # koliko real + fake

paths_small = random.sample(paths_real, n) + random.sample(paths_fake, n)
y_small = [0]*n + [1]*n

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

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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

dataset = PathLabelDataset(paths_small, y_small, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

def get_loader():
    return loader