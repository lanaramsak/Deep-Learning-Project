from pathlib import Path
import random

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
random.seed(42)
N_SAMPLES_PER_CLASS = 500


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


def collect_paths(root, exts={".jpg", ".jpeg", ".png"}):
    root = Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


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


def get_sample_paths(n=N_SAMPLES_PER_CLASS):
    paths_real = collect_paths(SCRIPT_DIR.parent / "wiki")
    paths_fake = (
        collect_paths(SCRIPT_DIR.parent / "inpainting") +
        collect_paths(SCRIPT_DIR.parent / "insight") +
        collect_paths(SCRIPT_DIR.parent / "text2img")
    )

    paths_small = random.sample(paths_real, n) + random.sample(paths_fake, n)
    y_small = [0] * n + [1] * n
    return filter_valid_images(paths_small, y_small)


def build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_loader(image_size=224, batch_size=64, shuffle=False, num_workers=0, n=N_SAMPLES_PER_CLASS):
    paths_small, y_small = get_sample_paths(n=n)
    dataset = PathLabelDataset(paths_small, y_small, transform=build_transform(image_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
