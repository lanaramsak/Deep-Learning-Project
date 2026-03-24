from pathlib import Path
import random

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 42
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


def get_sample_paths(n=N_SAMPLES_PER_CLASS, seed=DEFAULT_SEED):
    paths_real = collect_paths(SCRIPT_DIR.parent / "wiki")
    paths_fake = (
        collect_paths(SCRIPT_DIR.parent / "inpainting") +
        collect_paths(SCRIPT_DIR.parent / "insight") +
        collect_paths(SCRIPT_DIR.parent / "text2img")
    )

    rng = random.Random(seed)
    paths_small = rng.sample(paths_real, n) + rng.sample(paths_fake, n)
    y_small = [0] * n + [1] * n
    return filter_valid_images(paths_small, y_small)


DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL = get_sample_paths()


def build_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_loader(
    paths=None,
    labels=None,
    image_size=224,
    batch_size=64,
    shuffle=False,
    num_workers=0,
):
    if paths is None or labels is None:
        paths = DEFAULT_PATHS_SMALL
        labels = DEFAULT_Y_SMALL
    dataset = PathLabelDataset(paths, labels, transform=build_transform(image_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
