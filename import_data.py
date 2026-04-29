from pathlib import Path
import random

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 42
N_SAMPLES_PER_CLASS = 500


def resolve_data_dir(name):
    candidates = [SCRIPT_DIR / name, SCRIPT_DIR.parent / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
    paths_real = collect_paths(resolve_data_dir("wiki"))
    paths_fake = (
        collect_paths(resolve_data_dir("inpainting")) +
        collect_paths(resolve_data_dir("insight")) +
        collect_paths(resolve_data_dir("text2img"))
    )

    if not paths_real or not paths_fake:
        raise ValueError(
            "Could not build sample set. "
            f"Found {len(paths_real)} real images and {len(paths_fake)} fake images."
        )

    sample_size = min(n, len(paths_real), len(paths_fake))
    if sample_size < n:
        print(
            f"Requested {n} samples per class, but only {sample_size} are available. "
            "Using the maximum balanced subset instead."
        )

    rng = random.Random(seed)
    paths_small = rng.sample(paths_real, sample_size) + rng.sample(paths_fake, sample_size)
    y_small = [0] * sample_size + [1] * sample_size
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
