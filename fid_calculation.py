"""
Compute a single FID score between a real-image directory and a generated-image
directory.

Typical usage:
    python fid_calculation.py \
        --real-dir ../wiki \
        --fake-dir Image_Generation/upgrade_epochs_100_fid_test/fid_samples \
        --label DCGAN
"""

from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

try:
    from scipy.linalg import sqrtm as scipy_sqrtm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "fid_single"
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_IMAGE_SIZE = 224


def resolve_input_dir(path_like):
    path = Path(path_like)
    candidates = [path, SCRIPT_DIR / path, SCRIPT_DIR.parent / path]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return path


def collect_image_paths(image_dir):
    image_dir = resolve_input_dir(image_dir)
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in VALID_SUFFIXES)


def load_images_as_tensor(image_paths, limit=None, image_size=DEFAULT_IMAGE_SIZE):
    if limit is not None:
        image_paths = image_paths[:limit]

    tensors = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append(tensor * 2 - 1)

    if not tensors:
        raise ValueError("No valid images were loaded.")

    return torch.stack(tensors, dim=0)


def get_inception(device):
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.aux_logits = False
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_features(images, inception, device, batch_size=64):
    features = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch + 1) / 2
        features.append(inception(batch).cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(real_images, fake_images, device):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required to compute FID.")

    inception = get_inception(device)
    real_features = extract_features(real_images, inception, device)
    fake_features = extract_features(fake_images, inception, device)

    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    cov_sqrt, _ = scipy_sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    return float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt))


def save_result(label, fid_value, shared_count, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "fid_score.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["model", "fid", "n_fake", "n_real"])
        writer.writeheader()
        writer.writerow({
            "model": label,
            "fid": fid_value,
            "n_fake": shared_count,
            "n_real": shared_count,
        })

    txt_path = output_dir / "fid_report.txt"
    with txt_path.open("w") as report_file:
        report_file.write("FID result\n")
        report_file.write("=" * 40 + "\n")
        report_file.write(
            f"{label}: FID={fid_value:.4f} | fake_images={shared_count} | real_images={shared_count}\n"
        )

    return csv_path, txt_path


def parse_args():
    parser = ArgumentParser(description="Compute a single FID score.")
    parser.add_argument("--real-dir", type=Path, required=True, help="Directory with real reference images.")
    parser.add_argument("--fake-dir", type=Path, required=True, help="Directory with generated images.")
    parser.add_argument("--label", type=str, default="Generated", help="Name used in the saved result.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to use. If omitted, the smaller folder size is used.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Temporary resize applied before stacking images into a tensor.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()

    real_paths = collect_image_paths(args.real_dir)
    fake_paths = collect_image_paths(args.fake_dir)

    if not real_paths:
        raise ValueError(f"No images found in real-dir: {args.real_dir}")
    if not fake_paths:
        raise ValueError(f"No images found in fake-dir: {args.fake_dir}")

    if args.limit is None:
        shared_count = min(len(real_paths), len(fake_paths))
    else:
        shared_count = min(args.limit, len(real_paths), len(fake_paths))

    if shared_count < 2:
        raise ValueError("Need at least 2 images in each directory for a valid FID computation.")

    real_images = load_images_as_tensor(real_paths, limit=shared_count, image_size=args.image_size)
    fake_images = load_images_as_tensor(fake_paths, limit=shared_count, image_size=args.image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid_value = compute_fid(real_images, fake_images, device)
    csv_path, txt_path = save_result(args.label, fid_value, shared_count, args.output_dir)

    print(f"Compared {shared_count} real images against {shared_count} fake images.")
    print(f"{args.label}: FID = {fid_value:.4f}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved report to: {txt_path}")


if __name__ == "__main__":
    main()
