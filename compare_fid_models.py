"""
COMPARE FID SCORES ACROSS TWO GENERATIVE MODELS

This script evaluates two image-generation models against the same real-image
reference set using the same FID formulation as `ddpm_pipeline_v0.7.py`.

Typical usage:
    python compare_fid_models.py \
        --real-dir /path/to/real_images \
        --model-a-dir /path/to/model_a_samples \
        --model-b-dir /path/to/model_b_samples \
        --label-a DCGAN \
        --label-b DDPM
"""

from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib.pyplot as plt
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "fid_comparison"
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_IMAGE_SIZE = 224


def resolve_input_dir(path_like):
    """
    Resolve an input directory by checking:
    - the given path as-is
    - the project root
    - the project parent directory

    This matches the layout used elsewhere in the repository, where folders
    such as `wiki` may live one level above `Deep-Learning-Project`.
    """

    path = Path(path_like)
    candidates = [
        path,
        SCRIPT_DIR / path,
        SCRIPT_DIR.parent / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return path


def _get_inception(device):
    """
    Match the Inception-based FID feature extractor used in ddpm_pipeline_v0.7.
    """

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.aux_logits = False
    model.eval()
    return model.to(device)


@torch.no_grad()
def _extract_features(images, inception, device, batch_size=64):
    """
    Extract Inception features after resizing to 299x299 and remapping
    image tensors from [-1, 1] to [0, 1].
    """

    features = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch + 1) / 2
        features.append(inception(batch).cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(real_images, fake_images, device):
    """
    Compute FID with the same formulation used in ddpm_pipeline_v0.7.
    """

    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required to compute FID.")

    inception = _get_inception(device)
    real_features = _extract_features(real_images, inception, device)
    fake_features = _extract_features(fake_images, inception, device)

    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    cov_sqrt, _ = scipy_sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt))
    del inception
    return fid


def collect_image_paths(image_dir):
    """
    Recursively collect image files from a directory.
    """

    image_dir = resolve_input_dir(image_dir)
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in VALID_SUFFIXES)


def load_images_as_tensor(image_paths, limit=None, image_size=DEFAULT_IMAGE_SIZE):
    """
    Load RGB images, resize them to a common size, and convert them to a
    tensor in the range [-1, 1].

    The downstream FID helper from ddpm_pipeline_v0.7.py expects tensors in
    this range and handles the final resize to Inception resolution itself.
    """

    if limit is not None:
        image_paths = image_paths[:limit]

    tensors = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = tensor * 2 - 1
        tensors.append(tensor)

    if not tensors:
        raise ValueError("No valid images were loaded.")

    return torch.stack(tensors, dim=0)


def save_results(results, output_dir):
    """
    Save the FID comparison summary as CSV and text.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "fid_scores.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["model", "fid", "n_fake", "n_real"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    txt_path = output_dir / "fid_report.txt"
    with txt_path.open("w") as report_file:
        report_file.write("FID comparison\n")
        report_file.write("=" * 40 + "\n")
        for row in results:
            report_file.write(
                f"{row['model']}: FID={row['fid']:.4f} | "
                f"fake_images={row['n_fake']} | real_images={row['n_real']}\n"
            )

    return csv_path, txt_path


def plot_results(results, output_dir):
    """
    Plot a simple FID comparison bar chart.
    """

    labels = [row["model"] for row in results]
    values = [row["fid"] for row in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["steelblue", "darkorange"])
    ax.set_title("FID Comparison")
    ax.set_ylabel("FID (lower is better)")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    plot_path = output_dir / "fid_comparison.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def parse_args():
    parser = ArgumentParser(description="Compare two generative models with FID.")
    parser.add_argument("--real-dir", type=Path, required=True, help="Directory with real reference images.")
    parser.add_argument("--model-a-dir", type=Path, required=True, help="Directory with samples from model A.")
    parser.add_argument("--model-b-dir", type=Path, required=True, help="Directory with samples from model B.")
    parser.add_argument("--label-a", type=str, default="Model A", help="Display name for model A.")
    parser.add_argument("--label-b", type=str, default="Model B", help="Display name for model B.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to use per folder. "
        "If omitted, the script uses the full shared count.",
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
    model_a_paths = collect_image_paths(args.model_a_dir)
    model_b_paths = collect_image_paths(args.model_b_dir)

    if not real_paths:
        raise ValueError(f"No images found in real-dir: {args.real_dir}")
    if not model_a_paths:
        raise ValueError(f"No images found in model-a-dir: {args.model_a_dir}")
    if not model_b_paths:
        raise ValueError(f"No images found in model-b-dir: {args.model_b_dir}")

    if args.limit is None:
        shared_count = min(len(real_paths), len(model_a_paths), len(model_b_paths))
    else:
        shared_count = min(args.limit, len(real_paths), len(model_a_paths), len(model_b_paths))

    if shared_count < 2:
        raise ValueError("Need at least 2 images in each directory for a valid FID comparison.")

    real_images = load_images_as_tensor(real_paths, limit=shared_count, image_size=args.image_size)
    model_a_images = load_images_as_tensor(model_a_paths, limit=shared_count, image_size=args.image_size)
    model_b_images = load_images_as_tensor(model_b_paths, limit=shared_count, image_size=args.image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid_a = compute_fid(real_images, model_a_images, device)
    fid_b = compute_fid(real_images, model_b_images, device)

    results = [
        {"model": args.label_a, "fid": fid_a, "n_fake": shared_count, "n_real": shared_count},
        {"model": args.label_b, "fid": fid_b, "n_fake": shared_count, "n_real": shared_count},
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, txt_path = save_results(results, args.output_dir)
    plot_path = plot_results(results, args.output_dir)

    print(f"Compared {shared_count} real images against {shared_count} fake images per model.")
    for row in results:
        print(f"{row['model']}: FID = {row['fid']:.4f}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved report to: {txt_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
