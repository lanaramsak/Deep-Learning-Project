"""
ENSEMBLE OF SINGLE-VIEW AND TWO-BRANCH CLASSIFIERS

This script trains several strong single-view classifiers on a shared split,
loads a previously trained two-branch checkpoint, and evaluates both
individual models and ensemble voting strategies on the same validation set.

The main motivation is to combine complementary model types:
- CNN-based single-view classifiers
- a Vision Transformer classifier
- a two-branch classifier that sees both the original image and a transformed
  secondary view

By keeping the validation split shared across models, the comparison and the
final ensemble evaluation remain aligned and methodologically consistent.
"""

from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from evaluation_metrics import get_eer_score, get_f1_score
from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, PathLabelDataset, build_transform
from MobileNet_model import get_trained_MobileNet_model
from ResNet50_model import get_trained_ResNet50_model
from TwoBranchResNet18 import TwoBranchResNet18, TwoBranchPathLabelDataset
from VisionTransformer import get_trained_ViT_model, vit_transform


device = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "ensemble_with_two_branch"


def create_shared_split(paths, labels, test_size=0.2, random_state=42):
    """
    Create one shared train/validation split for all models in the ensemble.

    This is important because ensemble members must be evaluated on the same
    validation examples in the same order; otherwise their predictions cannot
    be combined meaningfully.
    """

    return train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
        shuffle=True,
    )


def build_single_view_loaders(train_paths, val_paths, train_labels, val_labels, batch_size=32):
    """
    Build DataLoaders for the CNN-based single-view classifiers.

    These models use the standard image transform from the existing
    classification pipeline.
    """

    train_dataset = PathLabelDataset(train_paths, train_labels, transform=build_transform(224))
    val_dataset = PathLabelDataset(val_paths, val_labels, transform=build_transform(224))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_vit_loaders(train_paths, val_paths, train_labels, val_labels, batch_size=32):
    """
    Build DataLoaders for the Vision Transformer.

    ViT uses its own preprocessing transform so that inputs stay aligned with
    the pretrained weights used by the model.
    """

    train_dataset = PathLabelDataset(train_paths, train_labels, transform=vit_transform)
    val_dataset = PathLabelDataset(val_paths, val_labels, transform=vit_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_two_branch_val_loader(
    val_paths,
    val_labels,
    second_view_type="rotation_blur",
    blur_radius=2.0,
    rotation_degrees=15.0,
    batch_size=32,
):
    """
    Build the validation loader for the two-branch model.

    Only the validation loader is needed here because the two-branch model is
    assumed to be trained already and is loaded from a checkpoint.
    """

    dataset = TwoBranchPathLabelDataset(
        val_paths,
        val_labels,
        image_size=224,
        second_view_type=second_view_type,
        blur_radius=blur_radius,
        rotation_degrees=rotation_degrees,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def cnn_predict_fn(model, images):
    """
    Convert CNN logits to positive-class probabilities.
    """

    return torch.sigmoid(model(images)).squeeze()


def vit_predict_fn(model, images):
    """
    Convert ViT logits to positive-class probabilities.
    """

    logits = model(images)
    return torch.softmax(logits, dim=1)[:, 1]


def two_branch_predict_fn(model, x_original, x_second_view):
    """
    Convert two-branch logits to positive-class probabilities.
    """

    logits = model(x_original, x_second_view)
    return torch.softmax(logits, dim=1)[:, 1]


class EnsembleVoter:
    """
    Evaluate individual models and aggregate their predictions into an
    ensemble.

    Each model specification provides:
    - a trained model
    - a prediction function
    - a validation loader
    - a `kind` label describing whether it expects single-view or two-branch
      input
    """

    def __init__(self, model_specs, device):
        self.model_specs = model_specs
        self.device = device

    def _predict_single_model(self, spec):
        """
        Collect labels, hard predictions, and probabilities for one model.

        The logic branches depending on whether the model consumes a single
        image input or a two-branch `(original, transformed)` pair.
        """

        model = spec["model"]
        loader = spec["loader"]
        kind = spec["kind"]
        predict_fn = spec["predict_fn"]

        model.eval()
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                # The two-branch model expects two aligned views of the same
                # image, while the single-view models expect only one image.
                if kind == "two_branch":
                    x_original, x_second_view, labels = batch
                    x_original = x_original.to(self.device)
                    x_second_view = x_second_view.to(self.device)
                    probs = predict_fn(model, x_original, x_second_view)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    probs = predict_fn(model, images)

                # Convert probabilities to hard class predictions using the
                # standard 0.5 threshold.
                preds = (probs >= 0.5).long()
                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def evaluate(self, mode="soft"):
        """
        Evaluate all individual models and build an ensemble prediction.

        `mode="soft"` averages probabilities.
        `mode="hard"` averages binary decisions.
        """

        all_model_preds = {}
        all_model_probs = {}
        y_true = None
        rows = []

        print(f"\nRunning ensemble ({mode} voting)...")

        for spec in self.model_specs:
            labels, preds, probs = self._predict_single_model(spec)
            if y_true is None:
                y_true = labels
            else:
                # All models must predict on the exact same validation split
                # so that ensemble aggregation is well-defined.
                if not np.array_equal(y_true, labels):
                    raise ValueError("Model loaders are not aligned on the same validation split.")

            all_model_preds[spec["name"]] = preds
            all_model_probs[spec["name"]] = probs

            rows.append(
                {
                    "name": spec["name"],
                    "accuracy": accuracy_score(y_true, preds),
                    "f1": get_f1_score(y_true, preds),
                    "auc": roc_auc_score(y_true, probs),
                    "eer": get_eer_score(y_true, probs),
                }
            )
            print(f"  [{spec['name']}] predictions collected")

        # Hard voting combines already-thresholded predictions, while soft
        # voting combines probabilities before thresholding.
        if mode == "hard":
            stacked = np.stack(list(all_model_preds.values()), axis=0)
            final_preds = (stacked.mean(axis=0) >= 0.5).astype(int)
            final_probs = stacked.mean(axis=0)
        else:
            stacked_probs = np.stack(list(all_model_probs.values()), axis=0)
            final_probs = stacked_probs.mean(axis=0)
            final_preds = (final_probs >= 0.5).astype(int)

        ensemble_result = {
            "name": f"ensemble_{mode}",
            "accuracy": accuracy_score(y_true, final_preds),
            "f1": get_f1_score(y_true, final_preds),
            "auc": roc_auc_score(y_true, final_probs),
            "eer": get_eer_score(y_true, final_probs),
            "confusion_matrix": confusion_matrix(y_true, final_preds),
            "report": classification_report(y_true, final_preds, digits=4),
        }

        return rows, ensemble_result


def load_two_branch_model(checkpoint_path):
    """
    Load a trained two-branch checkpoint for ensemble evaluation.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TwoBranchResNet18(num_classes=2, pretrained=False, dropout=0.3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_metrics_csv(individual_rows, ensemble_results, output_dir):
    """
    Save summary metrics for both individual models and the ensemble.
    """

    output_path = output_dir / "ensemble_metrics.csv"
    fieldnames = ["name", "accuracy", "f1", "auc", "eer"]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in individual_rows + ensemble_results:
            writer.writerow({k: row[k] for k in fieldnames})
    return output_path


def save_report(ensemble_results, output_dir):
    """
    Save a more detailed text report for the ensemble variants.
    """

    output_path = output_dir / "ensemble_report.txt"
    sections = []
    for result in ensemble_results:
        sections.append(result["name"])
        sections.append(
            f"Accuracy: {result['accuracy']:.4f}\n"
            f"F1: {result['f1']:.4f}\n"
            f"AUC: {result['auc']:.4f}\n"
            f"EER: {result['eer']:.4f}\n"
            f"Confusion matrix:\n{result['confusion_matrix']}\n"
            f"Classification report:\n{result['report']}\n"
        )
        sections.append("=" * 72)
    output_path.write_text("\n".join(sections) + "\n")
    return output_path


def plot_metric_bars(individual_rows, ensemble_results, output_dir):
    """
    Plot side-by-side metric bars for all individual models and ensembles.
    """

    rows = individual_rows + ensemble_results
    labels = [row["name"] for row in rows]
    metrics = [
        ("accuracy", "Accuracy"),
        ("f1", "F1 Score"),
        ("auc", "ROC-AUC"),
        ("eer", "EER"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (metric_key, metric_label) in zip(axes.flat, metrics):
        values = [row[metric_key] for row in rows]
        axis.bar(labels, values, color="#4472C4")
        axis.set_title(metric_label)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(0, 1)

    fig.suptitle("Ensemble With Two-Branch Model")
    fig.tight_layout()
    output_path = output_dir / "ensemble_metric_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args():
    """
    Parse CLI arguments for the ensemble experiment.
    """

    parser = ArgumentParser(description="Train a shared-split ensemble and add the best two-branch model.")
    parser.add_argument(
        "--two-branch-checkpoint",
        type=Path,
        required=True,
        help="Path to the saved best checkpoint from two_branch_combo_experiment.py",
    )
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=15.0,
        help="Rotation used by the two-branch validation view.",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=2.0,
        help="Blur radius used by the two-branch validation view.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where ensemble outputs are saved.",
    )
    return parser.parse_args()


def main():
    """
    Run the full ensemble pipeline.

    The workflow is:
    1. build one shared split
    2. train the single-view models on that split
    3. load the trained two-branch checkpoint
    4. evaluate individual models
    5. evaluate hard and soft ensemble voting
    6. save metrics, report, and plots
    """

    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every model in the ensemble sees the same train/validation partition so
    # that their scores can be compared and combined fairly.
    train_paths, val_paths, train_labels, val_labels = create_shared_split(
        DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL
    )

    # CNN models and ViT use different preprocessing pipelines, so they get
    # separate loaders even though the underlying split is shared.
    train_loader, val_loader = build_single_view_loaders(
        train_paths, val_paths, train_labels, val_labels
    )
    vit_train_loader, vit_val_loader = build_vit_loaders(
        train_paths, val_paths, train_labels, val_labels
    )
    two_branch_val_loader = build_two_branch_val_loader(
        val_paths,
        val_labels,
        second_view_type="rotation_blur",
        blur_radius=args.blur_radius,
        rotation_degrees=args.rotation_degrees,
    )

    print(f"Running on device: {device}")
    print(f"Train size: {len(train_paths)}")
    print(f"Validation size: {len(val_paths)}")

    # Train the single-view ensemble members from scratch on the shared split.
    resnet50 = get_trained_ResNet50_model(train_loader, device)
    mobilenet = get_trained_MobileNet_model(train_loader, device)
    vit = get_trained_ViT_model(vit_train_loader, device)

    # The two-branch model is trained separately and then plugged into the
    # ensemble as an additional, structurally different member.
    two_branch_model = load_two_branch_model(args.two_branch_checkpoint)

    ensemble = EnsembleVoter(
        model_specs=[
            {"name": "ResNet50", "model": resnet50, "predict_fn": cnn_predict_fn, "loader": val_loader, "kind": "single"},
            {"name": "MobileNetV2", "model": mobilenet, "predict_fn": cnn_predict_fn, "loader": val_loader, "kind": "single"},
            {"name": "ViT", "model": vit, "predict_fn": vit_predict_fn, "loader": vit_val_loader, "kind": "single"},
            {"name": "TwoBranch", "model": two_branch_model, "predict_fn": two_branch_predict_fn, "loader": two_branch_val_loader, "kind": "two_branch"},
        ],
        device=device,
    )

    individual_rows, soft_result = ensemble.evaluate(mode="soft")
    _, hard_result = ensemble.evaluate(mode="hard")
    ensemble_results = [soft_result, hard_result]

    print("\nSummary")
    for row in individual_rows + ensemble_results:
        print(
            f"{row['name']:<16} "
            f"Acc: {row['accuracy']:.4f} "
            f"F1: {row['f1']:.4f} "
            f"AUC: {row['auc']:.4f} "
            f"EER: {row['eer']:.4f}"
        )

    metrics_path = save_metrics_csv(individual_rows, ensemble_results, output_dir)
    report_path = save_report(ensemble_results, output_dir)
    plot_path = plot_metric_bars(individual_rows, ensemble_results, output_dir)

    print()
    print(f"Saved ensemble metrics to: {metrics_path}")
    print(f"Saved ensemble report to: {report_path}")
    print(f"Saved ensemble plot to: {plot_path}")


if __name__ == "__main__":
    main()
