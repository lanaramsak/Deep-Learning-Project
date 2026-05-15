"""
VISION TRANSFORMER TRAINING AND EVALUATION

This module provides a standalone Vision Transformer (ViT) pipeline for the
binary real-vs-fake image classification task used in this project.

The file serves two purposes:
1. It can be imported as a helper module by other scripts, such as ensemble
   experiments.
2. It can be executed directly to run a full ViT experiment end-to-end:
   data loading, training, evaluation, checkpoint saving, and plotting.

Unlike the earlier Hugging Face-based ViT version, this implementation uses
`torchvision.models.vit_b_16`, which keeps it compatible with the current
PyTorch environment used in the repository. - 2.2.2
"""

from argparse import ArgumentParser
import copy
import csv
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import models, transforms

from evaluation_metrics import get_eer_score, get_f1_score
from import_data import DEFAULT_SEED, PathLabelDataset, get_sample_paths


random.seed(DEFAULT_SEED)
N_SAMPLES_PER_CLASS = 5000
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "vit_experiment"

# Use the default pretrained torchvision ViT weights.
# We also reuse the matching preprocessing transform so the input pipeline
# stays aligned with how the model was originally trained.
VIT_WEIGHTS = models.ViT_B_16_Weights.DEFAULT
vit_transform = VIT_WEIGHTS.transforms()

# Explicit preprocessing pipeline for controlled experiments. The evaluation
# transform stays fixed, while the training transform can optionally include
# light augmentations such as color jitter and Gaussian blur.
VIT_IMAGE_SIZE = 224
VIT_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
VIT_NORMALIZE_STD = (0.229, 0.224, 0.225)


def build_vit_transform(use_color_jitter=False, use_blur=False):
    """
    Build a ViT-compatible transform pipeline.

    The baseline preprocessing is kept identical to the pretrained ImageNet
    setup. Optional train-time augmentations can be inserted before tensor
    conversion so we can test whether they improve the strong single-view ViT
    baseline without changing the rest of the pipeline.
    """

    transform_steps = [
        transforms.Resize(VIT_IMAGE_SIZE),
        transforms.CenterCrop(VIT_IMAGE_SIZE),
    ]

    if use_color_jitter:
        transform_steps.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.02,
            )
        )

    if use_blur:
        transform_steps.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=VIT_NORMALIZE_MEAN, std=VIT_NORMALIZE_STD),
        ]
    )

    return transforms.Compose(transform_steps)


def get_loaders(
    n=N_SAMPLES_PER_CLASS,
    batch_size=DEFAULT_BATCH_SIZE,
    train_split=0.8,
    use_color_jitter=False,
    use_blur=False,
):
    """
    Build train/test DataLoaders for the ViT experiment.

    `n` is the number of samples per class, not the total number of images.
    For example, `n=500` means:
    - 500 real images
    - 500 fake images
    - 1000 total images

    A stratified split is used so the train/test sets preserve class balance.
    """

    paths, labels = get_sample_paths(n=n)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths,
        labels,
        train_size=train_split,
        random_state=DEFAULT_SEED,
        stratify=labels,
        shuffle=True,
    )

    train_transform = build_vit_transform(
        use_color_jitter=use_color_jitter,
        use_blur=use_blur,
    )
    train_dataset = PathLabelDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = PathLabelDataset(test_paths, test_labels, transform=vit_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_vit_model(num_classes=2):
    """
    Create the ViT model used in this project.

    The pretrained backbone is kept, but the classification head is replaced
    with a new linear layer for binary classification.
    """

    model = models.vit_b_16(weights=VIT_WEIGHTS)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate_vit_model(model, loader, device, return_dict=False):
    """
    Evaluate the ViT model on a validation/test loader.

    The function computes:
    - accuracy
    - F1
    - ROC-AUC
    - EER
    - confusion matrix
    - classification report

    When `return_dict=True`, the raw arrays are also returned so downstream
    code can make plots such as the ROC curve.
    """

    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)

        # ViT outputs raw logits for the two classes.
        logits = model(images)

        # Convert logits to class probabilities and hard predictions.
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_pred = np.array(all_preds)

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": get_f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_probs),
        "eer": get_eer_score(y_true, y_probs),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=4),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
    }

    if return_dict:
        return result

    print("\n" + "=" * 30)
    print("VIT MODEL PERFORMANCE")
    print("=" * 30)
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"F1 Score:  {result['f1']:.4f}")
    print(f"AUC Score: {result['auc']:.4f}")
    print(f"EER Score: {result['eer']:.4f}")
    print("\nClassification Report:")
    print(result["report"])


def get_trained_ViT_model(train_loader, device, epochs=DEFAULT_EPOCHS, val_loader=None, output_dir=None):
    """
    Train the ViT model and optionally track validation performance.

    If a validation loader is provided, the function:
    - evaluates after each epoch
    - keeps the best checkpoint based on validation AUC
    - stores a simple epoch-by-epoch history on the returned model

    This function is intentionally reusable so it can be imported by
    ensemble scripts as well as used in standalone runs.
    """

    model = build_vit_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    history = []
    best_state = None
    best_auc = float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Store the average training loss for this epoch.
        avg_train_loss = total_loss / len(train_loader)
        epoch_record = {"epoch": epoch, "train_loss": avg_train_loss}

        if val_loader is not None:
            # Evaluate after each epoch so we can monitor the model and keep
            # the best checkpoint instead of blindly taking the final epoch.
            val_result = evaluate_vit_model(model, val_loader, device, return_dict=True)
            epoch_record.update(
                {
                    "val_accuracy": val_result["accuracy"],
                    "val_f1": val_result["f1"],
                    "val_auc": val_result["auc"],
                    "val_eer": val_result["eer"],
                }
            )

            # Validation AUC is used as the checkpoint-selection metric because
            # it is threshold-independent and usually more informative than
            # raw accuracy for imbalanced or uncertain classification settings.
            if val_result["auc"] > best_auc:
                best_auc = val_result["auc"]
                best_state = copy.deepcopy(model.state_dict())
                if output_dir is not None:
                    torch.save(
                        {
                            "model_state_dict": best_state,
                            "best_val_auc": best_auc,
                            "epoch": epoch,
                        },
                        output_dir / "vit_best.pt",
                    )

            print(
                f"Epoch {epoch}/{epochs} | train_loss: {avg_train_loss:.4f} | "
                f"val_acc: {val_result['accuracy']:.4f} | val_f1: {val_result['f1']:.4f} | "
                f"val_auc: {val_result['auc']:.4f} | val_eer: {val_result['eer']:.4f}"
            )
        else:
            print(f"Epoch {epoch}/{epochs} | train_loss: {avg_train_loss:.4f}")

        history.append(epoch_record)

    # If validation-based checkpointing was active, restore the best weights
    # before returning the model.
    if best_state is not None:
        model.load_state_dict(best_state)

    model.training_history = history
    return model


def save_metrics(result, output_dir):
    """
    Save the final summary metrics to CSV.
    """

    output_path = output_dir / "summary_metrics.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["accuracy", "f1", "auc", "eer"])
        writer.writeheader()
        writer.writerow(
            {
                "accuracy": result["accuracy"],
                "f1": result["f1"],
                "auc": result["auc"],
                "eer": result["eer"],
            }
        )
    return output_path


def save_report(result, output_dir):
    """
    Save an evaluation report to a text file.
    """

    output_path = output_dir / "detailed_report.txt"
    text = (
        f"Accuracy: {result['accuracy']:.4f}\n"
        f"F1: {result['f1']:.4f}\n"
        f"AUC: {result['auc']:.4f}\n"
        f"EER: {result['eer']:.4f}\n"
        f"Confusion matrix:\n{result['confusion_matrix']}\n\n"
        f"Classification report:\n{result['report']}\n"
    )
    output_path.write_text(text)
    return output_path


def plot_training_curves(history, output_dir):
    """
    Plot the main training/validation curves collected during training.

    The left subplot shows training loss and validation AUC.
    The right subplot shows validation accuracy and validation F1.
    """

    if not history:
        return None

    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o", label="train_loss")
    if "val_auc" in history[0]:
        axes[0].plot(epochs, [item["val_auc"] for item in history], marker="o", label="val_auc")
    axes[0].set_title("Training Loss and Validation AUC")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "val_accuracy" in history[0]:
        axes[1].plot(epochs, [item["val_accuracy"] for item in history], marker="o", label="val_accuracy")
        axes[1].plot(epochs, [item["val_f1"] for item in history], marker="o", label="val_f1")
        axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_evaluation(result, output_dir):
    """
    Plot two final evaluation views:
    - confusion matrix
    - ROC curve
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm = result["confusion_matrix"]
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            axes[0].text(col, row, cm[row, col], ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    fpr, tpr, _ = roc_curve(result["y_true"], result["y_probs"])
    axes[1].plot(fpr, tpr, label=f"AUC = {result['auc']:.4f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    output_path = output_dir / "evaluation_plots.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args():
    """
    Parse CLI arguments for standalone ViT runs.
    """

    parser = ArgumentParser(description="Run a standalone Vision Transformer experiment.")
    parser.add_argument(
        "--n",
        type=int,
        default=N_SAMPLES_PER_CLASS,
        help="Number of samples per class. n=500 means 1000 total images.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--color-jitter",
        action="store_true",
        help="Apply light ColorJitter augmentation to training images only.",
    )
    parser.add_argument(
        "--gaussian-blur",
        action="store_true",
        help="Apply light GaussianBlur augmentation to training images only.",
    )
    return parser.parse_args()


def main():
    """
    Run the full standalone ViT experiment.

    This function:
    1. creates the output directory
    2. builds train/test loaders
    3. trains the model
    4. evaluates the best checkpoint
    5. saves metrics, report, and plots
    """

    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(
        n=args.n,
        batch_size=args.batch_size,
        use_color_jitter=args.color_jitter,
        use_blur=args.gaussian_blur,
    )
    model = get_trained_ViT_model(
        train_loader,
        device,
        epochs=args.epochs,
        val_loader=test_loader,
        output_dir=output_dir,
    )
    result = evaluate_vit_model(model, test_loader, device, return_dict=True)

    metrics_path = save_metrics(result, output_dir)
    report_path = save_report(result, output_dir)
    curves_path = plot_training_curves(model.training_history, output_dir)
    eval_plot_path = plot_evaluation(result, output_dir)

    print("\nFinal")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"F1: {result['f1']:.4f}")
    print(f"AUC: {result['auc']:.4f}")
    print(f"EER: {result['eer']:.4f}")
    print(f"Saved summary CSV to: {metrics_path}")
    print(f"Saved detailed report to: {report_path}")
    print(f"Saved training curves plot to: {curves_path}")
    print(f"Saved evaluation plots to: {eval_plot_path}")


if __name__ == "__main__":
    main()
