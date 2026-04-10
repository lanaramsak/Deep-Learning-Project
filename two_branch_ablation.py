# FILE TO TEST IF BLURRING THE SECOND BRANCH IMPROVES PERFORMANCE COMPARED TO ORIGINAL-ONLY CONTROL

from argparse import ArgumentParser

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from evaluation_metrics import get_eer_score, get_f1_score
from TwoBranchResNet18 import (
    TwoBranchResNet18,
    create_two_branch_dataloaders,
    fit_two_stage_model,
)
from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL


device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate_with_probabilities(model, loader, device=device):
    """
    Evaluate a trained two-branch model and return labels, predictions, and scores.
        - Returns raw probabilities for the positive class to allow AUC and EER calculation.
    """

    model.eval()
    all_true = []
    all_pred = []
    all_score = []

    for x_original, x_blur, labels in loader:
        x_original = x_original.to(device)
        x_blur = x_blur.to(device)

        logits = model(x_original, x_blur)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)

        all_true.append(labels.cpu())
        all_pred.append(predictions.cpu())
        all_score.append(probabilities.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_score = torch.cat(all_score).numpy()
    return y_true, y_pred, y_score


def run_single_experiment(blur_radius):
    """
    Train and evaluate one configuration.

    blur_radius = 0.0 acts as an "original only" control setting:
    both branches see the same unblurred image.
    """

    train_loader, val_loader = create_two_branch_dataloaders(
        paths=DEFAULT_PATHS_SMALL,
        labels=DEFAULT_Y_SMALL,
        image_size=224,
        blur_radius=blur_radius,
        batch_size=32,
        test_size=0.2,
        random_state=42,
        num_workers=0,
    )

    model = TwoBranchResNet18(
        num_classes=2,
        pretrained=False,  # Ablation should be done without pretrained weights to isolate the effect of blur
        dropout=0.3,
    )

    history = fit_two_stage_model(
        model,
        train_loader,
        val_loader,
        phase1_epochs=3,
        phase2_epochs=5,
        phase1_lr=1e-3,
        phase2_lr=1e-4,
        device=device,
    )

    y_true, y_pred, y_score = evaluate_with_probabilities(model, val_loader, device=device)

    return {
        "blur_radius": blur_radius,
        "history": history,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": get_f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "eer": get_eer_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=4),
    }


def format_result(result):
    label = "original_only" if result["blur_radius"] == 0 else f"blur_radius_{result['blur_radius']}"

    lines = [
        f"Experiment: {label}",
        f"Accuracy: {result['accuracy']:.4f}",
        f"F1: {result['f1']:.4f}",
        f"AUC: {result['auc']:.4f}",
        f"EER: {result['eer']:.4f}",
        "Confusion matrix:",
        str(result["confusion_matrix"]),
        "Classification report:",
        result["report"],
    ]
    return "\n".join(lines)


def main():

    blur_radii = [0.0, 1.0, 2.0, 3.0]
    results = []

    print(f"Running on device: {device}")
    print(f"Dataset size: {len(DEFAULT_PATHS_SMALL)} images")

    for blur_radius in blur_radii:
        label = "original_only" if blur_radius == 0 else f"blur_radius_{blur_radius}"
        print(f"\nRunning experiment: {label}")
        result = run_single_experiment(blur_radius)
        results.append(result)
        print(format_result(result))

    print("\nSummary")
    print(f"{'Experiment':<20} {'Acc':>8} {'F1':>8} {'AUC':>8} {'EER':>8}")
    print("-" * 56)
    for result in sorted(results, key=lambda item: item["auc"], reverse=True):
        label = "original_only" if result["blur_radius"] == 0 else f"blur_{result['blur_radius']}"
        print(
            f"{label:<20} "
            f"{result['accuracy']:>8.4f} "
            f"{result['f1']:>8.4f} "
            f"{result['auc']:>8.4f} "
            f"{result['eer']:>8.4f}"
        )


if __name__ == "__main__":
    main()


#results
# Summary
# Experiment                Acc       F1      AUC      EER
# --------------------------------------------------------
# blur_3.0               0.7550   0.7101   0.8321   0.2700
# blur_2.0               0.7250   0.7291   0.7877   0.2900
# blur_1.0               0.6550   0.6567   0.6850   0.3500
# original_only          0.6400   0.6505   0.6740   0.3700