# FILE TO TEST WHETHER DIFFERENT SECOND-BRANCH VIEWS IMPROVE PERFORMANCE

from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from evaluation_metrics import get_eer_score, get_f1_score
from TwoBranchResNet18 import (
    TwoBranchResNet18,
    create_two_branch_dataloaders,
    fit_two_stage_model,
)
from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL


device = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "two_branch_ablation"


@torch.no_grad()
def evaluate_with_probabilities(model, loader, device=device):
    """
    Evaluate a trained two-branch model and return labels, predictions, and scores.
    Returns raw probabilities for the positive class to allow AUC and EER calculation.
    """

    model.eval()
    all_true = []
    all_pred = []
    all_score = []

    for x_original, x_second_view, labels in loader:
        x_original = x_original.to(device)
        x_second_view = x_second_view.to(device)

        logits = model(x_original, x_second_view)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)

        all_true.append(labels.cpu())
        all_pred.append(predictions.cpu())
        all_score.append(probabilities.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_score = torch.cat(all_score).numpy()
    return y_true, y_pred, y_score


def make_experiment(label, second_view_type, blur_radius=0.0, rotation_degrees=0.0):
    return {
        "label": label,
        "second_view_type": second_view_type,
        "blur_radius": blur_radius,
        "rotation_degrees": rotation_degrees,
    }


def get_experiments(include_rotation=True):
    experiments = [
        make_experiment("original_only", "identity"),
        make_experiment("blur_1.0", "blur", blur_radius=1.0),
        make_experiment("blur_2.0", "blur", blur_radius=2.0),
        make_experiment("blur_3.0", "blur", blur_radius=3.0),
    ]

    if include_rotation:
        experiments.extend(
            [
                make_experiment("rotation_5", "rotation", rotation_degrees=5.0),
                make_experiment("rotation_10", "rotation", rotation_degrees=10.0),
                make_experiment("rotation_15", "rotation", rotation_degrees=15.0),
            ]
        )

    return experiments


def run_single_experiment(experiment):
    train_loader, val_loader = create_two_branch_dataloaders(
        paths=DEFAULT_PATHS_SMALL,
        labels=DEFAULT_Y_SMALL,
        image_size=224,
        second_view_type=experiment["second_view_type"],
        blur_radius=experiment["blur_radius"],
        rotation_degrees=experiment["rotation_degrees"],
        batch_size=32,
        test_size=0.2,
        random_state=42,
        num_workers=0,
    )

    model = TwoBranchResNet18(
        num_classes=2,
        pretrained=False,  # Keep ablations comparable and focused on the second-view effect.
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
        "label": experiment["label"],
        "second_view_type": experiment["second_view_type"],
        "blur_radius": experiment["blur_radius"],
        "rotation_degrees": experiment["rotation_degrees"],
        "history": history,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": get_f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "eer": get_eer_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=4),
    }


def format_result(result):
    lines = [
        f"Experiment: {result['label']}",
        f"Second view: {result['second_view_type']}",
        f"Blur radius: {result['blur_radius']}",
        f"Rotation degrees: {result['rotation_degrees']}",
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


def save_metrics_csv(results, output_dir):
    output_path = output_dir / "summary_metrics.csv"
    fieldnames = [
        "label",
        "second_view_type",
        "blur_radius",
        "rotation_degrees",
        "accuracy",
        "f1",
        "auc",
        "eer",
    ]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result[name] for name in fieldnames})

    return output_path


def save_detailed_report(results, output_dir):
    output_path = output_dir / "detailed_report.txt"
    sections = []
    for result in results:
        sections.append(format_result(result))
        sections.append("\n" + "=" * 72 + "\n")

    output_path.write_text("\n".join(sections).rstrip() + "\n")
    return output_path


def plot_metric_bars(results, output_dir):
    labels = [result["label"] for result in results]
    metrics = [
        ("accuracy", "Accuracy"),
        ("f1", "F1 Score"),
        ("auc", "ROC-AUC"),
        ("eer", "EER"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (metric_key, metric_label) in zip(axes.flat, metrics):
        values = [result[metric_key] for result in results]
        axis.bar(labels, values, color="#4472C4")
        axis.set_title(metric_label)
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylim(0, 1)

    fig.suptitle("Two-Branch View Ablation Metrics")
    fig.tight_layout()
    output_path = output_dir / "metric_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training_curves(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for result in results:
        epochs = list(range(1, len(result["history"]) + 1))
        val_loss = [epoch_result["val_loss"] for epoch_result in result["history"]]
        val_acc = [epoch_result["val_acc"] for epoch_result in result["history"]]

        axes[0].plot(epochs, val_loss, marker="o", label=result["label"])
        axes[1].plot(epochs, val_acc, marker="o", label=result["label"])

    axes[0].set_title("Validation Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].set_title("Validation Accuracy by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)

    axes[1].legend(loc="lower right")
    fig.tight_layout()
    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args():
    parser = ArgumentParser(description="Run two-branch second-view ablations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV summaries, text reports, and plots are saved.",
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Skip rotation experiments and only compare blur against the original-only control.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = get_experiments(include_rotation=not args.no_rotation)
    results = []

    print(f"Running on device: {device}")
    print(f"Dataset size: {len(DEFAULT_PATHS_SMALL)} images")

    for experiment in experiments:
        print(f"\nRunning experiment: {experiment['label']}")
        result = run_single_experiment(experiment)
        results.append(result)
        print(format_result(result))

    print("\nSummary")
    print(f"{'Experiment':<20} {'Acc':>8} {'F1':>8} {'AUC':>8} {'EER':>8}")
    print("-" * 56)
    for result in sorted(results, key=lambda item: item["auc"], reverse=True):
        print(
            f"{result['label']:<20} "
            f"{result['accuracy']:>8.4f} "
            f"{result['f1']:>8.4f} "
            f"{result['auc']:>8.4f} "
            f"{result['eer']:>8.4f}"
        )

    metrics_path = save_metrics_csv(results, output_dir)
    report_path = save_detailed_report(results, output_dir)
    bars_path = plot_metric_bars(results, output_dir)
    curves_path = plot_training_curves(results, output_dir)

    print()
    print(f"Saved summary CSV to: {metrics_path}")
    print(f"Saved detailed report to: {report_path}")
    print(f"Saved metric comparison plot to: {bars_path}")
    print(f"Saved training curves plot to: {curves_path}")


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