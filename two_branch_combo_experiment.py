from argparse import ArgumentParser
import copy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from evaluation_metrics import get_eer_score, get_f1_score
from TwoBranchResNet18 import (
    TwoBranchResNet18,
    build_optimizer,
    create_two_branch_dataloaders,
    freeze_backbones,
    train_one_epoch,
    unfreeze_last_resnet_block,
)
from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL


device = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "two_branch_combo"


def make_experiment(label, blur_radius, rotation_degrees):
    return {
        "label": label,
        "second_view_type": "rotation_blur",
        "blur_radius": blur_radius,
        "rotation_degrees": rotation_degrees,
    }


def get_experiments():
    return [
        make_experiment("rotation_10_blur_2.0", blur_radius=2.0, rotation_degrees=10.0),
        make_experiment("rotation_15_blur_2.0", blur_radius=2.0, rotation_degrees=15.0),
    ]


@torch.no_grad()
def evaluate_with_probabilities(model, loader, criterion, device=device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_true = []
    all_pred = []
    all_score = []

    for x_original, x_second_view, labels in loader:
        x_original = x_original.to(device)
        x_second_view = x_second_view.to(device)
        labels = labels.to(device)

        logits = model(x_original, x_second_view)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)

        running_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_true.append(labels.cpu())
        all_pred.append(predictions.cpu())
        all_score.append(probabilities.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_score = torch.cat(all_score).numpy()

    return {
        "val_loss": running_loss / total,
        "val_acc": correct / total,
        "val_f1": get_f1_score(y_true, y_pred),
        "val_auc": roc_auc_score(y_true, y_score),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def is_better(candidate, best, metric_name):
    if best is None:
        return True
    return candidate[metric_name] > best[metric_name]


def train_with_best_checkpoint(
    model,
    train_loader,
    val_loader,
    output_dir,
    experiment_label,
    best_metric="val_auc",
    phase1_epochs=3,
    phase2_epochs=5,
    phase1_lr=1e-3,
    phase2_lr=1e-4,
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_state = None
    best_metrics = None
    best_checkpoint_path = output_dir / f"{experiment_label}_best.pt"

    def run_epoch_loop(epochs, phase_name, optimizer):
        nonlocal best_state, best_metrics

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device=device
            )
            val_metrics = evaluate_with_probabilities(model, val_loader, criterion, device=device)

            record = {
                "phase": phase_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["val_acc"],
                "val_f1": val_metrics["val_f1"],
                "val_auc": val_metrics["val_auc"],
            }
            history.append(record)

            if is_better(record, best_metrics, best_metric):
                best_metrics = copy.deepcopy(record)
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "experiment_label": experiment_label,
                        "best_metric": best_metric,
                        "best_metrics": best_metrics,
                        "model_state_dict": best_state,
                    },
                    best_checkpoint_path,
                )

            print(
                f"[{experiment_label}] {phase_name} epoch {epoch}/{epochs} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
                f"val_loss: {val_metrics['val_loss']:.4f} | val_acc: {val_metrics['val_acc']:.4f} | "
                f"val_f1: {val_metrics['val_f1']:.4f} | val_auc: {val_metrics['val_auc']:.4f}"
            )

    freeze_backbones(model)
    optimizer = build_optimizer(model, learning_rate=phase1_lr)
    run_epoch_loop(phase1_epochs, "head_training", optimizer)

    unfreeze_last_resnet_block(model)
    optimizer = build_optimizer(model, learning_rate=phase2_lr)
    run_epoch_loop(phase2_epochs, "fine_tuning", optimizer)

    model.load_state_dict(best_state)
    final_metrics = evaluate_with_probabilities(model, val_loader, criterion, device=device)

    return history, best_metrics, final_metrics, best_checkpoint_path


def run_single_experiment(experiment, output_dir, best_metric):
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
        pretrained=False,
        dropout=0.3,
    )

    history, best_epoch_metrics, final_metrics, checkpoint_path = train_with_best_checkpoint(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        experiment_label=experiment["label"],
        best_metric=best_metric,
    )

    return {
        "label": experiment["label"],
        "second_view_type": experiment["second_view_type"],
        "blur_radius": experiment["blur_radius"],
        "rotation_degrees": experiment["rotation_degrees"],
        "history": history,
        "best_epoch_metrics": best_epoch_metrics,
        "best_checkpoint_path": str(checkpoint_path),
        "accuracy": accuracy_score(final_metrics["y_true"], final_metrics["y_pred"]),
        "f1": get_f1_score(final_metrics["y_true"], final_metrics["y_pred"]),
        "auc": roc_auc_score(final_metrics["y_true"], final_metrics["y_score"]),
        "eer": get_eer_score(final_metrics["y_true"], final_metrics["y_score"]),
        "confusion_matrix": confusion_matrix(final_metrics["y_true"], final_metrics["y_pred"]),
        "report": classification_report(final_metrics["y_true"], final_metrics["y_pred"], digits=4),
    }


def format_result(result):
    best = result["best_epoch_metrics"]
    lines = [
        f"Experiment: {result['label']}",
        f"Second view: {result['second_view_type']}",
        f"Blur radius: {result['blur_radius']}",
        f"Rotation degrees: {result['rotation_degrees']}",
        f"Best checkpoint: {result['best_checkpoint_path']}",
        f"Best epoch phase: {best['phase']}",
        f"Best epoch number: {best['epoch']}",
        f"Best val_acc: {best['val_acc']:.4f}",
        f"Best val_f1: {best['val_f1']:.4f}",
        f"Best val_auc: {best['val_auc']:.4f}",
        f"Final accuracy: {result['accuracy']:.4f}",
        f"Final F1: {result['f1']:.4f}",
        f"Final AUC: {result['auc']:.4f}",
        f"Final EER: {result['eer']:.4f}",
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
        "best_phase",
        "best_epoch",
        "best_val_acc",
        "best_val_f1",
        "best_val_auc",
        "best_checkpoint_path",
    ]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            best = result["best_epoch_metrics"]
            writer.writerow(
                {
                    "label": result["label"],
                    "second_view_type": result["second_view_type"],
                    "blur_radius": result["blur_radius"],
                    "rotation_degrees": result["rotation_degrees"],
                    "accuracy": result["accuracy"],
                    "f1": result["f1"],
                    "auc": result["auc"],
                    "eer": result["eer"],
                    "best_phase": best["phase"],
                    "best_epoch": best["epoch"],
                    "best_val_acc": best["val_acc"],
                    "best_val_f1": best["val_f1"],
                    "best_val_auc": best["val_auc"],
                    "best_checkpoint_path": result["best_checkpoint_path"],
                }
            )

    return output_path


def save_detailed_report(results, output_dir):
    output_path = output_dir / "detailed_report.txt"
    sections = []
    for result in results:
        sections.append(format_result(result))
        sections.append("\n" + "=" * 72 + "\n")

    output_path.write_text("\n".join(sections).rstrip() + "\n")
    return output_path


def plot_training_curves(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metric_specs = [
        ("val_loss", "Validation Loss"),
        ("val_f1", "Validation F1"),
        ("val_auc", "Validation ROC-AUC"),
    ]

    for result in results:
        epochs = list(range(1, len(result["history"]) + 1))
        for axis, (metric_key, title) in zip(axes, metric_specs):
            values = [epoch_result[metric_key] for epoch_result in result["history"]]
            axis.plot(epochs, values, marker="o", label=result["label"])
            axis.set_title(title)
            axis.set_xlabel("Global Epoch")
            axis.set_ylim(0, 1 if metric_key != "val_loss" else max(values) * 1.15)

    axes[0].set_ylabel("Metric")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
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
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(0, 1)

    fig.suptitle("Two-Branch Rotation+Blur Experiment Metrics")
    fig.tight_layout()
    output_path = output_dir / "metric_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args():
    parser = ArgumentParser(description="Run two-branch combined rotation+blur experiments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where reports, plots, and checkpoints are saved.",
    )
    parser.add_argument(
        "--best-metric",
        choices=["val_auc", "val_f1"],
        default="val_auc",
        help="Metric used to decide which checkpoint to keep.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = get_experiments()
    results = []

    print(f"Running on device: {device}")
    print(f"Dataset size: {len(DEFAULT_PATHS_SMALL)} images")
    print(f"Best checkpoint metric: {args.best_metric}")

    for experiment in experiments:
        print(f"\nRunning experiment: {experiment['label']}")
        result = run_single_experiment(experiment, output_dir, args.best_metric)
        results.append(result)
        print(format_result(result))

    print("\nSummary")
    print(f"{'Experiment':<24} {'Acc':>8} {'F1':>8} {'AUC':>8} {'EER':>8}")
    print("-" * 64)
    for result in sorted(results, key=lambda item: item["auc"], reverse=True):
        print(
            f"{result['label']:<24} "
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
