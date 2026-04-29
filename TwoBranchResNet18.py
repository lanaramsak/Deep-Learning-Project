# Two-branch ResNet18 model for binary image classification.
# One branch processes the original image, while the second branch
# processes a transformed version of the same image. Features from both
# branches are concatenated and passed to a final classifier.
# This allows the model to learn from two complementary views of
# the same input in an end-to-end fine-tuning setup.

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, build_transform


device = "cpu"


class TwoBranchPathLabelDataset(Dataset):
    """
    Dataset for a two-branch model.

    Each sample is read only once from disk, then converted into:
    1. the original image tensor
    2. a transformed version of the same image tensor
    3. the class label

    This keeps the dataset aligned with the rest of the repository:
    labels are still simple binary integers, and paths still come from
    the same image collection utilities used elsewhere in the project.
    """

    def __init__(
        self,
        paths,
        labels,
        image_size=224,
        second_view_type="blur",
        blur_radius=2.0,
        rotation_degrees=10.0,
    ):
        self.paths = paths
        self.labels = labels
        self.second_view_type = second_view_type
        self.blur_radius = blur_radius
        self.rotation_degrees = rotation_degrees

        # We reuse the repository's standard ImageNet-style preprocessing
        # so the new model stays compatible with pretrained ResNet weights.
        self.transform = build_transform(image_size=image_size)

    def _build_second_view(self, image):
        if self.second_view_type == "identity":
            return image.copy()

        if self.second_view_type == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        if self.second_view_type == "rotation":
            return image.rotate(
                self.rotation_degrees,
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=(0, 0, 0),
            )

        raise ValueError(f"Unsupported second_view_type: {self.second_view_type}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = Path(self.paths[idx])
        image = Image.open(image_path).convert("RGB")

        # Branch 1 receives the standard image.
        original_tensor = self.transform(image)

        # Branch 2 receives an alternate view of the same image.
        second_view_image = self._build_second_view(image)
        second_view_tensor = self.transform(second_view_image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return original_tensor, second_view_tensor, label


def create_two_branch_dataloaders(
    paths=None,
    labels=None,
    image_size=224,
    second_view_type="blur",
    blur_radius=2.0,
    rotation_degrees=10.0,
    batch_size=32,
    test_size=0.2,
    random_state=42,
    num_workers=0,
):
    """
    Split paths/labels into train and validation subsets and build DataLoaders.
    - returns image loaders instead of precomputed feature matrices
    """

    if paths is None or labels is None:
        paths = DEFAULT_PATHS_SMALL
        labels = DEFAULT_Y_SMALL

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
        shuffle=True,
    )

    train_dataset = TwoBranchPathLabelDataset(
        train_paths,
        train_labels,
        image_size=image_size,
        second_view_type=second_view_type,
        blur_radius=blur_radius,
        rotation_degrees=rotation_degrees,
    )
    val_dataset = TwoBranchPathLabelDataset(
        val_paths,
        val_labels,
        image_size=image_size,
        second_view_type=second_view_type,
        blur_radius=blur_radius,
        rotation_degrees=rotation_degrees,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


class TwoBranchResNet18(nn.Module):
    """
    Two-branch fine-tuned ResNet18 model.

    Branch A processes the original image.
    Branch B processes a blurred version of the same image.

    Both branches extract 512-dimensional features, which are concatenated
    and passed through a small classifier head.
    """

    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None

        # Two independent branches are used on purpose.
        # Even though both start from the same pretrained weights, they are
        # free to specialize differently during fine-tuning.
        self.branch_original = models.resnet18(weights=weights)
        self.branch_blur = models.resnet18(weights=weights)

        # Replacing the final fully connected layer with Identity turns
        # each ResNet into a feature extractor that outputs a 512-d vector.
        self.branch_original.fc = nn.Identity()
        self.branch_blur.fc = nn.Identity()

        feature_dim = 512
        fused_dim = feature_dim * 2

        # The fusion head learns how to combine both representations.
        # This is still an end-to-end CNN model, not a separate classic
        # ML classifier such as SVM or logistic regression.
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_original, x_blur):
        original_features = self.branch_original(x_original)
        blur_features = self.branch_blur(x_blur)

        fused_features = torch.cat([original_features, blur_features], dim=1)
        logits = self.classifier(fused_features)
        return logits


def freeze_backbones(model):
    """
    Freeze both ResNet backbones and train only the fusion classifier.

    This is the standard first transfer-learning phase:
    learn the new task-specific head before changing the pretrained
    convolutional feature extractors.
    """

    for param in model.branch_original.parameters():
        param.requires_grad = False

    for param in model.branch_blur.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_resnet_block(model):
    """
    Unfreeze only the deepest residual block in each branch.

    This is a conservative fine-tuning strategy.
    Instead of updating every convolutional layer immediately, we only let
    the highest-level features adapt to the deepfake detection task.
    """

    for name, param in model.branch_original.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    for name, param in model.branch_blur.named_parameters():
        if "layer4" in name:
            param.requires_grad = True


def build_optimizer(model, learning_rate):
    """
    Create an optimizer only for parameters that are currently trainable.

    This matters because after freezing/unfreezing, we do not want to pass
    frozen parameters to the optimizer.
    """

    return torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
    )


def train_one_epoch(model, loader, criterion, optimizer, device=device):
    """
    Train the model for one epoch and return average loss and accuracy.
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_original, x_blur, labels in loader:
        x_original = x_original.to(device)
        x_blur = x_blur.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(x_original, x_blur)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device=device):
    """
    Evaluate the model without gradient tracking.

    Besides loss and accuracy, this function also returns the raw predicted
    labels and ground-truth labels so they can later be used for metrics
    such as F1, ROC-AUC, confusion matrix, and so on.
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    for x_original, x_blur, labels in loader:
        x_original = x_original.to(device)
        x_blur = x_blur.to(device)
        labels = labels.to(device)

        logits = model(x_original, x_blur)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

    avg_loss = running_loss / total
    accuracy = correct / total
    y_pred = torch.cat(all_predictions)
    y_true = torch.cat(all_labels)
    return avg_loss, accuracy, y_true, y_pred


def fit_two_stage_model(
    model,
    train_loader,
    val_loader,
    phase1_epochs=3,
    phase2_epochs=5,
    phase1_lr=1e-3,
    phase2_lr=1e-4,
    device=device,
):
    """
    Train the model in two phases.

    Phase 1:
    Only the fusion classifier is trained.

    Phase 2:
    The last residual block in both backbones is unfrozen and fine-tuned
    with a smaller learning rate.

    The returned history is a simple list of dictionaries so it is easy to
    inspect in a notebook or convert into a pandas DataFrame later.
    """

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    history = []

    freeze_backbones(model)
    optimizer = build_optimizer(model, learning_rate=phase1_lr)

    for epoch in tqdm(range(phase1_epochs), desc="Phase 1"):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device=device)
        history.append(
            {
                "phase": "head_training",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    unfreeze_last_resnet_block(model)
    optimizer = build_optimizer(model, learning_rate=phase2_lr)

    for epoch in tqdm(range(phase2_epochs), desc="Phase 2"):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device=device)
        history.append(
            {
                "phase": "fine_tuning",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    return history


def build_default_two_branch_setup(
    image_size=224,
    second_view_type="blur",
    blur_radius=2.0,
    rotation_degrees=10.0,
    batch_size=32,
    test_size=0.2,
    random_state=42,
    pretrained=True,
):
    """
    Convenience helper for quick experiments with the repository's default sample.

    It returns:
    - model
    - train_loader
    - val_loader

    This keeps the interface close to the project's existing "default sample"
    convention while still avoiding heavy training at import time.
    """

    train_loader, val_loader = create_two_branch_dataloaders(
        paths=DEFAULT_PATHS_SMALL,
        labels=DEFAULT_Y_SMALL,
        image_size=image_size,
        second_view_type=second_view_type,
        blur_radius=blur_radius,
        rotation_degrees=rotation_degrees,
        batch_size=batch_size,
        test_size=test_size,
        random_state=random_state,
    )
    model = TwoBranchResNet18(pretrained=pretrained)
    return model, train_loader, val_loader
