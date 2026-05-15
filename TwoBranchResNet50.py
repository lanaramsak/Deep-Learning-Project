"""
Two-branch ResNet50 model for binary image classification.

The API mirrors `TwoBranchResNet18.py` so the model can be swapped into
existing ablation runners with minimal code changes.
"""

from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from import_data import DEFAULT_PATHS_SMALL, DEFAULT_Y_SMALL, build_transform


device = "cpu"


class TwoBranchPathLabelDataset(Dataset):
    """
    Dataset for a two-branch ResNet50 model.

    Each sample is converted into:
    1. the original image tensor
    2. a transformed second-view tensor
    3. the class label
    """

    def __init__(
        self,
        paths,
        labels,
        image_size=224,
        second_view_type="blur",
        blur_radius=2.0,
        rotation_degrees=10.0,
        jitter_brightness=0.2,
        jitter_contrast=0.2,
        jitter_saturation=0.15,
        jitter_hue=0.02,
    ):
        self.paths = paths
        self.labels = labels
        self.second_view_type = second_view_type
        self.blur_radius = blur_radius
        self.rotation_degrees = rotation_degrees
        self.transform = build_transform(image_size=image_size)
        self.color_jitter = transforms.ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
            hue=jitter_hue,
        )

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

        if self.second_view_type == "rotation_blur":
            rotated = image.rotate(
                self.rotation_degrees,
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=(0, 0, 0),
            )
            return rotated.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        if self.second_view_type == "color_jitter":
            return self.color_jitter(image.copy())

        if self.second_view_type == "rotation_color_jitter":
            rotated = image.rotate(
                self.rotation_degrees,
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=(0, 0, 0),
            )
            return self.color_jitter(rotated)

        raise ValueError(f"Unsupported second_view_type: {self.second_view_type}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = Path(self.paths[idx])
        image = Image.open(image_path).convert("RGB")

        original_tensor = self.transform(image)
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
    jitter_brightness=0.2,
    jitter_contrast=0.2,
    jitter_saturation=0.15,
    jitter_hue=0.02,
    batch_size=32,
    test_size=0.2,
    random_state=42,
    num_workers=0,
):
    """
    Split paths/labels into train and validation subsets and build DataLoaders.
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
        jitter_brightness=jitter_brightness,
        jitter_contrast=jitter_contrast,
        jitter_saturation=jitter_saturation,
        jitter_hue=jitter_hue,
    )
    val_dataset = TwoBranchPathLabelDataset(
        val_paths,
        val_labels,
        image_size=image_size,
        second_view_type=second_view_type,
        blur_radius=blur_radius,
        rotation_degrees=rotation_degrees,
        jitter_brightness=jitter_brightness,
        jitter_contrast=jitter_contrast,
        jitter_saturation=jitter_saturation,
        jitter_hue=jitter_hue,
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


class TwoBranchResNet50(nn.Module):
    """
    Two-branch fine-tuned ResNet50 model.

    Both branches extract 2048-dimensional features, which are concatenated
    and passed through a small classifier head.
    """

    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None

        self.branch_original = models.resnet50(weights=weights)
        self.branch_blur = models.resnet50(weights=weights)

        self.branch_original.fc = nn.Identity()
        self.branch_blur.fc = nn.Identity()

        feature_dim = 2048
        fused_dim = feature_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_original, x_blur):
        original_features = self.branch_original(x_original)
        blur_features = self.branch_blur(x_blur)

        fused_features = torch.cat([original_features, blur_features], dim=1)
        logits = self.classifier(fused_features)
        return logits


def freeze_backbones(model):
    for param in model.branch_original.parameters():
        param.requires_grad = False

    for param in model.branch_blur.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_resnet_block(model):
    for name, param in model.branch_original.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    for name, param in model.branch_blur.named_parameters():
        if "layer4" in name:
            param.requires_grad = True


def build_optimizer(model, learning_rate):
    return torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
    )


def train_one_epoch(model, loader, criterion, optimizer, device=device):
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
    jitter_brightness=0.2,
    jitter_contrast=0.2,
    jitter_saturation=0.15,
    jitter_hue=0.02,
    batch_size=32,
    test_size=0.2,
    random_state=42,
    pretrained=True,
):
    train_loader, val_loader = create_two_branch_dataloaders(
        paths=DEFAULT_PATHS_SMALL,
        labels=DEFAULT_Y_SMALL,
        image_size=image_size,
        second_view_type=second_view_type,
        blur_radius=blur_radius,
        rotation_degrees=rotation_degrees,
        jitter_brightness=jitter_brightness,
        jitter_contrast=jitter_contrast,
        jitter_saturation=jitter_saturation,
        jitter_hue=jitter_hue,
        batch_size=batch_size,
        test_size=test_size,
        random_state=random_state,
    )
    model = TwoBranchResNet50(pretrained=pretrained)
    return model, train_loader, val_loader
