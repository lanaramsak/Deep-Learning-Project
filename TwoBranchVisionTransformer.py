"""
Two-branch Vision Transformer model for binary image classification.
One branch processes the original image, while the second branch processes a transformed version of the same image. 
Features from both branches are concatenated and passed to a final classifier.
This allows the model to learn from two complementary views of
the same input in an end-to-end fine-tuning setup.
"""

from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from import_data import DEFAULT_PATHS_SMALL, DEFAULT_SEED, DEFAULT_Y_SMALL
from VisionTransformer import vit_transform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoBranchVisionDataset(Dataset):
    """
    Dataset for a two-branch Vision Transformer model.

    Each sample is loaded once and converted into:
    1. the original image tensor
    2. a transformed second-view tensor
    3. the class label

    The second view can be identity, blur, rotation, or a rotation+blur
    composition, matching the experiments already used with the two-branch
    ResNet setup.
    """

    def __init__(
        self,
        paths,
        labels,
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
        self.transform = vit_transform
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


def create_two_branch_vit_dataloaders(
    paths=None,
    labels=None,
    second_view_type="blur",
    blur_radius=2.0,
    rotation_degrees=10.0,
    jitter_brightness=0.2,
    jitter_contrast=0.2,
    jitter_saturation=0.15,
    jitter_hue=0.02,
    batch_size=16,
    test_size=0.2,
    random_state=DEFAULT_SEED,
    num_workers=0,
):
    """
    Build train/validation loaders for a two-branch ViT setup.
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

    train_dataset = TwoBranchVisionDataset(
        train_paths,
        train_labels,
        second_view_type=second_view_type,
        blur_radius=blur_radius,
        rotation_degrees=rotation_degrees,
        jitter_brightness=jitter_brightness,
        jitter_contrast=jitter_contrast,
        jitter_saturation=jitter_saturation,
        jitter_hue=jitter_hue,
    )
    val_dataset = TwoBranchVisionDataset(
        val_paths,
        val_labels,
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
    batch_size=16,
    test_size=0.2,
    random_state=DEFAULT_SEED,
    num_workers=0,
):
    """
    Backward-compatible wrapper matching the ResNet two-branch API.

    The `image_size` argument is accepted to keep the same call signature as
    `TwoBranchResNet18.create_two_branch_dataloaders`, even though the ViT
    pipeline uses `vit_transform` from the pretrained weights definition.
    """

    return create_two_branch_vit_dataloaders(
        paths=paths,
        labels=labels,
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
        num_workers=num_workers,
    )


class TwoBranchVisionTransformer(nn.Module):
    """
    Two-branch Vision Transformer model.

    Branch A processes the original image.
    Branch B processes a transformed version of the same image.

    Both branches produce feature vectors, which are concatenated and passed
    through a small classifier head.
    """

    def __init__(self, num_classes=2, pretrained=True, dropout=0.3):
        super().__init__()

        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None

        self.branch_original = models.vit_b_16(weights=weights)
        self.branch_second = models.vit_b_16(weights=weights)

        hidden_dim = self.branch_original.hidden_dim

        self.branch_original.heads = nn.Identity()
        self.branch_second.heads = nn.Identity()

        fused_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_original, x_second):
        original_features = self.branch_original(x_original)
        second_features = self.branch_second(x_second)
        fused_features = torch.cat([original_features, second_features], dim=1)
        logits = self.classifier(fused_features)
        return logits


def freeze_backbones(model):
    """
    Freeze both ViT backbones and train only the fusion classifier.
    """

    for param in model.branch_original.parameters():
        param.requires_grad = False

    for param in model.branch_second.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_vit_block(model):
    """
    Unfreeze only the final encoder block in each ViT branch.

    This mirrors the conservative fine-tuning strategy used in the two-branch
    ResNet model: adapt only the highest-level features first.
    """

    for name, param in model.branch_original.named_parameters():
        if "encoder.layers.encoder_layer_11" in name or "encoder.ln" in name:
            param.requires_grad = True

    for name, param in model.branch_second.named_parameters():
        if "encoder.layers.encoder_layer_11" in name or "encoder.ln" in name:
            param.requires_grad = True


def build_optimizer(model, learning_rate):
    """
    Find parameters that require gradients and build an AdamW optimizer for them.
    """

    return torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )


def train_one_epoch(model, loader, criterion, optimizer, device=device):
    """
    Train the two-branch ViT model for one epoch.
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_original, x_second, labels in loader:
        x_original = x_original.to(device)
        x_second = x_second.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(x_original, x_second)
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
    Evaluate the two-branch ViT model.
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    for x_original, x_second, labels in loader:
        x_original = x_original.to(device)
        x_second = x_second.to(device)
        labels = labels.to(device)

        logits = model(x_original, x_second)
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
    phase1_lr=1e-4,
    phase2_lr=1e-5,
    device=device,
):
    """
    Train the two-branch ViT model in two phases.

    Phase 1:
    Only the fusion classifier is trained.

    Phase 2:
    The final encoder block in both ViT branches is unfrozen and fine-tuned
    with a smaller learning rate.
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

    unfreeze_last_vit_block(model)
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


def build_default_two_branch_vit_setup(
    second_view_type="blur",
    blur_radius=2.0,
    rotation_degrees=10.0,
    jitter_brightness=0.2,
    jitter_contrast=0.2,
    jitter_saturation=0.15,
    jitter_hue=0.02,
    batch_size=16,
    test_size=0.2,
    random_state=DEFAULT_SEED,
    pretrained=True,
):
    """
    Convenience helper for quick two-branch ViT experiments.
    """

    train_loader, val_loader = create_two_branch_vit_dataloaders(
        paths=DEFAULT_PATHS_SMALL,
        labels=DEFAULT_Y_SMALL,
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
    model = TwoBranchVisionTransformer(pretrained=pretrained)
    return model, train_loader, val_loader
