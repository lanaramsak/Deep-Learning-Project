import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from evaluation_metrics import get_classification_report, get_f1_score, get_auc_score, get_eer_score
import random
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from import_data import get_sample_paths, PathLabelDataset, build_transform

random.seed(42)
N_SAMPLES_PER_CLASS = 500

def get_loaders(n=N_SAMPLES_PER_CLASS, batch_size=32, train_split=0.8):
    # 1. Collect and filter all sample paths
    paths, labels = get_sample_paths(n=n)
    
    # 2. Create the full dataset
    full_dataset = PathLabelDataset(paths, labels, transform=build_transform(224))

    # 3. Calculate lengths for split
    train_len = int(len(full_dataset) * train_split)
    test_len = len(full_dataset) - train_len

    # 4. Perform the random split
    train_ds, test_ds = random_split(
        full_dataset, 
        [train_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    # 5. Create two separate loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# train_loader, test_loader = get_loaders()
device = "cuda" if torch.cuda.is_available() else "cpu"

# RESNET-50 MODEL TRAINING
def get_trained_ResNet50_model(train_loader, device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if hasattr(model.fc, 'in_features'):
        num_ftrs = model.fc.in_features
    else:
        # If we already replaced it, we know ResNet-50 uses 2048
        num_ftrs = 2048

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1) # Single output for binary classification
    )

    # Freeze all "backbone" layers (initially)
    # This prevents the gradients from changing the pre-trained weights
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    # Use BCEWithLogitsLoss because we have 1 output node
    criterion = nn.BCEWithLogitsLoss()

    # Only optimize the parameters that are NOT frozen (the new fc head)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    def train_fine_tune(model, loader, optimizer, criterion, epochs=5):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device).float().view(-1, 1)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

    # Phase 1: Train just the head
    train_fine_tune(model, train_loader, optimizer, criterion)

    # Unfreeze the last block (layer4) and the head
    for name, child in model.named_children():
        if name in ['layer4', 'fc']:
            print(f"Unfreezing {name}")
            for param in child.parameters():
                param.requires_grad = True

    # Use a MUCH smaller learning rate for fine-tuning the backbone
    # You don't want to "break" the pre-trained weights, just nudge them.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Phase 2: Fine-tune the deeper layers + the head
    train_fine_tune(model, train_loader, optimizer, criterion, epochs=3)

    return model

# model = get_trained_ResNet50_model(train_loader, device)

def evaluate_resnet50_model(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    print("Evaluating ResNet50...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            # Forward pass - ResNet returns the tensor directly
            logits = model(images) 
            
            # Use Sigmoid for binary (1-node) classification
            probs = torch.sigmoid(logits).squeeze()
            
            # Threshold at 0.5 for hard predictions
            preds = (probs > 0.5).float()

            # Handle single-item batch case (squeezing might remove batch dim)
            if probs.dim() == 0:
                all_probs.append(probs.item())
                all_preds.append(preds.item())
            else:
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
            all_labels.extend(labels.numpy())

    y_test = np.array(all_labels)
    y_probs = np.array(all_probs)
    final_preds = np.array(all_preds)

    print("\n" + "="*30 + "\nRESNET50 PERFORMANCE\n" + "="*30)
    print(f"F1 Score:  {get_f1_score(y_test, final_preds):.4f}")
    print(f"AUC Score: {get_auc_score(y_test, y_probs):.4f}")
    print(f"EER Score: {get_eer_score(y_test, y_probs):.4f}")
    print("\nClassification Report:\n", get_classification_report(y_test, final_preds))

# evaluate_resnet50_model(model, test_loader, device)