""" VISION TRANSFORMER TRAINING AND EVALUATION 
ViT is trained on the data. It replaces the head with a custom binary classifier and fine-tunes for a few epochs.
"""

import torch
import numpy as np
import torch
import torch.nn as nn
import random
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from evaluation_metrics import get_classification_report, get_f1_score, get_auc_score, get_eer_score

from import_data import get_sample_paths, PathLabelDataset

random.seed(42)
N_SAMPLES_PER_CLASS = 1000

# Initialize model and processor
model_name = "google/vit-base-patch16-224-in21k" # pretrained model
processor = ViTImageProcessor.from_pretrained(model_name)

# Transformation of every image to match ViT's expected input format (resize, normalize, etc.)
vit_transform = transforms.Compose([
    transforms.Resize((processor.size["height"], processor.size["width"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Get DataLoaders for training and testing: This function collects image paths and labels, creates a dataset, splits it into training and testing sets, and returns two separate DataLoaders.
def get_loaders(n=N_SAMPLES_PER_CLASS, batch_size=32, train_split=0.8):
    # Collect and filter all sample paths
    paths, labels = get_sample_paths(n=n)
    
    # Create the full dataset
    full_dataset = PathLabelDataset(paths, labels, transform=vit_transform)

    # Calculate lengths for split
    train_len = int(len(full_dataset) * train_split)
    test_len = len(full_dataset) - train_len

    # Perform the random split
    train_ds, test_ds = random_split(
        full_dataset, 
        [train_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    # Create two separate loaders: to wrap a dataset and to handle the mechanics of feeding it to the model during training/evaluation
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# train_loader, test_loader = get_loaders()

# TRAINING THE VIT MODEL
# Loads the pretrained ViT and replaces its classification head with a 2-class output (Real/Fake). 
# Then fine-tunes the whole model for 5 epochs.
def get_trained_ViT_model(train_loader, device):
    # Define the model with 2 classes: 0=Real, 1=Fake
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Real", 1: "Fake"},
        label2id={"Real": 0, "Fake": 1}
    )

    model.to(device) # Move model to GPU if available

    # Training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Trainning loop for one epoch
    def train_one_epoch(model, loader, optimizer):
        model.train()
        total_loss = 0
        for batch in loader:
            # 'loader' provides (images, labels) from your PathLabelDataset
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # Clear gradients before backpropagation
            
            # Forward pass
            # ViT expects 'pixel_values'
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            
            # Backward pass and optimization step (uses the gradients to update the model's parameters)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader) # Return average loss for the epoch

    # Run for a few epochs
    for epoch in range(5):
        avg_loss = train_one_epoch(model, train_loader, optimizer)    
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = get_trained_ViT_model(train_loader, device)

# EVALUATION ON TEST SET
def evaluate_vit_model(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            images = images.to(device)
            
            # Get model output
            outputs = model(pixel_values=images)
            logits = outputs.logits
            
            # Get probabilities (for AUC and EER)
            # We apply softmax to the logits and take the 'Fake' class (index 1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            # Get hard predictions (for F1 and Classification Report)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert to numpy for your custom functions
    y_test = np.array(all_labels)
    y_probs = np.array(all_probs)
    final_preds = np.array(all_preds)

    print("\n" + "="*30)
    print("VIT MODEL PERFORMANCE")
    print("="*30)
    
    print(f"F1 Score: {get_f1_score(y_test, final_preds):.4f}")
    print(f"AUC Score: {get_auc_score(y_test, y_probs):.4f}")
    print(f"EER Score: {get_eer_score(y_test, y_probs):.4f}")
    print("\nClassification Report:")
    print(get_classification_report(y_test, final_preds))

# evaluate_vit_model(model, test_loader, device)