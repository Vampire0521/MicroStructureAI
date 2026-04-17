"""
Step 5: Training Loop
======================
Trains the ResNet50 model on preprocessed data.
Supports both UHCS (microstructure) and NEU (surface defect) datasets.

Usage:
    python step5_train.py                         # UHCS (default)
    python step5_train.py --dataset neu            # NEU surface defects
    python step5_train.py --epochs 50 --lr 1e-4    # Custom hyperparams

For Colab:
    Upload this file + step4_model.py + data/processed/ to Colab
    Runtime → Change runtime type → T4 GPU
    !python step5_train.py
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# Local imports
from step4_model import create_model, get_transforms


# -- Configuration ---------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DATASET_CONFIGS = {
    "uhcs": {
        "processed_dir": os.path.join(BASE_DIR, "data", "processed"),
        "model_name": "best_microstructure_model.pth",
        "title": "Microstructure Classifier",
    },
    "neu": {
        "processed_dir": os.path.join(BASE_DIR, "data", "processed_neu"),
        "model_name": "best_surface_defect_model.pth",
        "title": "Surface Defect Classifier",
    },
}

# Active config -- set at runtime
PROCESSED_DIR = DATASET_CONFIGS["uhcs"]["processed_dir"]
MODEL_NAME = DATASET_CONFIGS["uhcs"]["model_name"]

DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
EARLY_STOP_PATIENCE = 7


def get_dataloaders(batch_size=DEFAULT_BATCH_SIZE, num_workers=0):
    """Create train/val/test DataLoaders using ImageFolder."""

    train_transform = get_transforms(is_training=True)
    eval_transform = get_transforms(is_training=False)

    train_dataset = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "val"),
        transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "test"),
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"  Train:      {len(train_dataset):>5} images  ({len(train_loader)} batches)")
    print(f"  Validation: {len(val_dataset):>5} images  ({len(val_loader)} batches)")
    print(f"  Test:       {len(test_dataset):>5} images  ({len(test_loader)} batches)")
    print(f"  Classes:    {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset.classes


def get_class_weights(device):
    """Load class weights from preprocessing metadata."""
    meta_path = os.path.join(PROCESSED_DIR, "metadata.npz")
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        weights = torch.FloatTensor(meta["class_weights"]).to(device)
        print(f"  Class weights loaded: {weights.cpu().numpy().round(3)}")
        return weights
    else:
        print("  ⚠️ No class weights found, using uniform weights")
        return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100. * correct / total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_training_curves(history, save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_title("Loss Curves", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o",
             markersize=3, label="Train Acc")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-o",
             markersize=3, label="Val Acc")
    ax2.set_title("Accuracy Curves", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n📈 Training curves saved to: {save_path}")
    plt.show()


# ── Main Training Loop ──────────────────────────────────────────────────────

def main():
    global PROCESSED_DIR, MODEL_NAME

    parser = argparse.ArgumentParser(description="Train microstructure/defect classifier")
    parser.add_argument("--dataset", choices=["uhcs", "neu"], default="uhcs",
                        help="Dataset to train on: 'uhcs' (microstructure) or 'neu' (surface defects)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    args = parser.parse_args()

    # Apply dataset config
    cfg = DATASET_CONFIGS[args.dataset]
    PROCESSED_DIR = cfg["processed_dir"]
    MODEL_NAME = cfg["model_name"]

    print("=" * 62)
    print(f"  STEP 5: Training {cfg['title']}")
    print("=" * 62)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Data
    print(f"\n📦 Loading data from {PROCESSED_DIR}")
    num_workers = 2 if device.type == "cuda" else 0
    train_loader, val_loader, _, classes = get_dataloaders(
        batch_size=args.batch_size, num_workers=num_workers
    )

    # Model
    print(f"\n🏗️  Building model...")
    model = create_model(num_classes=len(classes), pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Loss & Optimizer
    class_weights = get_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Patience:   {args.patience}")
    print()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        print(f"  Epoch {epoch:>3}/{args.epochs}  |  "
              f"Train: {train_loss:.4f} / {train_acc * 100:.1f}%  |  "
              f"Val: {val_loss:.4f} / {val_acc * 100:.1f}%  |  "
              f"LR: {current_lr:.2e}  |  "
              f"{epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": classes,
                "history": history,
            }, best_model_path)
            print(f"          ↑ Best model saved! (Val Acc: {val_acc * 100:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                break

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 62)
    print("  TRAINING COMPLETE")
    print("=" * 62)
    print(f"  Total time:      {total_time / 60:.1f} minutes")
    print(f"  Best Val Acc:    {best_val_acc * 100:.1f}%")
    print(f"  Best Val Loss:   {best_val_loss:.4f}")
    print(f"  Model saved to:  {best_model_path}")

    # Plot training curves
    plot_training_curves(
        history,
        save_path=os.path.join(OUTPUT_DIR, "training_curves.png")
    )

    print("\n✅ Step 5 complete! Model trained and saved.")


if __name__ == "__main__":
    main()
