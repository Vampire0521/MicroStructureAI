"""
=============================================================
  COLAB TRAINING SCRIPT — Steel Diagnostic Tool
=============================================================
Run this script in Google Colab to train both models.

BEFORE RUNNING: Upload your MT_Project.zip to Colab and
run the setup cells (see instructions below).

Usage in Colab:
  !python colab_train.py --dataset uhcs   # Train microstructure model
  !python colab_train.py --dataset neu    # Train surface defect model
  !python colab_train.py --dataset both   # Train both (default)
=============================================================
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from tqdm import tqdm


# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {
    "uhcs": {
        "processed_dir": os.path.join(BASE_DIR, "data", "processed"),
        "model_save_path": os.path.join(BASE_DIR, "models", "best_microstructure_model.pth"),
        "num_classes": 6,
        "epochs": 25,
        "batch_size": 16,
        "lr": 1e-4,
        "title": "Microstructure Classification (UHCS)",
    },
    "neu": {
        "processed_dir": os.path.join(BASE_DIR, "data", "processed_neu"),
        "model_save_path": os.path.join(BASE_DIR, "models", "best_surface_defect_model.pth"),
        "num_classes": 6,
        "epochs": 20,
        "batch_size": 32,
        "lr": 1e-4,
        "title": "Surface Defect Detection (NEU)",
    },
}

# ImageNet normalization
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


# ── Model ────────────────────────────────────────────────────────────────────

def create_model(num_classes, pretrained=True):
    """Create ResNet50 with custom classifier head."""
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # Freeze up to layer3
    freeze = True
    for name, param in model.named_parameters():
        if "layer3" in name:
            freeze = False
        if freeze:
            param.requires_grad = False

    # Replace FC head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ── Data Loaders ─────────────────────────────────────────────────────────────

def get_loaders(processed_dir, batch_size):
    """Create train/val/test data loaders."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE,
    ])

    train_ds = datasets.ImageFolder(
        os.path.join(processed_dir, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(
        os.path.join(processed_dir, "val"), transform=eval_transform)
    test_ds = datasets.ImageFolder(
        os.path.join(processed_dir, "test"), transform=eval_transform)

    num_workers = 2 if torch.cuda.is_available() else 0

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


def compute_class_weights(train_dir, classes, device):
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = []
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        count = len([f for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        counts.append(count)
    total = sum(counts)
    weights = [total / (len(counts) * c) for c in counts]
    print(f"  Class weights: {[f'{w:.2f}' for w in weights]}")
    return torch.FloatTensor(weights).to(device)


# ── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_model(config_name):
    """Full training pipeline for one dataset."""
    cfg = CONFIGS[config_name]
    print("\n" + "=" * 60)
    print(f"  {cfg['title']}")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print(f"\n📦 Loading data from: {cfg['processed_dir']}")
    train_loader, val_loader, test_loader, classes = get_loaders(
        cfg["processed_dir"], cfg["batch_size"])
    print(f"  Classes: {classes}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Class weights
    weights = compute_class_weights(
        os.path.join(cfg["processed_dir"], "train"), classes, device)

    # Model
    print(f"\n🧠 Creating ResNet50 model ({cfg['num_classes']} classes)...")
    model = create_model(cfg["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5)

    # Training
    print(f"\n🚀 Training for {cfg['epochs']} epochs...\n")
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    os.makedirs(os.path.dirname(cfg["model_save_path"]), exist_ok=True)

    start_time = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "epoch": epoch,
            }, cfg["model_save_path"])
            improved = " ★ SAVED"

        print(f"  Epoch {epoch:>2d}/{cfg['epochs']}  │  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  │  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}{improved}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training completed in {elapsed / 60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")

    # Test evaluation
    print(f"\n📊 Evaluating on test set...")
    checkpoint = torch.load(cfg["model_save_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=classes))

    # Save history
    history_path = cfg["model_save_path"].replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"📈 Training history saved to: {history_path}")
    print(f"💾 Model saved to: {cfg['model_save_path']}")

    return best_val_acc


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Steel Diagnostic Models")
    parser.add_argument("--dataset", choices=["uhcs", "neu", "both"],
                        default="both", help="Which dataset to train on")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEEL DIAGNOSTIC TOOL — Model Training")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    if args.dataset in ("uhcs", "both"):
        train_model("uhcs")

    if args.dataset in ("neu", "both"):
        train_model("neu")

    print("\n" + "=" * 60)
    print("  ✅ ALL DONE!")
    print("  Download these model files back to your laptop:")
    print(f"    → {CONFIGS['uhcs']['model_save_path']}")
    print(f"    → {CONFIGS['neu']['model_save_path']}")
    print("=" * 60)
