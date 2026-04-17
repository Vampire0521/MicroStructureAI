"""
Step 6: Model Evaluation
=========================
Loads the trained model and generates detailed evaluation metrics:
- Confusion matrix
- Per-class precision/recall/F1
- Most confident correct/incorrect predictions

Usage:
    python step6_evaluate.py                  # UHCS (default)
    python step6_evaluate.py --dataset neu    # NEU surface defects
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from PIL import Image

from step4_model import create_model, get_transforms


# -- Configuration ---------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DATASET_CONFIGS = {
    "uhcs": {
        "processed_dir": os.path.join(BASE_DIR, "data", "processed"),
        "model_path": os.path.join(BASE_DIR, "models", "best_microstructure_model.pth"),
        "title": "Microstructure Classification",
        "display_names": {
            "network": "Network Carbides",
            "pearlite": "Pearlite",
            "pearlite+spheroidite": "Pearlite+Spheroidite",
            "pearlite+widmanstatten": "Pearl.+Widm.",
            "spheroidite": "Spheroidite",
            "spheroidite+widmanstatten": "Spher.+Widm.",
        },
    },
    "neu": {
        "processed_dir": os.path.join(BASE_DIR, "data", "processed_neu"),
        "model_path": os.path.join(BASE_DIR, "models", "best_surface_defect_model.pth"),
        "title": "Surface Defect Classification",
        "display_names": {
            "crazing": "Crazing",
            "inclusion": "Inclusion",
            "patches": "Patches",
            "pitted_surface": "Pitted Surface",
            "rolled-in_scale": "Rolled-in Scale",
            "scratches": "Scratches",
        },
    },
}

# Active config -- set at runtime
PROCESSED_DIR = DATASET_CONFIGS["uhcs"]["processed_dir"]
MODEL_PATH = DATASET_CONFIGS["uhcs"]["model_path"]
DISPLAY_NAMES = DATASET_CONFIGS["uhcs"]["display_names"]
EVAL_TITLE = DATASET_CONFIGS["uhcs"]["title"]


def load_model(model_path, device):
    """Load the trained model and class names."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Model loaded from: {model_path}")
    print(f"  Best val accuracy: {checkpoint['val_acc'] * 100:.1f}%")
    print(f"  Trained for:       {checkpoint['epoch']} epochs")
    print(f"  Classes:           {class_names}")

    return model, class_names


def evaluate_on_test(model, test_loader, device):
    """Run model on test set, return predictions and true labels."""
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    display_names = [DISPLAY_NAMES.get(c, c) for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=display_names, yticklabels=display_names,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})

    ax.set_title(f"Confusion Matrix - {EVAL_TITLE}",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Confusion matrix saved to: {save_path}")
    plt.show()


def plot_per_class_accuracy(y_true, y_pred, class_names, save_path=None):
    """Plot per-class accuracy bar chart."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    display_names = [DISPLAY_NAMES.get(c, c) for c in class_names]

    # Sort by accuracy
    sorted_idx = np.argsort(per_class_acc)
    sorted_names = [display_names[i] for i in sorted_idx]
    sorted_acc = per_class_acc[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#FF6B6B" if a < 0.7 else "#FFD93D" if a < 0.85 else "#6BCB77" for a in sorted_acc]
    bars = ax.barh(range(len(sorted_names)), sorted_acc * 100, color=colors, edgecolor="white")

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)

    for bar, acc in zip(bars, sorted_acc):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{acc * 100:.1f}%", va="center", fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Per-class accuracy saved to: {save_path}")
    plt.show()


def show_predictions_gallery(test_dataset, y_pred, y_true, probs, class_names,
                              n_correct=5, n_incorrect=5, save_path=None):
    """Show most confident correct and incorrect predictions."""
    display_names = [DISPLAY_NAMES.get(c, c) for c in class_names]
    confidences = probs.max(axis=1)

    # Most confident correct
    correct_mask = y_pred == y_true
    correct_conf = np.where(correct_mask, confidences, 0)
    top_correct = np.argsort(correct_conf)[-n_correct:][::-1]

    # Most confident INCORRECT
    incorrect_mask = y_pred != y_true
    incorrect_conf = np.where(incorrect_mask, confidences, 0)
    top_incorrect = np.argsort(incorrect_conf)[-n_incorrect:][::-1]

    fig, axes = plt.subplots(2, max(n_correct, n_incorrect),
                             figsize=(3.5 * max(n_correct, n_incorrect), 7))

    for col, idx in enumerate(top_correct):
        if col >= n_correct:
            break
        ax = axes[0, col]
        img = test_dataset[idx][0]
        # Denormalize
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"✅ {display_names[y_pred[idx]]}\n{confidences[idx]*100:.1f}%",
                     fontsize=9, color="green")
        ax.axis("off")

    for col, idx in enumerate(top_incorrect):
        if col >= n_incorrect or confidences[idx] == 0:
            break
        ax = axes[1, col]
        img = test_dataset[idx][0]
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"❌ Pred: {display_names[y_pred[idx]]}\n"
                     f"True: {display_names[y_true[idx]]}\n{confidences[idx]*100:.1f}%",
                     fontsize=8, color="red")
        ax.axis("off")

    # Hide empty axes
    for row in range(2):
        for col in range(max(n_correct, n_incorrect)):
            if axes[row, col].get_images() == [] and not axes[row, col].texts:
                axes[row, col].axis("off")

    axes[0, 0].annotate("Most Confident CORRECT", xy=(0, 1.15),
                        xycoords="axes fraction", fontsize=13, fontweight="bold", color="green")
    axes[1, 0].annotate("Most Confident INCORRECT", xy=(0, 1.15),
                        xycoords="axes fraction", fontsize=13, fontweight="bold", color="red")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n🖼️  Predictions gallery saved to: {save_path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate microstructure/defect classifier")
    parser.add_argument("--dataset", choices=["uhcs", "neu"], default="uhcs",
                        help="Dataset to evaluate: 'uhcs' (microstructure) or 'neu' (surface defects)")
    args = parser.parse_args()

    # Apply dataset config
    cfg = DATASET_CONFIGS[args.dataset]
    PROCESSED_DIR = cfg["processed_dir"]
    MODEL_PATH = cfg["model_path"]
    DISPLAY_NAMES = cfg["display_names"]
    EVAL_TITLE = cfg["title"]

    print("=" * 62)
    print(f"  STEP 6: Model Evaluation [{args.dataset.upper()}]")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # Load model
    print(f"\n📦 Loading model...")
    model, class_names = load_model(MODEL_PATH, device)

    # Load test data
    print(f"\n📦 Loading test data...")
    eval_transform = get_transforms(is_training=False)
    test_dataset = datasets.ImageFolder(
        os.path.join(PROCESSED_DIR, "test"),
        transform=eval_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"  Test samples: {len(test_dataset)}")

    # Evaluate
    print(f"\n🔍 Running evaluation...")
    y_pred, y_true, probs = evaluate_on_test(model, test_loader, device)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 62}")
    print(f"  OVERALL ACCURACY: {accuracy * 100:.1f}%")
    print(f"  Correct: {(y_pred == y_true).sum()} / {len(y_true)}")
    print(f"{'=' * 62}")

    # Classification report
    display_names = [DISPLAY_NAMES.get(c, c) for c in class_names]
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=display_names, digits=3))

    # Plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    plot_per_class_accuracy(
        y_true, y_pred, class_names,
        save_path=os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    )

    show_predictions_gallery(
        test_dataset, y_pred, y_true, probs, class_names,
        save_path=os.path.join(OUTPUT_DIR, "predictions_gallery.png")
    )

    print("\n✅ Step 6 complete! Evaluation results saved to outputs/")
