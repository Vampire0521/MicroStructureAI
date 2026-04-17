"""
Step 3: Preprocessing & Augmentation
======================================
Resizes all images to 224x224, splits into train/val/test sets,
and applies data augmentation for class balancing.

Supports both datasets:
    python step3_preprocessing.py                # UHCS (default)
    python step3_preprocessing.py --dataset neu   # NEU surface defects
"""

import os
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt


# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMAGE_SIZE = 224  # ResNet50 standard input
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# Dataset-specific configs
DATASET_CONFIGS = {
    "uhcs": {
        "data_dir": os.path.join(BASE_DIR, "data", "UHCS"),
        "processed_dir": os.path.join(BASE_DIR, "data", "processed"),
        "class_names": [
            "network", "pearlite", "pearlite+spheroidite",
            "pearlite+widmanstatten", "spheroidite", "spheroidite+widmanstatten",
        ],
    },
    "neu": {
        "data_dir": os.path.join(BASE_DIR, "data", "NEU"),
        "processed_dir": os.path.join(BASE_DIR, "data", "processed_neu"),
        "class_names": [
            "crazing", "inclusion", "patches",
            "pitted_surface", "rolled-in_scale", "scratches",
        ],
    },
}

# Active config — set at runtime
DATA_DIR = DATASET_CONFIGS["uhcs"]["data_dir"]
PROCESSED_DIR = DATASET_CONFIGS["uhcs"]["processed_dir"]
CLASS_NAMES = DATASET_CONFIGS["uhcs"]["class_names"]


def get_image_paths(cls):
    """Get all image file paths for a given class."""
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.exists(cls_dir):
        return []
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(cls_dir, f"*{ext}")))
    return sorted(images)


def load_and_resize_image(path, size=IMAGE_SIZE):
    """Load an image, convert to RGB, and resize to (size, size)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def build_dataset():
    """Load all images, resize, and return arrays + labels."""
    print("📦 Loading and resizing images...")

    all_images = []
    all_labels = []
    all_paths = []

    for cls_idx, cls in enumerate(CLASS_NAMES):
        paths = get_image_paths(cls)
        if not paths:
            print(f"  ⚠️  {cls}: No images found, skipping")
            continue

        print(f"  {cls:<28} → {len(paths):>4} images", end="")
        for p in paths:
            try:
                img = load_and_resize_image(p)
                all_images.append(img)
                all_labels.append(cls_idx)
                all_paths.append(p)
            except Exception as e:
                print(f"\n    ⚠️ Error loading {os.path.basename(p)}: {e}")
        print(f"  (loaded {sum(1 for l in all_labels if l == cls_idx)})")

    X = np.array(all_images, dtype=np.uint8)
    y = np.array(all_labels, dtype=np.int64)

    print(f"\n✅ Total: {len(X)} images loaded")
    print(f"   Shape: {X.shape} (N, H, W, C)")
    print(f"   Labels: {dict(Counter(y))}")

    return X, y, all_paths


def stratified_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Stratified 70/15/15 split.
    Ensures each class is proportionally represented in all splits.
    """
    # First split: 85% train+val, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: from 85%, take val_size/(1-test_size) for validation
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_as_folder_structure(X, y, split_name, base_dir):
    """
    Save images into folder structure for PyTorch ImageFolder:
      base_dir/split_name/class_name/image_XXXX.png
    """
    split_dir = os.path.join(base_dir, split_name)

    for cls_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(split_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

    for i, (img, label) in enumerate(zip(X, y)):
        cls = CLASS_NAMES[label]
        save_path = os.path.join(split_dir, cls, f"img_{i:05d}.png")
        Image.fromarray(img).save(save_path)


def compute_class_weights(y_train):
    """Compute class weights inversely proportional to frequency."""
    counts = Counter(y_train)
    total = len(y_train)
    n_classes = len(CLASS_NAMES)

    weights = np.zeros(n_classes)
    for cls_idx in range(n_classes):
        if counts[cls_idx] > 0:
            weights[cls_idx] = total / (n_classes * counts[cls_idx])
        else:
            weights[cls_idx] = 1.0

    return weights


def plot_split_distribution(y_train, y_val, y_test, save_path=None):
    """Plot class distribution across train/val/test splits."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (y_split, title) in zip(axes, [
        (y_train, f"Train ({len(y_train)})"),
        (y_val, f"Validation ({len(y_val)})"),
        (y_test, f"Test ({len(y_test)})")
    ]):
        counts = Counter(y_split)
        classes = range(len(CLASS_NAMES))
        values = [counts.get(c, 0) for c in classes]
        colors = plt.cm.Set2(np.linspace(0, 1, len(CLASS_NAMES)))

        ax.bar(classes, values, color=colors)
        ax.set_xticks(classes)
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Count")

        for i, v in enumerate(values):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=8)

    plt.suptitle("Dataset Split Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Split distribution saved to: {save_path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess microstructure or surface defect images")
    parser.add_argument("--dataset", choices=["uhcs", "neu"], default="uhcs",
                        help="Dataset to preprocess: 'uhcs' (microstructure) or 'neu' (surface defects)")
    args = parser.parse_args()

    # Apply dataset config
    cfg = DATASET_CONFIGS[args.dataset]
    DATA_DIR = cfg["data_dir"]
    PROCESSED_DIR = cfg["processed_dir"]
    CLASS_NAMES = cfg["class_names"]

    print("=" * 62)
    print(f"  STEP 3: Preprocessing & Splitting [{args.dataset.upper()}]")
    print("=" * 62)

    # 1. Load and resize all images
    X, y, paths = build_dataset()

    if len(X) == 0:
        print(f"No images loaded. Check {DATA_DIR}/ directory.")
        exit(1)

    # 2. Stratified split (70/15/15)
    print("\n✂️  Splitting into train/val/test (70/15/15, stratified)...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    print(f"  Train:      {len(X_train):>5} images")
    print(f"  Validation: {len(X_val):>5} images")
    print(f"  Test:       {len(X_test):>5} images")

    # 3. Compute class weights
    class_weights = compute_class_weights(y_train)
    print(f"\n⚖️  Class weights:")
    for i, (cls, w) in enumerate(zip(CLASS_NAMES, class_weights)):
        print(f"  {cls:<28} {w:.3f}")

    # 4. Save as folder structure for PyTorch ImageFolder
    print(f"\n💾 Saving to {PROCESSED_DIR}/...")
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    save_as_folder_structure(X_train, y_train, "train", PROCESSED_DIR)
    save_as_folder_structure(X_val, y_val, "val", PROCESSED_DIR)
    save_as_folder_structure(X_test, y_test, "test", PROCESSED_DIR)

    # 5. Save class weights and metadata
    np.savez(
        os.path.join(PROCESSED_DIR, "metadata.npz"),
        class_names=CLASS_NAMES,
        class_weights=class_weights,
        train_labels=y_train,
        val_labels=y_val,
        test_labels=y_test,
    )
    print(f"  ✅ Metadata saved to {os.path.join(PROCESSED_DIR, 'metadata.npz')}")

    # 6. Plot split distribution
    plot_split_distribution(
        y_train, y_val, y_test,
        save_path=os.path.join(OUTPUT_DIR, "split_distribution.png")
    )

    # 7. Summary
    print("\n" + "=" * 62)
    print("  ✅ Step 3 complete!")
    print(f"  Processed data saved to: {PROCESSED_DIR}")
    print(f"  Structure: {PROCESSED_DIR}/train|val|test/<class_name>/")
    print("=" * 62)
