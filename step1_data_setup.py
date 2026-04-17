"""
Step 1: Data Setup & Verification
==================================
Verifies that the UHCS dataset is properly organized, prints class
distribution, and generates a sample gallery of micrographs.

Usage:
    python step1_data_setup.py

Prerequisites:
    1. Download the UHCS dataset from Kaggle:
       https://www.kaggle.com/datasets/vanvalkenberg/ultrahighcarbonsteel
       (or from NIST: https://materialsdata.nist.gov/handle/11256/940)
    2. Extract it into data/UHCS/ with subfolders per class:
       data/UHCS/
       ├── pearlite/
       ├── spheroidite/
       ├── martensite/
       ├── network/
       ├── widmanstatten/
       ├── pearlite+spheroidite/
       └── spheroidite+widmanstatten/
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter


# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "UHCS")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Expected class folders (UHCS dataset)
EXPECTED_CLASSES = [
    "pearlite",
    "spheroidite",
    "martensite",
    "network",
    "widmanstatten",
    "pearlite+spheroidite",
    "spheroidite+widmanstatten",
]

# Supported image extensions
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def check_dataset():
    """Verify dataset folder structure exists and report class distribution."""
    print("=" * 62)
    print("  STEP 1: Dataset Verification")
    print("=" * 62)

    if not os.path.exists(DATA_DIR):
        print(f"\n❌ Data directory not found: {DATA_DIR}")
        print("\nPlease download the UHCS dataset and extract it to:")
        print(f"  {DATA_DIR}")
        print("\nDownload from:")
        print("  Kaggle: https://www.kaggle.com/datasets/vanvalkenberg/ultrahighcarbonsteel")
        print("  NIST:   https://materialsdata.nist.gov/handle/11256/940")
        return None

    print(f"\n✅ Data directory found: {DATA_DIR}")

    # Check each class folder
    class_counts = {}
    total_images = 0

    print(f"\n{'Class':<30} {'Images':>8}  {'Status'}")
    print("─" * 55)

    for cls in EXPECTED_CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_dir):
            images = []
            for ext in IMAGE_EXTENSIONS:
                images.extend(glob.glob(os.path.join(cls_dir, f"*{ext}")))
            count = len(images)
            class_counts[cls] = count
            total_images += count
            status = "✅" if count > 0 else "⚠️ Empty"
            print(f"  {cls:<28} {count:>6}  {status}")
        else:
            class_counts[cls] = 0
            print(f"  {cls:<28} {'N/A':>6}  ❌ Missing")

    print("─" * 55)
    print(f"  {'TOTAL':<28} {total_images:>6}")
    print()

    if total_images == 0:
        print("❌ No images found. Please check the dataset extraction.")
        return None

    return class_counts


def get_image_paths(cls):
    """Get all image file paths for a given class."""
    cls_dir = os.path.join(DATA_DIR, cls)
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(cls_dir, f"*{ext}")))
    return sorted(images)


def analyze_dimensions(class_counts):
    """Analyze image dimensions across the dataset."""
    print("\n📐 Image Dimension Analysis")
    print("─" * 40)

    heights, widths = [], []
    for cls, count in class_counts.items():
        if count == 0:
            continue
        paths = get_image_paths(cls)
        for p in paths[:50]:  # Sample first 50 per class for speed
            try:
                img = Image.open(p)
                w, h = img.size
                heights.append(h)
                widths.append(w)
            except Exception:
                pass

    if heights:
        print(f"  Height range: {min(heights)} – {max(heights)} px")
        print(f"  Width  range: {min(widths)} – {max(widths)} px")
        print(f"  Most common: {Counter(zip(widths, heights)).most_common(3)}")
    return heights, widths


def plot_sample_gallery(class_counts, save_path=None):
    """Plot 3 sample images from each class."""
    n_classes = sum(1 for c in class_counts.values() if c > 0)
    samples_per_class = 3

    fig, axes = plt.subplots(n_classes, samples_per_class,
                             figsize=(4 * samples_per_class, 4 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    row = 0
    for cls, count in class_counts.items():
        if count == 0:
            continue
        paths = get_image_paths(cls)
        for col in range(samples_per_class):
            ax = axes[row, col]
            if col < len(paths):
                try:
                    img = Image.open(paths[col]).convert("L")  # Grayscale
                    ax.imshow(np.array(img), cmap="gray")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{e}",
                            ha="center", va="center", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No image",
                        ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            if col == 0:
                ax.set_title(f"{cls}\n({count} images)", fontsize=11, fontweight="bold")

        row += 1

    plt.suptitle("UHCS Dataset — Sample Micrographs", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n📸 Sample gallery saved to: {save_path}")

    plt.show()


def plot_class_distribution(class_counts, save_path=None):
    """Plot class distribution bar chart."""
    classes = [c for c, n in class_counts.items() if n > 0]
    counts = [class_counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(classes)), counts, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(classes))))

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("UHCS Dataset — Class Distribution", fontsize=14, fontweight="bold")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Distribution chart saved to: {save_path}")

    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    class_counts = check_dataset()

    if class_counts is None:
        sys.exit(1)

    # Analyze image dimensions
    analyze_dimensions(class_counts)

    # Plot class distribution
    plot_class_distribution(
        class_counts,
        save_path=os.path.join(OUTPUT_DIR, "class_distribution.png")
    )

    # Plot sample gallery
    plot_sample_gallery(
        class_counts,
        save_path=os.path.join(OUTPUT_DIR, "sample_gallery.png")
    )

    print("\n✅ Step 1 complete! Dataset verified and visualized.")
