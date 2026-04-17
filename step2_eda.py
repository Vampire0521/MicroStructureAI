"""
Step 2: Exploratory Data Analysis (EDA)
========================================
Deeper analysis of the UHCS microstructure dataset — image statistics,
pixel intensity distributions, and class-wise comparisons.

Usage:
    python step2_eda.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict


# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "UHCS")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

CLASS_DISPLAY_NAMES = {
    "pearlite":                   "Pearlite",
    "spheroidite":                "Spheroidite",
    "martensite":                 "Martensite",
    "network":                    "Network Carbides",
    "widmanstatten":              "Widmanstätten",
    "pearlite+spheroidite":       "Pearlite+Spheroidite",
    "spheroidite+widmanstatten":  "Spheroidite+Widmanstätten",
}


def get_image_paths(cls):
    """Get all image paths for a class."""
    cls_dir = os.path.join(DATA_DIR, cls)
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(cls_dir, f"*{ext}")))
    return sorted(images)


def get_all_classes():
    """Get class names that have images."""
    classes = []
    for cls in CLASS_DISPLAY_NAMES.keys():
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_dir) and len(get_image_paths(cls)) > 0:
            classes.append(cls)
    return classes


def analyze_pixel_intensity(classes):
    """Analyze pixel intensity distributions per class."""
    print("\n🔬 Pixel Intensity Analysis")
    print("─" * 40)

    class_means = defaultdict(list)
    class_stds = defaultdict(list)

    for cls in classes:
        paths = get_image_paths(cls)
        for p in paths:
            try:
                img = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
                class_means[cls].append(img.mean())
                class_stds[cls].append(img.std())
            except Exception:
                pass

    # Print stats
    for cls in classes:
        if class_means[cls]:
            mean_val = np.mean(class_means[cls])
            std_val = np.mean(class_stds[cls])
            print(f"  {CLASS_DISPLAY_NAMES[cls]:<25}  "
                  f"Mean: {mean_val:.3f}  Std: {std_val:.3f}")

    return class_means, class_stds


def plot_intensity_distributions(class_means, class_stds, save_path=None):
    """Plot pixel intensity distribution comparison across classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    classes = list(class_means.keys())
    display_names = [CLASS_DISPLAY_NAMES[c] for c in classes]
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))

    # Mean intensity boxplot
    data_means = [class_means[c] for c in classes]
    bp1 = ax1.boxplot(data_means, labels=display_names, patch_artist=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
    ax1.set_title("Mean Pixel Intensity per Class", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Mean Intensity (0–1)")
    ax1.tick_params(axis="x", rotation=45)

    # Std deviation boxplot
    data_stds = [class_stds[c] for c in classes]
    bp2 = ax2.boxplot(data_stds, labels=display_names, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
    ax2.set_title("Pixel Intensity Std Dev per Class", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Std Deviation")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Intensity distributions saved to: {save_path}")
    plt.show()


def plot_dimension_histograms(classes, save_path=None):
    """Plot histograms of image dimensions."""
    heights, widths = [], []
    for cls in classes:
        for p in get_image_paths(cls):
            try:
                img = Image.open(p)
                w, h = img.size
                heights.append(h)
                widths.append(w)
            except Exception:
                pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(widths, bins=30, color="#4ECDC4", edgecolor="white", alpha=0.8)
    ax1.set_title("Image Width Distribution", fontweight="bold")
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Count")

    ax2.hist(heights, bins=30, color="#FF6B6B", edgecolor="white", alpha=0.8)
    ax2.set_title("Image Height Distribution", fontweight="bold")
    ax2.set_xlabel("Height (pixels)")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n📐 Dimension histograms saved to: {save_path}")
    plt.show()


def plot_class_gallery(classes, n_samples=4, save_path=None):
    """Plot a detailed gallery with n_samples per class."""
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, n_samples,
                             figsize=(3.5 * n_samples, 3.5 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for row, cls in enumerate(classes):
        paths = get_image_paths(cls)
        # Pick evenly spaced samples
        indices = np.linspace(0, len(paths) - 1, min(n_samples, len(paths)), dtype=int)

        for col in range(n_samples):
            ax = axes[row, col]
            if col < len(indices):
                try:
                    img = np.array(Image.open(paths[indices[col]]).convert("L"))
                    ax.imshow(img, cmap="gray")
                    ax.set_title(f"{img.shape[1]}×{img.shape[0]}", fontsize=9)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center",
                            transform=ax.transAxes)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CLASS_DISPLAY_NAMES[cls], fontsize=11,
                              fontweight="bold", rotation=0, labelpad=100)

    plt.suptitle("UHCS Dataset — Microstructure Gallery", fontsize=16,
                 fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n🖼️  Gallery saved to: {save_path}")
    plt.show()


def compute_class_balance_metrics(classes):
    """Compute and display class balance metrics."""
    print("\n⚖️  Class Balance Analysis")
    print("─" * 50)

    counts = {}
    for cls in classes:
        counts[cls] = len(get_image_paths(cls))

    total = sum(counts.values())
    max_count = max(counts.values())
    min_count = min(counts.values())

    print(f"  Total images:       {total}")
    print(f"  Largest class:      {max_count} images")
    print(f"  Smallest class:     {min_count} images")
    print(f"  Imbalance ratio:    {max_count / min_count:.1f}x")
    print()

    for cls in classes:
        pct = counts[cls] / total * 100
        bar = "█" * int(pct * 2)
        print(f"  {CLASS_DISPLAY_NAMES[cls]:<25} {counts[cls]:>5}  ({pct:5.1f}%)  {bar}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  STEP 2: Exploratory Data Analysis")
    print("=" * 62)

    classes = get_all_classes()
    if not classes:
        print("\n❌ No classes found. Run step1_data_setup.py first.")
        exit(1)

    print(f"\nFound {len(classes)} classes with images.")

    # 1. Class balance
    compute_class_balance_metrics(classes)

    # 2. Pixel intensity analysis
    class_means, class_stds = analyze_pixel_intensity(classes)
    plot_intensity_distributions(
        class_means, class_stds,
        save_path=os.path.join(OUTPUT_DIR, "intensity_distributions.png")
    )

    # 3. Dimension histograms
    plot_dimension_histograms(
        classes,
        save_path=os.path.join(OUTPUT_DIR, "dimension_histograms.png")
    )

    # 4. Class gallery
    plot_class_gallery(
        classes, n_samples=4,
        save_path=os.path.join(OUTPUT_DIR, "eda_gallery.png")
    )

    print("\n✅ Step 2 complete! EDA plots saved to outputs/")
