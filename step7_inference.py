"""
Step 7: Inference Pipeline
===========================
Complete inference pipeline: load image -> preprocess -> classify ->
Grad-CAM visualization -> knowledge base lookup -> generate report.

Usage:
    python step7_inference.py path/to/image.png                    # Micro mode (default)
    python step7_inference.py path/to/image.png --mode surface     # Surface defect mode
    python step7_inference.py path/to/image.png --save result.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
import cv2

from step4_model import create_model, get_transforms, GradCAM
from knowledge_base import (
    UHCS_TO_KB_MAP,
    get_knowledge,
    format_report,
)
from defect_knowledge_base import (
    NEU_TO_KB_MAP,
    get_surface_defect,
    get_micro_defect,
    format_surface_report,
    format_defect_flag,
)


# -- Configuration ---------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
MICRO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_microstructure_model.pth")
SURFACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_surface_defect_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
IMAGE_SIZE = 224


# -- MicrostructureAnalyzer -------------------------------------------------------

class MicrostructureAnalyzer:
    """
    End-to-end microstructure analysis pipeline.

    Usage:
        analyzer = MicrostructureAnalyzer()
        result = analyzer.analyze("path/to/micrograph.png")
        print(result)

        # With visualization
        result = analyzer.analyze_and_visualize("image.png", "output.png")
    """

    def __init__(self, model_path=MICRO_MODEL_PATH, device=None):
        """Initialize the analyzer: load model, set up Grad-CAM."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🔬 Initializing MicrostructureAnalyzer...")
        print(f"   Device: {self.device}")

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint["class_names"]  # Folder names from ImageFolder
        num_classes = len(self.class_names)

        self.model = create_model(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Grad-CAM
        self.grad_cam = GradCAM(self.model, self.model.layer4)

        # Transforms
        self.transform = get_transforms(is_training=False)

        # Build KB mapping
        self.kb_map = {}
        for folder_name in self.class_names:
            kb_key = UHCS_TO_KB_MAP.get(folder_name, folder_name)
            self.kb_map[folder_name] = kb_key

        print(f"   Classes: {self.class_names}")
        print(f"   Model loaded: {os.path.basename(model_path)}")
        print(f"   ✅ Ready!\n")

    def preprocess(self, image_path):
        """Load and preprocess an image for inference."""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img, img_tensor

    def predict(self, image_path):
        """
        Classify a microstructure image and return full analysis.

        Args:
            image_path: Path to the micrograph image

        Returns:
            dict with keys: class_name, kb_name, confidence, all_probs, knowledge
        """
        original_img, img_tensor = self.preprocess(image_path)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # Map to knowledge base
        kb_name = self.kb_map.get(pred_class, pred_class)
        knowledge = get_knowledge(kb_name)

        # All class probabilities
        all_probs = {self.kb_map.get(c, c): float(probs[i])
                     for i, c in enumerate(self.class_names)}

        return {
            "class_name": pred_class,
            "kb_name": kb_name,
            "confidence": confidence,
            "all_probs": all_probs,
            "knowledge": knowledge,
            "predicted_index": pred_idx,
        }

    def generate_gradcam(self, image_path, target_class=None):
        """Generate Grad-CAM heatmap for an image."""
        _, img_tensor = self.preprocess(image_path)
        img_tensor.requires_grad_(True)

        self.model.train()  # Need gradients
        heatmap = self.grad_cam.generate(img_tensor, target_class)
        self.model.eval()

        return heatmap

    def create_gradcam_overlay(self, image_path, heatmap, alpha=0.5):
        """Overlay Grad-CAM heatmap on the original image."""
        # Load original image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img_array = np.array(img)

        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = (img_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
        return overlay, heatmap_resized

    def analyze(self, image_path):
        """Full analysis: predict + knowledge lookup + defect flag check."""
        result = self.predict(image_path)
        report = format_report(
            result["kb_name"],
            result["confidence"],
            result["all_probs"]
        )

        # Check for microstructural defect flag
        defect_flag = format_defect_flag(result["kb_name"])
        if defect_flag:
            report += "\n" + defect_flag
            result["is_defect"] = True
            result["defect_info"] = get_micro_defect(result["kb_name"])
        else:
            result["is_defect"] = False
            result["defect_info"] = None

        result["report"] = report
        return result

    def analyze_and_visualize(self, image_path, save_path=None):
        """
        Full analysis with Grad-CAM visualization.

        Args:
            image_path: Path to micrograph
            save_path: Optional path to save the visualization

        Returns:
            dict with analysis result + visualization
        """
        # Get prediction
        result = self.analyze(image_path)

        # Generate Grad-CAM
        heatmap = self.generate_gradcam(image_path, result["predicted_index"])
        overlay, heatmap_resized = self.create_gradcam_overlay(image_path, heatmap)

        # Create visualization figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Original image
        original = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        axes[0].imshow(np.array(original))
        axes[0].set_title("Original Micrograph", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Grad-CAM heatmap
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title("Grad-CAM Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        kb = result["knowledge"]
        fig.suptitle(
            f"Predicted: {result['kb_name'].replace('_', ' ')}  |  "
            f"Confidence: {result['confidence'] * 100:.1f}%",
            fontsize=14, fontweight="bold", y=1.02
        )

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n📸 Visualization saved to: {save_path}")

        plt.show()
        plt.close()

        result["overlay"] = overlay
        result["heatmap"] = heatmap_resized
        return result


# -- SurfaceDefectAnalyzer --------------------------------------------------------

class SurfaceDefectAnalyzer:
    """
    Surface defect analysis pipeline using NEU-trained model.

    Usage:
        analyzer = SurfaceDefectAnalyzer()
        result = analyzer.analyze("path/to/surface_image.png")
        print(result["report"])
    """

    def __init__(self, model_path=SURFACE_MODEL_PATH, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing SurfaceDefectAnalyzer...")
        print(f"   Device: {self.device}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint["class_names"]
        num_classes = len(self.class_names)

        self.model = create_model(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.grad_cam = GradCAM(self.model, self.model.layer4)
        self.transform = get_transforms(is_training=False)

        # NEU folder name -> KB key
        self.kb_map = {}
        for folder_name in self.class_names:
            self.kb_map[folder_name] = NEU_TO_KB_MAP.get(folder_name, folder_name)

        print(f"   Classes: {self.class_names}")
        print(f"   Ready!\n")

    def preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img, img_tensor

    def predict(self, image_path):
        original_img, img_tensor = self.preprocess(image_path)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        kb_name = self.kb_map.get(pred_class, pred_class)

        all_probs = {self.kb_map.get(c, c): float(probs[i])
                     for i, c in enumerate(self.class_names)}

        return {
            "class_name": pred_class,
            "kb_name": kb_name,
            "confidence": confidence,
            "all_probs": all_probs,
            "defect_info": get_surface_defect(kb_name),
            "predicted_index": pred_idx,
        }

    def generate_gradcam(self, image_path, target_class=None):
        _, img_tensor = self.preprocess(image_path)
        img_tensor.requires_grad_(True)
        self.model.train()
        heatmap = self.grad_cam.generate(img_tensor, target_class)
        self.model.eval()
        return heatmap

    def create_gradcam_overlay(self, image_path, heatmap, alpha=0.5):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img_array = np.array(img)
        heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (img_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
        return overlay, heatmap_resized

    def analyze(self, image_path):
        result = self.predict(image_path)
        report = format_surface_report(result["kb_name"], result["confidence"])
        result["report"] = report
        return result

    def analyze_and_visualize(self, image_path, save_path=None):
        result = self.analyze(image_path)
        heatmap = self.generate_gradcam(image_path, result["predicted_index"])
        overlay, heatmap_resized = self.create_gradcam_overlay(image_path, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        original = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        axes[0].imshow(np.array(original))
        axes[0].set_title("Original Surface", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title("Grad-CAM Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(
            f"Defect: {result['kb_name'].replace('_', ' ')}  |  "
            f"Confidence: {result['confidence'] * 100:.1f}%",
            fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\nVisualization saved to: {save_path}")
        plt.show()
        plt.close()

        result["overlay"] = overlay
        result["heatmap"] = heatmap_resized
        return result


# -- CLI Interface ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Steel Image Analyzer")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--mode", choices=["micro", "surface"], default="micro",
                        help="Analysis mode: 'micro' (microstructure) or 'surface' (defect)")
    parser.add_argument("--save", help="Save visualization to this path")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    save_path = args.save or os.path.join(OUTPUT_DIR, "inference_result.png")

    if args.mode == "surface":
        analyzer = SurfaceDefectAnalyzer(device=args.device)
    else:
        analyzer = MicrostructureAnalyzer(device=args.device)

    result = analyzer.analyze_and_visualize(args.image, save_path=save_path)
    print(result["report"])


if __name__ == "__main__":
    main()
