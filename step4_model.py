"""
Step 4: Model Definition — ResNet50 Transfer Learning
=======================================================
Loads a pre-trained ResNet50 and modifies it for microstructure
classification (6 classes). Includes Grad-CAM hooks for
interpretability.

Usage:
    python step4_model.py  # Print model summary
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

NUM_CLASSES = 6
FREEZE_UP_TO = "layer3"  # Freeze everything up to (and including) layer3


# ── Grad-CAM Helper ─────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    Highlights which regions of the input image the model focuses on.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = predicted class)

        Returns:
            heatmap: numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


# ── Model Factory ────────────────────────────────────────────────────────────

def create_model(num_classes=NUM_CLASSES, pretrained=True, freeze_up_to=FREEZE_UP_TO):
    """
    Create a ResNet50 model for microstructure classification.

    Architecture:
        ResNet50 (pretrained on ImageNet)
        ├── conv1, bn1, relu, maxpool     ← FROZEN
        ├── layer1 (3 bottleneck blocks)  ← FROZEN
        ├── layer2 (4 bottleneck blocks)  ← FROZEN
        ├── layer3 (6 bottleneck blocks)  ← FROZEN
        ├── layer4 (3 bottleneck blocks)  ← TRAINABLE (fine-tuned)
        ├── avgpool                       ← TRAINABLE
        └── fc (2048 → num_classes)       ← TRAINABLE (new)

    Args:
        num_classes: Number of output classes (default: 6)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_up_to: Freeze all layers up to and including this layer

    Returns:
        model: Modified ResNet50 model
    """
    # Load pretrained ResNet50
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # Freeze layers up to freeze_up_to
    freeze = True
    for name, param in model.named_parameters():
        if freeze_up_to in name:
            freeze = False  # Start unfreezing after this layer
        if freeze:
            param.requires_grad = False

    # Replace the final FC layer
    in_features = model.fc.in_features  # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model


def get_grad_cam(model):
    """Create a GradCAM instance targeting layer4 of the model."""
    return GradCAM(model, model.layer4)


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def get_transforms(is_training=False):
    """
    Get image transforms for training or evaluation.

    Training: augmentation + normalization
    Evaluation: just normalization
    """
    from torchvision import transforms

    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_training:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  STEP 4: Model Architecture — ResNet50 Transfer Learning")
    print("=" * 62)

    model = create_model()

    # Parameter count
    total, trainable, frozen = count_parameters(model)
    print(f"\n📊 Parameter Summary:")
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print(f"  Frozen parameters:    {frozen:>12,}")
    print(f"  Trainable %:          {trainable / total * 100:>11.1f}%")

    # Test forward pass
    print(f"\n🧪 Test forward pass:")
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Input shape:  {list(dummy_input.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Test Grad-CAM
    print(f"\n🔥 Test Grad-CAM:")
    model.train()  # Grad-CAM needs gradients
    grad_cam = get_grad_cam(model)
    test_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap = grad_cam.generate(test_input)
    print(f"  Heatmap shape: {heatmap.shape}")
    print(f"  Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    print(f"\n✅ Model architecture verified!")
