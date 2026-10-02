import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES

# ── Classifier head in_features per architecture ─────────────────────────────
_HEAD_IN = {
    "vgg16":             4096,
    "resnet50":          2048,
    "efficientnet_b3":   1536,
    "mobilenet_v3_large":1280,
}


def build_model(arch: str, num_classes: int = NUM_CLASSES,
                dropout: float = 0.3) -> nn.Module:
    if arch == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(4096, num_classes)
        )

    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2048, num_classes)
        )

    elif arch == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(1536, num_classes)
        )

    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )
        m.classifier[3] = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(1280, num_classes)
        )

    else:
        raise ValueError(f"Unknown arch: {arch}")

    return m


def freeze_backbone(model: nn.Module, arch: str):
    """Freeze everything; then re-enable gradients on the classifier head."""
    for p in model.parameters():
        p.requires_grad = False

    head = {
        "vgg16":             model.classifier,
        "resnet50":          model.fc,
        "efficientnet_b3":   model.classifier,
        "mobilenet_v3_large":model.classifier,
    }[arch]

    for p in head.parameters():
        p.requires_grad = True


def unfreeze_top(model: nn.Module, arch: str):
    """Unfreeze the last few backbone layers for stage-2 fine-tuning."""
    if arch == "vgg16":
        # Unfreeze features[24:] (last conv block) + classifier
        for layer in list(model.features.children())[24:]:
            for p in layer.parameters():
                p.requires_grad = True

    elif arch == "resnet50":
        for name, p in model.named_parameters():
            if "layer4" in name or "fc" in name:
                p.requires_grad = True

    elif arch == "efficientnet_b3":
        for layer in list(model.features.children())[-3:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

    elif arch == "mobilenet_v3_large":
        for layer in list(model.features.children())[-3:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True


def get_gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    """Return the target layer for GradCAM (last spatial feature map)."""
    if arch == "vgg16":
        return model.features[28]       # last Conv2d before MaxPool
    elif arch == "resnet50":
        return model.layer4[-1]         # last Bottleneck block
    elif arch == "efficientnet_b3":
        return model.features[-1]       # last MBConv block
    elif arch == "mobilenet_v3_large":
        return model.features[-1]       # last InvertedResidual block
    raise ValueError(f"Unknown arch: {arch}")
