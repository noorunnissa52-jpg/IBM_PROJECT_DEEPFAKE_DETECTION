"""Model: ImageNet-pretrained backbone + 2-class head (fake / real)."""
import torch
import torch.nn as nn
from torchvision import models


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True, num_classes: int = 2):
    weights = "DEFAULT" if pretrained else None
    try:
        return _make(arch, weights, num_classes)
    except Exception as e:  # e.g. no internet to download weights
        if pretrained:
            print(f"[warn] Could not load pretrained weights ({e.__class__.__name__}); "
                  "falling back to random init.")
            return _make(arch, None, num_classes)
        raise


def _make(arch, weights, num_classes):
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "resnet18":
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def save_checkpoint(path, model, arch, img_size, class_names, extra=None):
    torch.save({"state_dict": model.state_dict(), "arch": arch, "img_size": img_size,
                "class_names": class_names, **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
