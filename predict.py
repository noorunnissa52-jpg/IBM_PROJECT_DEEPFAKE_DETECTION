"""Classify one image or a whole folder as REAL or FAKE.

Examples:
    python predict.py --input photo.jpg
    python predict.py --input some_folder/ --gradcam
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

import config
from dataset import build_transforms
from model import load_checkpoint
from utils import get_device

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict_image(model, img: Image.Image, tf, device):
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    idx = int(probs.argmax())
    return config.CLASS_NAMES[idx].upper(), float(probs[idx]), {
        n: float(p) for n, p in zip(config.CLASS_NAMES, probs)}


def gradcam_overlay(model, img: Image.Image, tf, device, arch):
    """Simple Grad-CAM on the last conv block to show *where* the model looked."""
    import numpy as np
    import matplotlib.cm as cm
    target = model.features[-1] if arch.startswith("efficientnet") else model.layer4[-1]
    acts, grads = {}, {}

    def fwd_hook(module, inp, out):
        acts["a"] = out
        out.register_hook(lambda g: grads.__setitem__("g", g))  # tensor hook: safe with in-place ops

    h = target.register_forward_hook(fwd_hook)
    x = tf(img.convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)
    model.zero_grad()
    out = model(x)
    out[0, out.argmax()].backward()
    h.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1))[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(img.size, Image.BILINEAR)
    heat = (cm.jet(np.array(cam_img) / 255.0)[..., :3] * 255).astype("uint8")
    return Image.blend(img.convert("RGB"), Image.fromarray(heat), 0.4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--ckpt", default=str(config.CKPT_DIR / "best.pt"))
    p.add_argument("--gradcam", action="store_true", help="save Grad-CAM heatmaps to outputs/")
    args = p.parse_args()

    device = get_device()
    model, ck = load_checkpoint(args.ckpt, device)
    _, tf = build_transforms(ck["img_size"])

    inp = Path(args.input)
    files = [inp] if inp.is_file() else sorted(f for f in inp.rglob("*") if f.suffix.lower() in EXTS)
    if not files:
        raise SystemExit("No images found.")
    config.OUT_DIR.mkdir(exist_ok=True)
    for f in files:
        img = Image.open(f)
        label, conf, probs = predict_image(model, img, tf, device)
        print(f"{f.name:40s} -> {label:4s} ({conf*100:5.1f}%)  "
              f"[fake={probs['fake']:.3f}, real={probs['real']:.3f}]")
        if args.gradcam:
            gradcam_overlay(model, img, tf, device, ck["arch"]).save(
                config.OUT_DIR / f"gradcam_{f.stem}.png")


if __name__ == "__main__":
    main()
