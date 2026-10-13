"""Evaluate a trained model on data/test: accuracy, precision/recall/F1, AUC,
confusion matrix and ROC curve (saved to outputs/).

Example:
    python evaluate.py --ckpt checkpoints/best.pt
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)
from tqdm import tqdm

import config
from dataset import get_loaders
from model import load_checkpoint
from utils import get_device


@torch.no_grad()
def collect(model, loader, device):
    ys, ps = [], []
    for x, y in tqdm(loader, desc="eval"):
        prob = torch.softmax(model(x.to(device)), dim=1)[:, 1]  # P(real)
        ys.append(y.numpy()); ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(config.CKPT_DIR / "best.pt"))
    p.add_argument("--data_dir", default=str(config.DATA_DIR))
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    device = get_device()
    model, ck = load_checkpoint(args.ckpt, device)
    loaders, _ = get_loaders(args.data_dir, ck["img_size"], num_workers=0)
    y, p_real = collect(model, loaders[args.split], device)
    pred = (p_real >= 0.5).astype(int)  # 1 = real

    print(classification_report(y, pred, target_names=config.CLASS_NAMES, digits=4))
    print("Accuracy:", (pred == y).mean())
    print("ROC-AUC :", roc_auc_score(y, p_real))

    config.OUT_DIR.mkdir(exist_ok=True)
    cm = confusion_matrix(y, pred)
    ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES).plot(cmap="Blues")
    plt.title("Confusion matrix"); plt.savefig(config.OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y, p_real)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, p_real):.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(); plt.title("ROC curve"); plt.savefig(config.OUT_DIR / "roc_curve.png", dpi=150)
    print(f"Saved plots to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
