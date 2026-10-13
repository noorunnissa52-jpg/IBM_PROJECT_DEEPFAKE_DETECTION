"""Train the detector.

Example:
    python train.py --epochs 10 --arch efficientnet_b0
"""
import argparse
import time

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import get_loaders
from model import build_model, save_checkpoint
from utils import get_device, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(config.DATA_DIR))
    p.add_argument("--arch", default="efficientnet_b0",
                   choices=["efficientnet_b0", "resnet18", "resnet50"])
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--no_pretrained", action="store_true", help="train from scratch")
    p.add_argument("--out", default=str(config.CKPT_DIR / "best.pt"))
    args = p.parse_args()

    set_seed(config.SEED)
    device = get_device()
    print(f"Device: {device}")

    loaders, dsets = get_loaders(args.data_dir, args.img_size, args.batch_size, args.workers)
    print({k: len(v) for k, v in dsets.items()}, "images; classes:", dsets["train"].classes)

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    config.CKPT_DIR.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time()-t0:.0f}s")
        if va_acc > best_acc:
            best_acc = va_acc
            save_checkpoint(args.out, model, args.arch, args.img_size, config.CLASS_NAMES,
                            {"val_acc": va_acc, "epoch": epoch})
            print(f"  -> saved best model to {args.out}")
    print(f"Done. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
