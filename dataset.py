"""Data loading. Expects the folder layout:

data/{train,val,test}/{real,fake}/*.jpg|png
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


def build_transforms(img_size: int = config.IMG_SIZE):
    # NOTE: no heavy blur / aggressive resizing in training augmentation, because
    # those operations destroy the high-frequency artifacts detectors rely on.
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    return train_tf, eval_tf


def _check(split_dir: Path):
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} not found. Put images in data/<split>/{{real,fake}}/ "
            "or run `python make_demo_data.py` to generate a tiny demo dataset."
        )


def get_loaders(data_dir=config.DATA_DIR, img_size=config.IMG_SIZE,
                batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    train_tf, eval_tf = build_transforms(img_size)
    data_dir = Path(data_dir)
    loaders, datasets_ = {}, {}
    for split, tf in (("train", train_tf), ("val", eval_tf), ("test", eval_tf)):
        d = data_dir / split
        if split == "test" and not d.exists():
            continue
        _check(d)
        ds = datasets.ImageFolder(d, transform=tf)
        assert ds.classes == config.CLASS_NAMES, (
            f"Expected class folders {config.CLASS_NAMES}, found {ds.classes}")
        datasets_[split] = ds
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, pin_memory=True)
    return loaders, datasets_
