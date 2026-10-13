"""Split a flat dataset into train/val/test.

Use this if your downloaded data is just two folders (real images, fake images):

    python split_dataset.py --real_dir /path/to/real --fake_dir /path/to/fake \
                            --train 0.7 --val 0.15      # remaining 0.15 -> test

IMPORTANT: if multiple images come from the same person/video, split by identity/video
yourself first, otherwise near-duplicates leak between splits and inflate your accuracy.
"""
import argparse
import random
import shutil
from pathlib import Path

import config

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", required=True)
    p.add_argument("--fake_dir", required=True)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--move", action="store_true", help="move instead of copy")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rnd = random.Random(args.seed)
    op = shutil.move if args.move else shutil.copy2

    for cls, src in (("real", args.real_dir), ("fake", args.fake_dir)):
        files = sorted(f for f in Path(src).rglob("*") if f.suffix.lower() in EXTS)
        rnd.shuffle(files)
        n = len(files)
        a, b = int(n * args.train), int(n * (args.train + args.val))
        parts = {"train": files[:a], "val": files[a:b], "test": files[b:]}
        for split, fs in parts.items():
            out = config.DATA_DIR / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in fs:
                op(str(f), out / f.name)
            print(f"{cls:5s} {split:5s}: {len(fs)}")


if __name__ == "__main__":
    main()
