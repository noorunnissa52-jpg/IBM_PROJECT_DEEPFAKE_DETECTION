"""Generate a tiny SYNTHETIC dataset so you can verify the whole pipeline runs
without downloading anything.

"real" = smooth random scenes with fine sensor-like noise
"fake" = same scenes passed through a low-res -> upsample step (a crude stand-in for the
         up-sampling artifacts that generators leave behind).

This is ONLY a smoke test. For meaningful results, use a real dataset (see README).
"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import config


def scene(rng, size=256):
    base = rng.random((8, 8, 3))
    img = Image.fromarray((base * 255).astype("uint8")).resize((size, size), Image.BICUBIC)
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(3, 8)):
        x0, y0 = rng.integers(0, size - 40, 2)
        w, h = rng.integers(20, 90, 2)
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x0 + w, y0 + h], fill=col)
    return img


def make_real(rng):
    a = np.asarray(scene(rng), dtype=np.float32)
    a += rng.normal(0, 6, a.shape)                      # fine-grained noise
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def make_fake(rng):
    img = scene(rng)
    img = img.resize((64, 64), Image.BILINEAR).resize((256, 256), Image.NEAREST)  # blocky upsample
    a = np.asarray(img, dtype=np.float32)
    a[::4, ::4] += 8                                    # periodic grid artifact
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=200, help="images per class")
    p.add_argument("--n_val", type=int, default=40)
    p.add_argument("--n_test", type=int, default=40)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    for split, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        for cls, fn in (("real", make_real), ("fake", make_fake)):
            out = config.DATA_DIR / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                fn(rng).save(out / f"{cls}_{i:04d}.png")
    print("Demo dataset written to", config.DATA_DIR)


if __name__ == "__main__":
    main()
