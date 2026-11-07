#!/usr/bin/env python3
"""Make tiny SYNTHETIC circle images and masks to check the code works.

These generated images are NOT real pathology images or scientific evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def generate_sample(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.empty((size, size, 3), dtype=np.uint8)
    base[:] = (244, 204, 218)  # synthetic pink background in RGB
    noise = rng.normal(0, 9, size=(size, size, 1))
    image = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(8):
        cy = int(rng.integers(14, size - 14))
        cx = int(rng.integers(14, size - 14))
        ry = int(rng.integers(6, 13))
        rx = int(rng.integers(6, 13))
        shade = (int(rng.integers(90, 140)), int(rng.integers(45, 80)), int(rng.integers(115, 160)))
        cv2.ellipse(image, (cx, cy), (rx, ry), 0, 0, 360, shade, thickness=-1)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, color=255, thickness=-1)
    return image, mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("data/demo"))
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--train-count", type=int, default=8)
    p.add_argument("--val-count", type=int, default=2)
    args = p.parse_args()
    if args.size < 48 or args.train_count < 1 or args.val_count < 1:
        p.error("size must be >=48; train-count and val-count must be >=1")
    rng = np.random.default_rng(42)
    for split, count in [("train", args.train_count), ("val", args.val_count)]:
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(count):
            im, mask = generate_sample(rng, args.size)
            filename = f"toy_{split}_{i:03d}.png"
            Image.fromarray(im).save(args.output / split / "images" / filename)
            Image.fromarray(mask).save(args.output / split / "masks" / filename)
            if split == "train" and i == 0:
                docs = Path("docs")
                docs.mkdir(exist_ok=True)
                side = np.concatenate((im, np.repeat(mask[:, :, None], 3, axis=2)), axis=1)
                Image.fromarray(side).save(docs / "synthetic_example.png")
    print(f"Created SYNTHETIC demo dataset in {args.output}; not actual medical data.")


if __name__ == "__main__":
    main()
