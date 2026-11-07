#!/usr/bin/env python3
"""Prepare MoNuSeg original images/XML polygons as paired PNG patches.

IMPORTANT: splits happen by ORIGINAL IMAGE before patch extraction, avoiding
train/validation leakage between overlapping patches of the same source.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from histoseg.monuseg import xml_to_binary_mask
from histoseg.tiles import iter_tiles

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True, help="Folder with original MoNuSeg images")
    p.add_argument("--annotations", type=Path, required=True, help="Folder with matching XML annotations")
    p.add_argument("--output", type=Path, default=Path("data/monuseg"))
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.images.is_dir() or not args.annotations.is_dir():
        p.error("--images and --annotations must be directories")
    if not 0 < args.val_ratio < 1:
        p.error("--val-ratio must be strictly between 0 and 1")
    if args.tile_size <= 0 or not 0 < args.stride <= args.tile_size:
        p.error("tile-size > 0 and 0 < stride <= tile-size are required")

    source_images = sorted(x for x in args.images.iterdir() if x.suffix.lower() in EXTENSIONS)
    if len(source_images) < 2:
        p.error("Need at least two source images to form train and validation splits")
    for path in source_images:
        if not (args.annotations / f"{path.stem}.xml").exists():
            p.error(f"Missing XML annotation matching {path.name}: {path.stem}.xml")
    random.Random(args.seed).shuffle(source_images)
    n_val = min(len(source_images) - 1, max(1, round(len(source_images) * args.val_ratio)))
    split_sources = {"val": source_images[:n_val], "train": source_images[n_val:]}

    for split, paths in split_sources.items():
        image_output = args.output / split / "images"
        mask_output = args.output / split / "masks"
        image_output.mkdir(parents=True, exist_ok=True)
        mask_output.mkdir(parents=True, exist_ok=True)
        n_patches = 0
        for path in paths:
            with Image.open(path) as pil:
                rgb = np.asarray(pil.convert("RGB"))
            h, w = rgb.shape[:2]
            mask = xml_to_binary_mask(args.annotations / f"{path.stem}.xml", h, w)
            images = {(y, x): tile for y, x, tile in iter_tiles(rgb, args.tile_size, args.stride)}
            masks = {(y, x): tile for y, x, tile in iter_tiles(mask, args.tile_size, args.stride)}
            assert images.keys() == masks.keys()
            for (y, x), tile in images.items():
                filename = f"{path.stem}_y{y:05d}_x{x:05d}.png"
                Image.fromarray(tile).save(image_output / filename)
                Image.fromarray(masks[(y, x)]).save(mask_output / filename)
                n_patches += 1
        print(f"{split}: {len(paths)} original images -> {n_patches} paired patches")
    manifest = {
        "note": "Original images split before patch extraction to prevent leakage; challenge official test set not used.",
        "tile_size": args.tile_size,
        "stride": args.stride,
        "seed": args.seed,
        "original_images": {key: [p.name for p in paths] for key, paths in split_sources.items()},
    }
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Ready for training: {args.output}")


if __name__ == "__main__":
    main()
