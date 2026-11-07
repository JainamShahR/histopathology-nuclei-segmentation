#!/usr/bin/env python3
"""Predict a binary mask for a single image using tiled U-Net inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from histoseg.model import UNet
from histoseg.tiles import iter_tiles
from train import pick_device


@torch.no_grad()
def tiled_probability(model: UNet, image: np.ndarray, tile_size: int, overlap: int, device: torch.device) -> np.ndarray:
    """Average overlapping per-tile probabilities; output is HxW float32."""
    if tile_size < 16 or not 0 <= overlap < tile_size:
        raise ValueError("tile-size must be >=16 and overlap must satisfy 0 <= overlap < tile-size")
    h, w = image.shape[:2]
    sum_prob = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    for y, x, tile in iter_tiles(image, tile_size, tile_size - overlap):
        tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float()[None].to(device) / 255.0
        p = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        used_h, used_w = min(h-y, tile_size), min(w-x, tile_size)
        sum_prob[y:y + used_h, x:x + used_w] += p[:used_h, :used_w]
        counts[y:y + used_h, x:x + used_w] += 1.0
    assert counts.min() > 0, "internal tiling error: uncovered pixels"
    return sum_prob / counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/demo/best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/demo/prediction"))
    p.add_argument("--tile-size", type=int, default=128)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    if not 0 < args.threshold < 1:
        p.error("threshold must be between 0 and 1")
    device = pick_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = UNet(base_channels=checkpoint["base_channels"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    with Image.open(args.image) as img:
        rgb = np.array(img.convert("RGB"))
    probability = tiled_probability(model, rgb, args.tile_size, args.overlap, device)
    mask = (probability >= args.threshold).astype(np.uint8) * 255
    overlay = rgb.astype(np.float32)
    overlay[mask > 0] = 0.55 * overlay[mask > 0] + 0.45 * np.array([5, 225, 90])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.output_dir / "predicted_mask.png")
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(args.output_dir / "overlay.png")
    print(f"Saved predicted_mask.png and overlay.png in {args.output_dir}")
    if "demo" in args.checkpoint.as_posix().lower():
        print("CAUTION: This is a synthetic-demo model prediction, NOT a validated nuclei mask.")


if __name__ == "__main__":
    main()
