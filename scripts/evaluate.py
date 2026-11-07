#!/usr/bin/env python3
"""Compute binary Dice/IoU from a saved checkpoint on paired validation patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from histoseg.data import PairedImageDataset
from histoseg.metrics import batch_dice_iou
from histoseg.model import UNet
from train import pick_device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/demo"))
    p.add_argument("--checkpoint", type=Path, default=Path("runs/demo/best.pt"))
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--output", type=Path, default=Path("runs/demo/eval.json"))
    args = p.parse_args()

    device = pick_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = UNet(base_channels=checkpoint["base_channels"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    ds = PairedImageDataset(args.data, args.split, checkpoint["image_size"])
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    total_dice, total_iou, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            dice, iou = batch_dice_iou(logits, masks.to(device))
            total_dice += dice.sum().item()
            total_iou += iou.sum().item()
            n += images.size(0)
    results = {"split": args.split, "n_patches": n, "dice": total_dice/n, "iou": total_iou/n,
               "note": "Binary segmentation metrics. Synthetic demo values are not clinical or MoNuSeg results."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
