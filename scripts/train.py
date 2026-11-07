#!/usr/bin/env python3
"""Train a small U-Net for binary nuclei segmentation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from histoseg.data import PairedImageDataset
from histoseg.metrics import batch_dice_iou, segmentation_loss
from histoseg.model import UNet


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/demo"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--base-channels", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16 or args.base_channels < 1:
        p.error("epochs, batch-size, base-channels must be positive; image-size must be >=16")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    train_ds = PairedImageDataset(args.data, "train", args.image_size)
    val_ds = PairedImageDataset(args.data, "val", args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    history = []
    print(f"Device={device}, train_patches={len(train_ds)}, val_patches={len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        model.eval()
        dice_sum, iou_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                logits = model(images.to(device))
                dice, iou = batch_dice_iou(logits, masks.to(device))
                dice_sum += dice.sum().item()
                iou_sum += iou.sum().item()
                n += images.size(0)
        record = {"epoch": epoch, "train_loss": round(total_loss / len(train_ds), 5),
                  "val_dice": round(dice_sum / n, 5), "val_iou": round(iou_sum / n, 5)}
        history.append(record)
        print(record)
        if record["val_dice"] > best_dice:
            best_dice = record["val_dice"]
            # Simple state_dict checkpoint, no large dataset bundled.
            torch.save({"model_state": model.state_dict(), "base_channels": args.base_channels,
                        "image_size": args.image_size, "epoch": epoch,
                        "val_dice": record["val_dice"], "val_iou": record["val_iou"]},
                       args.out_dir / "best.pt")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint and training history in {args.out_dir}")
    if "demo" in args.data.as_posix().lower():
        print("CAUTION: Results from synthetic demo images are NOT medical validation metrics.")


if __name__ == "__main__":
    main()
