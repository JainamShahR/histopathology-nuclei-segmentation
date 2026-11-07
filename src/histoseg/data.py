"""Load image/binary-mask pairs from split folders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class PairedImageDataset(Dataset):
    """Expected layout: root/<split>/images/X.png and masks/X.png.

    Mask values >0 represent nuclei. Images and masks are resized for training
    with different interpolation to keep labels binary.
    """

    def __init__(self, root: str | Path, split: str = "train", image_size: int = 128) -> None:
        self.root = Path(root) / split
        self.image_size = image_size
        image_dir = self.root / "images"
        mask_dir = self.root / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Expected '{image_dir}' and '{mask_dir}'. See README.")
        self.items = []
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            self.items.append((image_path, mask_path))
        if not self.items:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.items[index]
        with Image.open(image_path) as im, Image.open(mask_path) as ma:
            image = im.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            mask = ma.convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            x = np.array(image, dtype=np.float32) / 255.0
            y = (np.array(mask, dtype=np.uint8) > 0).astype(np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(y[None])
