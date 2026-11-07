"""Image tiling utilities for preprocessing and overlap-aware inference."""
from __future__ import annotations

import numpy as np


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if min(length, tile_size, stride) <= 0:
        raise ValueError("length, tile_size and stride must be positive")
    if stride > tile_size:
        raise ValueError("stride cannot be greater than tile_size; gaps would remain")
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def iter_tiles(image: np.ndarray, tile_size: int, stride: int):
    """Yield (y, x, padded_tile) covering an HxW or HxWxC image."""
    h, w = image.shape[:2]
    for y in tile_starts(h, tile_size, stride):
        for x in tile_starts(w, tile_size, stride):
            piece = image[y:y + tile_size, x:x + tile_size]
            ph, pw = tile_size - piece.shape[0], tile_size - piece.shape[1]
            if ph or pw:
                pad_width = [(0, ph), (0, pw)] + ([(0, 0)] if image.ndim == 3 else [])
                piece = np.pad(piece, pad_width, mode="edge")
            yield y, x, piece
