"""MoNuSeg XML nucleus polygons -> binary segmentation mask."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def xml_to_binary_mask(xml_file: str | Path, height: int, width: int) -> np.ndarray:
    """Rasterize each <Region> polygon, using its <Vertex X='' Y=''> nodes.

    Handles the standard Aperio-style MoNuSeg polygon XML convention. Regions
    with fewer than 3 vertices are ignored. Coordinate values are rounded to
    pixel centers. Returns uint8 0/255 binary mask.
    """
    root = ET.parse(str(xml_file)).getroot()
    mask = np.zeros((height, width), dtype=np.uint8)
    count = 0
    for region in root.findall('.//Region'):
        vertices = []
        for vert in region.findall('.//Vertex'):
            if "X" not in vert.attrib or "Y" not in vert.attrib:
                continue
            vertices.append((round(float(vert.attrib["X"])), round(float(vert.attrib["Y"]))))
        if len(vertices) >= 3:
            pts = np.array(vertices, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            count += 1
    if count == 0:
        raise ValueError(f"No valid polygon <Region> elements found in {xml_file}")
    return mask
