"""Shared parser for YOLOv8-segmentation-format ground-truth label files.

Used by both neo_toolkit.ecc.yolo_dataset (background crops for synthetic
training data) and neo_toolkit.segmentation.evaluation (scoring the
segmentation pipeline against human-verified masks) -- kept here, outside
both packages, so neither has to depend on the other for it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_yolo_seg_file(path: str | Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    """Parse `class_id x1 y1 x2 y2 ...` lines (normalized) into pixel-space polygons."""
    instances = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        instances.append((cls, coords.astype(np.int32)))
    return instances
