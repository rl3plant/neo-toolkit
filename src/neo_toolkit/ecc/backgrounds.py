"""Build a bank of real-AFM-origami background crops, split by source image.

Fixes a train/val background-leakage bug in
old_stuff/NEO_ECC/data_pipeline.py: there, ExperimentConfig.image_paths
defaulted to a single real scan, and even with more scans listed, a
background was drawn independently at random for every synthetic sample --
so the very same background crop could land on both sides of the train/val
split, and validation wasn't really testing generalization to unseen
origami. Here, crops are grouped by which source scan they came from, and
the split is done over those source images, not over individual crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from neo_toolkit.segmentation.pipeline import Instance, segment_afm_file
from neo_toolkit.segmentation.sam_backend import SamSegmenter


@dataclass
class Crop:
    image: np.ndarray  # (size, size, 3) float32 in [0, 1]
    mask: np.ndarray  # (size, size) {0, 1} float32, footprint of the origami within the crop
    source: str  # which scan this came from, for identity-based splitting


def build_background_bank(
    image_paths: list[str | Path],
    sam: SamSegmenter,
    *,
    out_size: int = 128,
    rect_expand: float = 1.25,
    segment_kwargs: dict | None = None,
) -> list[Crop]:
    """Segment each scan and warp every instance into a canonical square crop."""
    segment_kwargs = segment_kwargs or {}
    crops: list[Crop] = []
    for path in image_paths:
        path = Path(path)
        image, instances = segment_afm_file(path, sam, **segment_kwargs)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for instance in instances:
            crop = _warp_instance(image_bgr, instance, out_size=out_size, rect_expand=rect_expand)
            if crop is not None:
                crops.append(Crop(image=crop[0], mask=crop[1], source=path.name))
    return crops


def _warp_instance(image_bgr: np.ndarray, instance: Instance, *, out_size: int, rect_expand: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Rectify an instance's oriented footprint to an axis-aligned square crop."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(instance.contour.astype(np.float32))
    if w <= 0 or h <= 0:
        return None
    box = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
    long_side, short_side = max(w, h), min(w, h)
    if w < h:
        box = np.roll(box, 1, axis=0)  # keep the long side mapped to dst's horizontal edge
    dst_w = max(2, int(round(long_side * rect_expand)))
    dst_h = max(2, int(round(short_side * rect_expand)))
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(box, dst)
    warped_image = cv2.warpPerspective(image_bgr, transform, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
    mask_u8 = instance.mask.astype(np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask_u8, transform, (dst_w, dst_h), flags=cv2.INTER_NEAREST)
    if warped_image.size == 0:
        return None

    return _letterbox_image(warped_image, out_size), _letterbox_mask(warped_mask, out_size)


def _letterbox_image(image_bgr: np.ndarray, size: int) -> np.ndarray:
    canvas = _letterbox(image_bgr.astype(np.float32) / 255.0, size)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _letterbox_mask(mask_u8: np.ndarray, size: int) -> np.ndarray:
    canvas = _letterbox(mask_u8.astype(np.float32) / 255.0, size, interpolation=cv2.INTER_NEAREST)
    return (canvas[..., 0] > 0.5).astype(np.float32)


def _letterbox(image: np.ndarray, size: int, *, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    canvas = np.zeros((size, size, resized.shape[2]), dtype=np.float32)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def save_bank(crops: list[Crop], path: str | Path) -> None:
    np.savez_compressed(
        path,
        images=np.stack([c.image for c in crops]),
        masks=np.stack([c.mask for c in crops]),
        sources=np.array([c.source for c in crops]),
    )


def load_bank(path: str | Path) -> list[Crop]:
    data = np.load(path)
    return [Crop(image=img, mask=mask, source=str(source)) for img, mask, source in zip(data["images"], data["masks"], data["sources"])]


def split_bank_by_source(crops: list[Crop], *, val_fraction: float = 0.25, seed: int = 0) -> tuple[list[Crop], list[Crop]]:
    """Split by which source scan each crop came from, so no scan contributes to both sides."""
    sources = sorted({c.source for c in crops})
    if len(sources) < 2:
        raise ValueError("need crops from at least 2 source images to split without leakage")
    rng = np.random.default_rng(seed)
    shuffled = list(sources)
    rng.shuffle(shuffled)
    n_val = max(1, round(val_fraction * len(shuffled)))
    val_sources = set(shuffled[:n_val])
    train = [c for c in crops if c.source not in val_sources]
    val = [c for c in crops if c.source in val_sources]
    return train, val
