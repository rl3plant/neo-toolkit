"""Render synthetic protein-position blobs onto real background crops.

Ported from old_stuff/NEO_ECC/data_pipeline.py's blob rendering
(_grid_centers_in_mask_bbox, _draw_gaussian_blob/_add_gaussian_to_field,
_render_single_rect, impose_grouped_blobs_on_rects, generate_data), unchanged
in the rendering math itself -- that part isn't in question. What's
different: backgrounds come from backgrounds.py's real-crop bank (built via
neo_toolkit.segmentation instead of an ad hoc SAM2 script), and the
train/val split happens once, by background source identity, before any
sample generation -- not by shuffling already-generated samples, which is
what let the same background land on both sides in the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neo_toolkit.ecc.backgrounds import Crop
from neo_toolkit.ecc.channel import Channel, ZChannel
from neo_toolkit.ecc.code import LinearCode
from neo_toolkit.ecc.detector import GRID_COLS, GRID_ROWS


@dataclass
class SyntheticConfig:
    num_messages: int = 256
    channel: Channel = field(default_factory=lambda: ZChannel(p=0.2))
    group_j_min: int = 10
    group_j_max: int = 20
    jitter_std: float = 1.0
    blob_alpha: float = 0.3  # blob sigma as a fraction of grid-cell size
    blob_peak: float = 0.5
    seed: int = 42
    augment: bool = False  # random flip + brightness jitter per background draw
    brightness_range: tuple[float, float] = (0.85, 1.15)


@dataclass
class Dataset:
    images: np.ndarray  # (num_groups, J_max, size, size, 3) float32
    clean_bits: np.ndarray  # (num_groups, 20) uint8
    corrupted_bits: np.ndarray  # (num_groups, J_max, 20) uint8 -- what was actually rendered
    message_indices: np.ndarray  # (num_groups,) int64
    group_sizes: np.ndarray  # (num_groups,) int64
    group_mask: np.ndarray  # (num_groups, J_max) float32, 1 for real observations


def _grid_centers_in_mask_bbox(mask: np.ndarray) -> list[tuple[float, float]]:
    """4x5 grid over the left 70% of the mask's bounding box (keeps blobs
    on/near the origami even when the crop keeps surrounding context)."""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        h, w = mask.shape
        x0, y0, x1, y1 = 0.0, 0.0, float(w), float(h)
    else:
        x0, y0 = float(xs.min()), float(ys.min())
        x1, y1 = float(xs.max() + 1), float(ys.max() + 1)
    x1 = x0 + 0.7 * (x1 - x0)
    cell_w, cell_h = max(1.0, x1 - x0) / GRID_COLS, max(1.0, y1 - y0) / GRID_ROWS
    return [(x0 + (c + 0.5) * cell_w, y0 + (r + 0.5) * cell_h) for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def _add_gaussian_blob(field_hw: np.ndarray, cx: float, cy: float, sigma: float, peak: float) -> None:
    h, w = field_hw.shape[:2]
    radius = int(np.ceil(3 * sigma))
    y0, y1 = max(0, int(cy) - radius), min(h, int(cy) + radius + 1)
    x0, x1 = max(0, int(cx) - radius), min(w, int(cx) + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    blob = peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    field_hw[y0:y1, x0:x1] += blob.astype(np.float32)


def _augment_crop(crop: Crop, rng: np.random.Generator, *, brightness_range: tuple[float, float]) -> Crop:
    """Random flip + brightness jitter -- cheap extra diversity from a limited
    real background bank (currently ~5 independent imaging batches, two of
    them tiny, however many source images/crops that adds up to)."""
    image, mask = crop.image, crop.mask
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[::-1, :], mask[::-1, :]
    brightness = rng.uniform(*brightness_range)
    image = np.clip(image * brightness, 0.0, 1.0).astype(np.float32)
    return Crop(image=np.ascontiguousarray(image), mask=np.ascontiguousarray(mask), source=crop.source)


def render_blobs(crop: Crop, bits: np.ndarray, cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """Overlay Gaussian blobs on `crop` at grid positions where `bits` is 1."""
    h, w = crop.mask.shape
    centers = _grid_centers_in_mask_bbox(crop.mask)
    ys, xs = np.where(crop.mask > 0.5)
    cell_w = (float(xs.max() + 1 - xs.min()) if len(xs) else w) / GRID_COLS
    cell_h = (float(ys.max() + 1 - ys.min()) if len(ys) else h) / GRID_ROWS
    sigma = cfg.blob_alpha * min(cell_w, cell_h)

    blob_field = np.zeros((h, w), dtype=np.float32)
    for index, bit in enumerate(bits):
        if not bit:
            continue
        cx, cy = centers[index]
        if cfg.jitter_std > 0:
            cx = cx + rng.normal(0.0, cfg.jitter_std)
            cy = cy + rng.normal(0.0, cfg.jitter_std)
        _add_gaussian_blob(blob_field, cx, cy, sigma=sigma, peak=1.0)

    return np.clip(crop.image + cfg.blob_peak * blob_field[:, :, None], 0.0, 1.0).astype(np.float32)


def generate_dataset(bank: list[Crop], code: LinearCode, cfg: SyntheticConfig) -> Dataset:
    """Sample distinct messages, render each as a group of J corrupted observations."""
    if not bank:
        raise ValueError("background bank is empty")
    rng = np.random.default_rng(cfg.seed)

    num_codewords = code.codebook.shape[0]
    if cfg.num_messages > num_codewords:
        raise ValueError(f"num_messages={cfg.num_messages} exceeds the code's {num_codewords} codewords")
    message_indices = rng.choice(num_codewords, size=cfg.num_messages, replace=False)

    size = bank[0].image.shape[0]
    j_max = cfg.group_j_max
    n_groups = len(message_indices)
    images = np.zeros((n_groups, j_max, size, size, 3), dtype=np.float32)
    clean_bits = np.zeros((n_groups, code.n), dtype=np.uint8)
    corrupted_bits = np.zeros((n_groups, j_max, code.n), dtype=np.uint8)
    group_sizes = np.zeros(n_groups, dtype=np.int64)
    group_mask = np.zeros((n_groups, j_max), dtype=np.float32)

    for g, message_index in enumerate(message_indices):
        clean = code.codebook[message_index]
        clean_bits[g] = clean
        j = int(rng.integers(cfg.group_j_min, cfg.group_j_max + 1))
        group_sizes[g] = j
        group_mask[g, :j] = 1.0
        for obs in range(j):
            observed = cfg.channel.corrupt(clean, rng)
            corrupted_bits[g, obs] = observed
            crop = bank[rng.integers(0, len(bank))]
            if cfg.augment:
                crop = _augment_crop(crop, rng, brightness_range=cfg.brightness_range)
            images[g, obs] = render_blobs(crop, observed, cfg, rng)
        for obs in range(j, j_max):
            corrupted_bits[g, obs] = clean  # padding slots, excluded via group_mask

    return Dataset(
        images=images,
        clean_bits=clean_bits,
        corrupted_bits=corrupted_bits,
        message_indices=message_indices,
        group_sizes=group_sizes,
        group_mask=group_mask,
    )
