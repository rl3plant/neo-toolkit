"""Generate SAM point prompts over the foreground of a preprocessed AFM image.

Ported from DNAO-Analysis-Tool/src/client/pipeline/instance_segmentation/prompting.py,
with the Session-based parameter overrides replaced by plain keyword arguments.
"""

from __future__ import annotations

import cv2
import numpy as np


def foreground_mask(image_uint8: np.ndarray, *, morph_kernel: int = 3, morph_op: str = "open", morph_iters: int = 1) -> np.ndarray:
    """Otsu-threshold the image and clean it up with a morphological operation."""
    _, thresh = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if morph_kernel < 1:
        morph_kernel = 1
    if morph_kernel % 2 == 0:
        morph_kernel += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    op = cv2.MORPH_OPEN if morph_op == "open" else cv2.MORPH_CLOSE
    return cv2.morphologyEx(thresh, op, kernel, iterations=morph_iters)


def _otsu_area_cutoff(areas: np.ndarray, *, min_separation_ratio: float = 4.0) -> float:
    """Split component areas into "noise speck" vs "real" via Otsu, in log-area
    space (areas span orders of magnitude, e.g. 1px specks vs 1000px origami).

    Same trick foreground_mask() uses on pixel intensities, applied instead
    to the histogram of component areas -- so the noise-vs-real cutoff is
    derived from the image itself instead of a fixed pixel constant, which
    would need re-tuning for every scan resolution / pixel size.

    Otsu always bisects a histogram into two groups, even a unimodal one
    (e.g. once morphological opening has already removed all the noise
    specks) -- so the split is only trusted when the two sides' typical
    areas actually differ by min_separation_ratio; otherwise there's no
    real evidence of a noise population and every component is kept.
    """
    if len(areas) < 2:
        return 0.0
    log_areas = np.log1p(areas.astype(np.float64))
    lo, hi = log_areas.min(), log_areas.max()
    if hi <= lo:
        return 0.0
    scaled = ((log_areas - lo) / (hi - lo) * 255).astype(np.uint8)
    otsu_level, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = float(np.expm1(lo + (otsu_level / 255.0) * (hi - lo)))

    below, above = areas[areas < cutoff], areas[areas >= cutoff]
    if len(below) == 0 or len(above) == 0 or np.median(above) < min_separation_ratio * np.median(below):
        return 0.0
    return cutoff


def foreground_component_points(
    image_uint8: np.ndarray,
    *,
    morph_kernel: int = 5,
    morph_op: str = "open",
    morph_iters: int = 1,
    min_component_area: float | None = None,
    merge_area_ratio: float = 1.37,
    dense_grid: tuple[int, int] = (5, 5),
) -> list[tuple[int, int]]:
    """One prompt per connected foreground component: its centroid if the
    component looks like a single instance, or a small grid restricted to
    its footprint if it looks like several touching instances merged by
    thresholding (area much bigger than the batch's typical component).

    A fixed dense grid (see foreground_grid_points) sends a prompt to every
    cell that happens to land on foreground, which under-samples small,
    well-separated instances between grid lines and over-samples large ones
    -- this instead guarantees every component gets a proposal, and spends
    the extra prompts only where components actually look merged.

    min_component_area: below this, a component is discarded as a noise
    speck rather than prompted. Left as None, it's derived per-image via
    _otsu_area_cutoff() -- pass an explicit value if you know your scan's
    typical noise floor and want it fixed across images.

    Defaults (morph_kernel, merge_area_ratio, dense_grid) are Bayesian-
    optimized (scripts/tune_segmentation.py, Optuna TPE, 25 trials) against
    real human-annotated ground truth (mean F1 0.87) -- see
    out/segmentation_tuning.json for the full sweep.
    """
    thresh = foreground_mask(image_uint8, morph_kernel=morph_kernel, morph_op=morph_op, morph_iters=morph_iters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels <= 1:
        return []
    all_areas = stats[1:, cv2.CC_STAT_AREA]
    if min_component_area is None:
        min_component_area = _otsu_area_cutoff(all_areas)
    component_ids = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_component_area]
    if not component_ids:
        return []

    typical_area = float(np.median([stats[i, cv2.CC_STAT_AREA] for i in component_ids]))
    points: list[tuple[int, int]] = []
    for i in component_ids:
        area = stats[i, cv2.CC_STAT_AREA]
        if typical_area <= 0 or area <= merge_area_ratio * typical_area:
            cx, cy = centroids[i]
            points.append((int(round(cx)), int(round(cy))))
            continue

        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        nx, ny = dense_grid
        for gx in range(nx):
            for gy in range(ny):
                px = x + int((gx + 0.5) * w / nx)
                py = y + int((gy + 0.5) * h / ny)
                if labels[py, px] == i:
                    points.append((px, py))
    return points


def foreground_grid_points(
    image_uint8: np.ndarray,
    *,
    grid: tuple[int, int] = (64, 64),
    morph_kernel: int = 3,
    morph_op: str = "open",
    morph_iters: int = 1,
) -> list[tuple[int, int]]:
    """Lay a grid over the image and keep the points that fall on foreground."""
    thresh = foreground_mask(image_uint8, morph_kernel=morph_kernel, morph_op=morph_op, morph_iters=morph_iters)
    height, width = image_uint8.shape[:2]
    num_cells_x, num_cells_y = grid
    cell_w = max(1, width // num_cells_x)
    cell_h = max(1, height // num_cells_y)

    points = []
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            x, y = i * cell_w, j * cell_h
            if thresh[y, x]:
                points.append((x, y))
    return points
