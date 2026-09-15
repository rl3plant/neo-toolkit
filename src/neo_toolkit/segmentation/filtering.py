"""Reduce SAM's raw per-point candidate masks down to one mask per origami.

Ported from DNAO-Analysis-Tool's filtering.py: pick the best candidate per
prompt point, drop masks whose area is far from the batch median, then
run non-max suppression on what's left. Rewritten to use a plain numpy NMS
instead of torchvision, so this module has no torch dependency.
"""

from __future__ import annotations

import numpy as np

from neo_toolkit.segmentation.classifier import MaskClassifier
from neo_toolkit.segmentation.features import mask_features
from neo_toolkit.segmentation.types import Candidate


def best_per_point(
    point_candidates: list[list[Candidate]],
    *,
    classifier: MaskClassifier | None = None,
    classifier_weight: float = 0.9,
) -> list[Candidate]:
    """Keep the single best-scoring candidate mask at each prompt point."""
    classifier_weight = max(0.0, min(1.0, classifier_weight))
    best: list[Candidate] = []
    for candidates in point_candidates:
        if not candidates:
            continue
        scored = []
        for candidate in candidates:
            score = (1.0 - classifier_weight) * candidate.sam_score
            if classifier is not None and classifier.is_trained:
                candidate.classifier_score = classifier.score(mask_features(candidate))
                score = classifier_weight * candidate.classifier_score + (1.0 - classifier_weight) * candidate.sam_score
            candidate.final_score = score
            scored.append((score, candidate))
        best.append(max(scored, key=lambda item: item[0])[1])
    return best


def filter_by_area(candidates: list[Candidate], *, min_ratio: float = 0.5, max_ratio: float = 2.0) -> list[Candidate]:
    """Drop candidates whose area is far outside [min_ratio, max_ratio] x median."""
    if not candidates:
        return []
    median_area = float(np.median([c.area for c in candidates]))
    if median_area == 0:
        return candidates
    return [c for c in candidates if min_ratio * median_area <= c.area <= max_ratio * median_area]


def non_max_suppression(candidates: list[Candidate], *, iou_threshold: float = 0.2) -> list[Candidate]:
    """Greedily keep the highest-scoring candidate in each cluster of overlapping masks.

    Uses exact pixel mask IoU, not a box approximation: origami sit at
    arbitrary angles, so an axis-aligned (or even oriented) box overstates a
    diagonal instance's footprint and would distort which candidates count
    as "the same object". The axis-aligned bbox is still used as a cheap
    broad-phase reject before paying for the full mask comparison.
    """
    if not candidates:
        return []
    order = sorted(range(len(candidates)), key=lambda i: candidates[i].final_score, reverse=True)
    kept: list[int] = []
    for idx in order:
        candidate = candidates[idx]
        suppressed = False
        for k in kept:
            other = candidates[k]
            if _box_iou(candidate.bbox, other.bbox) <= 0.0:
                continue  # boxes don't even touch, so the masks can't overlap either
            if _mask_iou(candidate.mask, other.mask) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(idx)
    return [candidates[i] for i in kept]


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.count_nonzero(a & b)
    if intersection == 0:
        return 0.0
    union = np.count_nonzero(a | b)
    return intersection / union if union > 0 else 0.0


def filter_candidates(
    point_candidates: list[list[Candidate]],
    *,
    classifier: MaskClassifier | None = None,
    classifier_weight: float = 0.9,
    area_ratio: tuple[float, float] = (0.62, 1.49),
    nms_iou: float = 0.065,
) -> list[Candidate]:
    """Full filtering pipeline: best-per-point -> area filter -> NMS.

    area_ratio/nms_iou defaults are Bayesian-optimized against real ground
    truth, see foreground_component_points()'s docstring.
    """
    candidates = best_per_point(point_candidates, classifier=classifier, classifier_weight=classifier_weight)
    candidates = filter_by_area(candidates, min_ratio=area_ratio[0], max_ratio=area_ratio[1])
    return non_max_suppression(candidates, iou_threshold=nms_iou)
