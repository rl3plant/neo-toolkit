"""Score neo_toolkit.segmentation against real, human-verified ground truth.

Ground truth comes from neo_project's YOLOv8-seg-format annotations (see
yolo_labels.py) -- class 0 polygons are origami instances, matching what
segment_image() tries to find. Used both to report real precision/recall/F1
(not just eyeballing a preview image) and as the objective for
scripts/tune_segmentation.py's hyperparameter search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from neo_toolkit.segmentation.pipeline import Instance
from neo_toolkit.yolo_labels import parse_yolo_seg_file


@dataclass
class ImageGroundTruth:
    image_path: Path
    masks: list[np.ndarray] = field(repr=False)  # one boolean HxW array per ground-truth origami instance


def load_ground_truth(category_dir: str | Path, *, origami_class: int = 0) -> list[ImageGroundTruth]:
    category_dir = Path(category_dir)
    images_dir, labels_dir = category_dir / "Images", category_dir / "Contours_YOLO"
    items: list[ImageGroundTruth] = []
    for image_path in sorted(images_dir.glob("*.png")):
        label_path = labels_dir / f"{image_path.stem}_yolo.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        h, w = image.shape[:2]
        masks = []
        for cls, polygon in parse_yolo_seg_file(label_path, w, h):
            if cls != origami_class or len(polygon) < 3:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            masks.append(mask.astype(bool))
        items.append(ImageGroundTruth(image_path=image_path, masks=masks))
    return items


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.count_nonzero(a & b)
    if intersection == 0:
        return 0.0
    union = np.count_nonzero(a | b)
    return intersection / union if union else 0.0


@dataclass
class MatchResult:
    precision: float
    recall: float
    f1: float
    mean_iou: float  # over matched pairs only
    n_predicted: int
    n_ground_truth: int
    n_matched: int
    matched_ious: list[float] = field(default_factory=list, repr=False)


def match_instances(predicted: list[Instance], ground_truth_masks: list[np.ndarray], *, iou_threshold: float = 0.5) -> MatchResult:
    """Hungarian-match predicted instances to ground-truth masks by IoU."""
    n_pred, n_gt = len(predicted), len(ground_truth_masks)
    if n_pred == 0 or n_gt == 0:
        both_empty = n_pred == 0 and n_gt == 0
        return MatchResult(
            precision=1.0 if both_empty else 0.0,
            recall=1.0 if both_empty else 0.0,
            f1=1.0 if both_empty else 0.0,
            mean_iou=0.0,
            n_predicted=n_pred,
            n_ground_truth=n_gt,
            n_matched=0,
        )

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, candidate in enumerate(predicted):
        for j, gt_mask in enumerate(ground_truth_masks):
            iou_matrix[i, j] = mask_iou(candidate.mask, gt_mask)

    row_idx, col_idx = linear_sum_assignment(-iou_matrix)  # maximize total IoU
    matched_ious = [float(iou_matrix[r, c]) for r, c in zip(row_idx, col_idx) if iou_matrix[r, c] >= iou_threshold]
    n_matched = len(matched_ious)

    precision = n_matched / n_pred
    recall = n_matched / n_gt
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return MatchResult(precision=precision, recall=recall, f1=f1, mean_iou=mean_iou, n_predicted=n_pred, n_ground_truth=n_gt, n_matched=n_matched, matched_ious=matched_ious)


def aggregate(results: list[MatchResult]) -> MatchResult:
    """Micro-average (pool matches/predictions/ground-truth counts across images)."""
    n_pred = sum(r.n_predicted for r in results)
    n_gt = sum(r.n_ground_truth for r in results)
    n_matched = sum(r.n_matched for r in results)
    matched_ious = [iou for r in results for iou in r.matched_ious]
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return MatchResult(precision=precision, recall=recall, f1=f1, mean_iou=mean_iou, n_predicted=n_pred, n_ground_truth=n_gt, n_matched=n_matched, matched_ious=matched_ious)


def evaluate_on_dataset(items: list[ImageGroundTruth], sam, *, iou_threshold: float = 0.5, **segment_kwargs) -> MatchResult:
    """Run segment_image() over every image and score against its ground truth."""
    from neo_toolkit.segmentation.pipeline import segment_image

    per_image = []
    for item in items:
        image = cv2.imread(str(item.image_path), cv2.IMREAD_GRAYSCALE)
        predicted = segment_image(image, sam, **segment_kwargs)
        per_image.append(match_instances(predicted, item.masks, iou_threshold=iou_threshold))
    return aggregate(per_image)
