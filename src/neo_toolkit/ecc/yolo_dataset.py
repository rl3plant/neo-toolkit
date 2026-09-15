"""Load neo_project's human-annotated YOLO-segmentation dataset into Crops.

neo_project/Data/<category>/{Images,Contours_YOLO}/ holds real AFM scans with
per-instance polygon labels in YOLOv8-seg format (class_id, normalized x/y
polygon points), class 0 = origami, class 1 = bound protein where annotated.
These are human-verified masks, so for background-crop purposes they're used
directly instead of running SAM2 -- strictly better than an automatic
proposal, and doesn't need a GPU.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from neo_toolkit.ecc.backgrounds import Crop, _warp_instance
from neo_toolkit.segmentation.types import Candidate
from neo_toolkit.yolo_labels import parse_yolo_seg_file

__all__ = ["load_categories_as_crops", "load_category_as_crops", "parse_yolo_seg_file"]


def load_category_as_crops(
    category_dir: str | Path,
    *,
    origami_class: int = 0,
    out_size: int = 128,
    rect_expand: float = 1.25,
    min_area: float = 25.0,
) -> list[Crop]:
    """One Crop per human-labeled origami polygon in a category's images."""
    category_dir = Path(category_dir)
    images_dir, labels_dir = category_dir / "Images", category_dir / "Contours_YOLO"
    crops: list[Crop] = []
    for image_path in sorted(images_dir.glob("*.png")):
        label_path = labels_dir / f"{image_path.stem}_yolo.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        for cls, polygon in parse_yolo_seg_file(label_path, w, h):
            if cls != origami_class or len(polygon) < 3:
                continue
            area = cv2.contourArea(polygon)
            if area < min_area:
                continue
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_u8, [polygon], 1)
            candidate = Candidate(point=(0, 0), contour=polygon, mask=mask_u8.astype(bool), area=float(area), sam_score=1.0)

            warped = _warp_instance(image, candidate, out_size=out_size, rect_expand=rect_expand)
            if warped is not None:
                crops.append(Crop(image=warped[0], mask=warped[1], source=f"{category_dir.name}/{image_path.name}"))
    return crops


def load_categories_as_crops(category_dirs: list[str | Path], **kwargs) -> list[Crop]:
    crops: list[Crop] = []
    for category_dir in category_dirs:
        crops.extend(load_category_as_crops(category_dir, **kwargs))
    return crops
