"""Local, in-process SAM2 inference (replaces DNAO-Analysis-Tool's client/server split).

The original tool ran SAM2 behind a Flask server and talked to it over HTTP
per prompt point; here the model is loaded once in-process and called
directly, which is simpler to embed in another project and avoids the
network round-trip.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from neo_toolkit.segmentation.types import Candidate


class SamSegmenter:
    """Wraps a SAM2 image predictor: one image embedding, many point prompts."""

    def __init__(self, checkpoint_path: str | Path, model_cfg: str, *, device: str | None = None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        sam_model = self._build_model(build_sam2, model_cfg, checkpoint_path, device)
        self._predictor = SAM2ImagePredictor(sam_model)
        self._image_set = False

    @staticmethod
    def _build_model(build_sam2, model_cfg: str, checkpoint_path: Path, device: str):
        try:
            return build_sam2(config_file=model_cfg, ckpt_path=str(checkpoint_path), device=device)
        except Exception:
            # Some sam2 installs (e.g. cloned rather than pip-installed) don't
            # register their config package with Hydra's search path by default.
            import sam2
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            config_dir = str(Path(sam2.__file__).resolve().parent / "configs")
            GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
                return build_sam2(config_file=model_cfg, ckpt_path=str(checkpoint_path), device=device)

    def set_image(self, image_rgb_uint8: np.ndarray) -> None:
        """Compute and cache the image embedding for subsequent point prompts."""
        self._predictor.set_image(image_rgb_uint8)
        self._image_set = True

    def predict_point(self, point: tuple[int, int], *, multimask: bool = True) -> list[Candidate]:
        """Return one Candidate per mask SAM proposes for a single point prompt."""
        if not self._image_set:
            raise RuntimeError("call set_image() before predict_point()")

        coords = np.array([point], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=multimask,
        )

        candidates = []
        for mask, score in zip(masks, scores):
            binary_mask = mask > 0.5
            contour = largest_contour(binary_mask)
            if contour is None:
                continue
            candidates.append(
                Candidate(
                    point=point,
                    contour=contour,
                    mask=binary_mask,
                    area=float(binary_mask.sum()),
                    sam_score=float(score),
                )
            )
        return candidates

    def predict_points(self, points: list[tuple[int, int]], *, multimask: bool = True) -> list[list[Candidate]]:
        """Predict candidate masks for each point in turn. See predict_point()."""
        return [self.predict_point(point, multimask=multimask) for point in points]


def largest_contour(binary_mask: np.ndarray) -> np.ndarray | None:
    """Extract the largest outer contour of a binary mask, or None if empty."""
    mask_u8 = binary_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2)
