#!/usr/bin/env python3
"""End-to-end demo: raw AFM scan -> SAM2 segmentation -> per-instance decode.

Proves the deployed pipeline chain actually connects: neo_toolkit.segmentation
(SAM2) finds origami instances in a raw scan, each is warped to a canonical
crop the same way training crops were built (backgrounds._warp_instance), the
trained Detector reads off per-cell logits, and neo_toolkit.ecc.decoding
exactly decodes each instance's most likely codeword. This is the real
inference-time pipeline -- unlike training, which can use human-annotated
masks for background diversity (see yolo_dataset.py), there is no
ground truth here, only the raw scan and SAM2.

    python scripts/end_to_end_demo.py checkpoints/sam2.1_hiera_tiny.pt \
        sam2.1/sam2.1_hiera_t.yaml out/best_detector.pt path/to/scan.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neo_toolkit.ecc.backgrounds import _warp_instance
from neo_toolkit.ecc.code import NEO_20_10
from neo_toolkit.ecc.decoding import codebook_tensor, decode_single
from neo_toolkit.ecc.detector import Detector
from neo_toolkit.segmentation.pipeline import segment_afm_file
from neo_toolkit.segmentation.sam_backend import SamSegmenter


def main() -> None:
    if len(sys.argv) != 5:
        print(f"usage: {sys.argv[0]} <sam2-checkpoint> <sam2-model-cfg> <detector-checkpoint> <scan-file>")
        raise SystemExit(1)
    sam_checkpoint, model_cfg, detector_checkpoint, scan_path = sys.argv[1:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = SamSegmenter(sam_checkpoint, model_cfg=model_cfg, device=device)
    print(f"segmenting {scan_path} with SAM2 ({device})...", flush=True)
    image, instances = segment_afm_file(scan_path, sam)
    print(f"found {len(instances)} origami instances", flush=True)

    checkpoint = torch.load(detector_checkpoint, map_location="cpu")
    detector = Detector()
    detector.load_state_dict(checkpoint["state_dict"])
    detector.eval()
    print(f"loaded detector (objective={checkpoint['objective']}, val group_wer={checkpoint['group_wer']:.4f})", flush=True)

    codebook = codebook_tensor(NEO_20_10)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    decoded = []
    with torch.no_grad():
        for instance in instances:
            warped = _warp_instance(image_bgr, instance, out_size=128, rect_expand=1.25)
            if warped is None:
                continue
            crop_image, _ = warped
            x = torch.as_tensor(crop_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            logits = detector(x)
            decoded_index = int(decode_single(logits, codebook).item())
            decoded.append((instance.bbox, decoded_index))

    print(f"decoded {len(decoded)} instances (single observation each, no grouping):", flush=True)
    for bbox, decoded_index in decoded[:20]:
        rounded_bbox = tuple(round(v) for v in bbox)
        print(f"  bbox={rounded_bbox} -> codeword #{decoded_index}", flush=True)
    if len(decoded) > 20:
        print(f"  ... and {len(decoded) - 20} more", flush=True)


if __name__ == "__main__":
    main()
