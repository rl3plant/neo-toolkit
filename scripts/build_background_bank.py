#!/usr/bin/env python3
"""Segment a set of real AFM scans into a background crop bank and save it.

    python scripts/build_background_bank.py checkpoints/sam2.1_hiera_tiny.pt \
        sam2.1/sam2.1_hiera_t.yaml out/background_bank.npz \
        path/to/scan1.png path/to/scan2.png ...

Saved as a single .npz (images, masks, sources arrays) so later steps don't
need to re-run SAM2 -- rebuild only when the source scans or segmentation
pipeline change.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neo_toolkit.ecc.backgrounds import build_background_bank, save_bank
from neo_toolkit.segmentation.sam_backend import SamSegmenter


def main() -> None:
    if len(sys.argv) < 5:
        print(f"usage: {sys.argv[0]} <sam2-checkpoint> <sam2-model-cfg> <out.npz> <scan1> [scan2 ...]")
        raise SystemExit(1)
    checkpoint_path, model_cfg, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    scan_paths = sys.argv[4:]

    sam = SamSegmenter(checkpoint_path, model_cfg=model_cfg, device="cpu")

    crops = []
    for path in scan_paths:
        t0 = time.time()
        image_crops = build_background_bank([path], sam)
        crops.extend(image_crops)
        print(f"{path}: {len(image_crops)} crops in {time.time() - t0:.1f}s", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_bank(crops, out_path)
    print(f"saved {len(crops)} crops from {len(scan_paths)} scans -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
