#!/usr/bin/env python3
"""Segment a single AFM scan and save a preview image.

    python examples/segment_example.py path/to/scan.spm checkpoints/sam2.1_hiera_tiny.pt
"""

from __future__ import annotations

import sys

from neo_toolkit.segmentation import SamSegmenter, segment_afm_file
from neo_toolkit.segmentation.visualization import save_preview


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <afm-scan-file> <sam2-checkpoint>")
        raise SystemExit(1)
    scan_path, checkpoint_path = sys.argv[1], sys.argv[2]

    sam = SamSegmenter(checkpoint_path, model_cfg="sam2.1/sam2.1_hiera_t.yaml")
    image, instances = segment_afm_file(scan_path, sam)
    print(f"found {len(instances)} instances")

    save_preview(image, instances, "preview.png")
    print("wrote preview.png")


if __name__ == "__main__":
    main()
