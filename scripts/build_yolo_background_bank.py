#!/usr/bin/env python3
"""Build a background crop bank from neo_project's human-annotated dataset.

Uses the ground-truth origami polygons directly (no SAM2 needed for this
one) -- see yolo_dataset.py for why that's fine for building *training*
background diversity specifically, while the actual pipeline still segments
real scans with SAM2 at inference time (scripts/end_to_end_demo.py).

    python scripts/build_yolo_background_bank.py out/yolo_background_bank.npz \
        "../neo_project/Data/osif shape rectangles" \
        "../neo_project/Data/final design rectangle Monometric Strepttavidin" \
        "../neo_project/Data/final design - rectangle + streptavidin" \
        "../neo_project/Data/rectangle+IgG-mica-dry-Bruker"

Only pass "rectangle"-shaped categories -- the (20, 10) code assumes a 4x5
grid over a rectangular origami; other shapes (bow-tie, triangle, z-shape,
seeman tile) don't match that layout.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neo_toolkit.ecc.backgrounds import save_bank
from neo_toolkit.ecc.yolo_dataset import load_categories_as_crops


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <out.npz> <category_dir> [category_dir ...]")
        raise SystemExit(1)
    out_path = Path(sys.argv[1])
    category_dirs = sys.argv[2:]

    t0 = time.time()
    crops = load_categories_as_crops(category_dirs)
    sources = sorted({c.source for c in crops})
    print(f"{len(crops)} crops from {len(sources)} images across {len(category_dirs)} categories in {time.time() - t0:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_bank(crops, out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
