#!/usr/bin/env python3
"""Fetch a SAM2 checkpoint into ./checkpoints (kept out of git, see .gitignore)."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# name -> (download url, hydra model_cfg name for SamSegmenter)
CHECKPOINTS = {
    "tiny": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sam2.1/sam2.1_hiera_t.yaml",
    ),
    "small": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sam2.1/sam2.1_hiera_s.yaml",
    ),
    "base_plus": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2.1/sam2.1_hiera_b+.yaml",
    ),
    "large": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sam2.1/sam2.1_hiera_l.yaml",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=CHECKPOINTS, default="tiny", nargs="?")
    parser.add_argument("--out-dir", default="checkpoints")
    args = parser.parse_args()

    url, model_cfg = CHECKPOINTS[args.size]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(url).name

    print(f"Downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print(f"Done. Use SamSegmenter({out_path!r}, {model_cfg!r})")


if __name__ == "__main__":
    main()
