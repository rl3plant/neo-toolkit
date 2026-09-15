# neo-toolkit

AFM image segmentation and error-correction decoding for the NEO DNA-origami
data storage project.

- `neo_toolkit.segmentation` — SAM2-based instance segmentation of DNA
  origami in AFM scans.
- `neo_toolkit.ecc` — the (20,10) error-correcting code, channel simulation,
  detector model, and decoding.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For raw AFM scanner files (`.spm`/`.jpk`/`.ibw`/`.mi`):

```bash
pip install --no-deps pySPM tifffile igor2
```

(`--no-deps` because pySPM's own pinned versions are outdated and will
otherwise break the install.)

Get a SAM2 checkpoint:

```bash
python scripts/download_sam2_checkpoint.py tiny
```

## Usage

```python
from neo_toolkit.segmentation import SamSegmenter, segment_afm_file

sam = SamSegmenter("checkpoints/sam2.1_hiera_tiny.pt", model_cfg="sam2.1/sam2.1_hiera_t.yaml")
image, instances = segment_afm_file("scan.spm", sam)
```

See `examples/segment_example.py` for a runnable script and `scripts/` for
training/evaluation scripts.

## Tests

```bash
pytest
```
