#!/usr/bin/env python3
"""Verify training through the trellis circuit reaches the same result as brute force.

    python scripts/rerun_d37_trellis.py out/d37_trellis.json out/background_bank.npz [more_bank.npz ...]

Not a repeat of rerun_d37.py's full 3-seed/400-epoch protocol -- that
question (what's the best achievable WER) is already answered, see
[[neo-ecc-repro-concerns]] / the README. This script asks a narrower
question: does *differentiating through the trellis instead of brute-force
enumeration* change what the detector learns? tests/ecc/test_trellis.py
already proves the two circuits compute identical losses and (on a tiny toy
set) identical training trajectories; this is the same check at full scale,
real data, the actual winning architecture -- one seed per (objective,
circuit) pair, since the trellis has no expected advantage to search for
(see trellis.py's docstring), only agreement to confirm.

supervised doesn't touch either circuit (it's a plain masked-BCE loss on the
detector's own logits), so it's run once, not once per circuit.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neo_toolkit.ecc.backgrounds import load_bank, split_bank_by_source
from neo_toolkit.ecc.channel import ZChannel
from neo_toolkit.ecc.code import NEO_20_10
from neo_toolkit.ecc.decoding import codebook_tensor
from neo_toolkit.ecc.detector import Detector
from neo_toolkit.ecc.metrics import oracle_metrics
from neo_toolkit.ecc.synthetic import SyntheticConfig, generate_dataset
from neo_toolkit.ecc.trellis import build_trellis
from neo_toolkit.ecc.training import TrainingConfig, train_detector

SEED = 0
CHANNEL_P = 0.2
NUM_TRAIN_MESSAGES = 256
NUM_VAL_MESSAGES = 128
GROUP_J_RANGE = (10, 20)
EPOCHS = 400
BATCH_SIZE = 32
LR = 0.00137  # same winning hyperparameters as rerun_d37.py -- see Detector's docstring

RUNS = [
    ("supervised", "brute_force"),
    ("neurosymbolic", "brute_force"),
    ("neurosymbolic", "trellis"),
    ("hybrid", "brute_force"),
    ("hybrid", "trellis"),
]


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <out_results.json> <bank1.npz> [bank2.npz ...]")
        raise SystemExit(1)
    out_path = Path(sys.argv[1])
    bank_paths = [Path(p) for p in sys.argv[2:]]

    bank = [crop for path in bank_paths for crop in load_bank(path)]
    train_crops, val_crops = split_bank_by_source(bank, val_fraction=0.3, seed=0)
    print(f"bank: {len(bank)} crops from {len({c.source for c in bank})} scans "
          f"-> {len(train_crops)} train crops / {len(val_crops)} val crops", flush=True)

    channel = ZChannel(p=CHANNEL_P)
    codebook = codebook_tensor(NEO_20_10)
    trellis = build_trellis(NEO_20_10)

    train_cfg = SyntheticConfig(num_messages=NUM_TRAIN_MESSAGES, channel=channel, group_j_min=GROUP_J_RANGE[0], group_j_max=GROUP_J_RANGE[1], augment=True, seed=1)
    val_cfg = SyntheticConfig(num_messages=NUM_VAL_MESSAGES, channel=channel, group_j_min=GROUP_J_RANGE[0], group_j_max=GROUP_J_RANGE[1], seed=2)
    train_dataset = generate_dataset(train_crops, NEO_20_10, train_cfg)
    val_dataset = generate_dataset(val_crops, NEO_20_10, val_cfg)

    val_corrupted_t = torch.as_tensor(val_dataset.corrupted_bits, dtype=torch.float32)
    val_clean_t = torch.as_tensor(val_dataset.clean_bits, dtype=torch.float32)
    val_targets_t = torch.as_tensor(val_dataset.message_indices, dtype=torch.long)
    val_mask_t = torch.as_tensor(val_dataset.group_mask, dtype=torch.float32)
    oracle = oracle_metrics(val_corrupted_t, val_clean_t, val_targets_t, val_mask_t, codebook, channel=channel)
    print(f"oracle (best possible): group_wer={oracle.group_wer:.4f} group_ber={oracle.group_ber:.4f}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, epochs={EPOCHS}, seed={SEED}", flush=True)

    results = {
        "bank_files": [str(p) for p in bank_paths],
        "channel_p": CHANNEL_P,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "oracle": vars(oracle),
        "runs": {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for objective, circuit in RUNS:
        key = f"{objective}/{circuit}"
        print(f"--- {key} ---", flush=True)
        torch.manual_seed(SEED)
        detector = Detector()
        cfg = TrainingConfig(
            objective=objective,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            device=device,
            circuit=circuit,
            trellis=trellis if circuit == "trellis" else None,
        )
        t0 = time.time()
        history = train_detector(detector, train_dataset, val_dataset, codebook, cfg)
        elapsed = time.time() - t0
        final = history.val_metrics[-1]
        best = history.val_metrics[history.best_epoch]
        print(
            f"{key}: final group_wer={final.group_wer:.4f} best group_wer={best.group_wer:.4f} "
            f"(epoch {history.best_epoch}) in {elapsed:.1f}s",
            flush=True,
        )
        results["runs"][key] = {
            "objective": objective,
            "circuit": circuit,
            "final": vars(final),
            "best_epoch": history.best_epoch,
            "best": vars(best),
            "train_seconds": elapsed,
        }
        out_path.write_text(json.dumps(results, indent=2))

    print("--- summary (best group_wer, oracle = %.4f) ---" % oracle.group_wer, flush=True)
    for key, run in results["runs"].items():
        print(f"  {key}: {run['best']['group_wer']:.4f}", flush=True)


if __name__ == "__main__":
    main()
