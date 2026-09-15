#!/usr/bin/env python3
"""Rerun the D3.7 experiment with the background-leakage fix in place.

    python scripts/rerun_d37.py out/d37_rerun.json out/background_bank.npz [more_bank.npz ...]

Differences from old_stuff/NEO_ECC/main.py's version: backgrounds are held
out by source scan identity across train/val (see backgrounds.py), from all
12 real scans instead of 1 (plus, optionally, more banks -- e.g. crops built
from neo_project's human-annotated real data via yolo_dataset.py -- merged
in for more training diversity; this only changes what real crops the
*synthetic training data* is rendered onto, the deployed pipeline still
segments raw scans with neo_toolkit.segmentation/SAM2 at inference time,
see scripts/end_to_end_demo.py), results are reported next to the oracle
metrics (the best any detector could do, decoding straight from the true
corrupted bits) so a WER/BER number can be read in context instead of taken
at face value, and each objective is trained under multiple seeds (same
train/val data, different model init + minibatch order) so the reported
numbers carry a mean +/- std instead of a single noisy run.
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
from neo_toolkit.ecc.training import TrainingConfig, train_detector

OBJECTIVES = ["supervised", "neurosymbolic", "hybrid"]
SEEDS = [0, 1, 2]
CHANNEL_P = 0.2  # matches old_stuff/NEO_ECC/report_plan's "Omer suggests p_dropout=0.2"
NUM_TRAIN_MESSAGES = 256
NUM_VAL_MESSAGES = 128
GROUP_J_RANGE = (10, 20)  # matches the original's default repetition range
EPOCHS = 400
BATCH_SIZE = 32
LR = 0.00137  # from scripts/tune_detector_architecture.py's search, see Detector's docstring


def _mean_std(values: list[float]) -> dict[str, float]:
    t = torch.tensor(values, dtype=torch.float32)
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "values": values}


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <out_results.json> <bank1.npz> [bank2.npz ...]")
        raise SystemExit(1)
    out_path = Path(sys.argv[1])
    bank_paths = [Path(p) for p in sys.argv[2:]]

    bank = [crop for path in bank_paths for crop in load_bank(path)]
    print(f"loaded {len(bank)} crops from {len(bank_paths)} bank file(s): {[p.name for p in bank_paths]}", flush=True)
    train_crops, val_crops = split_bank_by_source(bank, val_fraction=0.3, seed=0)
    print(f"bank: {len(bank)} crops from {len({c.source for c in bank})} scans "
          f"-> {len(train_crops)} train crops / {len(val_crops)} val crops, no shared source", flush=True)

    channel = ZChannel(p=CHANNEL_P)
    codebook = codebook_tensor(NEO_20_10)

    train_cfg = SyntheticConfig(num_messages=NUM_TRAIN_MESSAGES, channel=channel, group_j_min=GROUP_J_RANGE[0], group_j_max=GROUP_J_RANGE[1], augment=True, seed=1)
    val_cfg = SyntheticConfig(num_messages=NUM_VAL_MESSAGES, channel=channel, group_j_min=GROUP_J_RANGE[0], group_j_max=GROUP_J_RANGE[1], seed=2)

    t0 = time.time()
    train_dataset = generate_dataset(train_crops, NEO_20_10, train_cfg)
    val_dataset = generate_dataset(val_crops, NEO_20_10, val_cfg)
    print(f"generated {len(train_dataset.message_indices)} train / {len(val_dataset.message_indices)} val groups in {time.time() - t0:.1f}s", flush=True)

    val_corrupted_t = torch.as_tensor(val_dataset.corrupted_bits, dtype=torch.float32)
    val_clean_t = torch.as_tensor(val_dataset.clean_bits, dtype=torch.float32)
    val_targets_t = torch.as_tensor(val_dataset.message_indices, dtype=torch.long)
    val_mask_t = torch.as_tensor(val_dataset.group_mask, dtype=torch.float32)
    oracle = oracle_metrics(val_corrupted_t, val_clean_t, val_targets_t, val_mask_t, codebook, channel=channel)
    print(f"oracle (best possible): group_wer={oracle.group_wer:.4f} group_ber={oracle.group_ber:.4f}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, epochs={EPOCHS}, seeds={SEEDS}", flush=True)

    results = {
        "bank_files": [str(p) for p in bank_paths],
        "channel_p": CHANNEL_P,
        "group_j_range": list(GROUP_J_RANGE),
        "num_train_messages": NUM_TRAIN_MESSAGES,
        "num_val_messages": NUM_VAL_MESSAGES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seeds": SEEDS,
        "train_sources": sorted({c.source for c in train_crops}),
        "val_sources": sorted({c.source for c in val_crops}),
        "oracle": vars(oracle),
        "objectives": {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path.parent / "best_detector.pt"
    overall_best = {"group_wer": float("inf")}

    for objective in OBJECTIVES:
        print(f"--- training objective={objective} ---", flush=True)
        runs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            detector = Detector()
            cfg = TrainingConfig(objective=objective, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=device)
            t0 = time.time()
            history = train_detector(detector, train_dataset, val_dataset, codebook, cfg)
            elapsed = time.time() - t0
            final = history.val_metrics[-1]
            best = history.val_metrics[history.best_epoch]
            print(
                f"{objective} seed={seed}: final group_wer={final.group_wer:.4f} "
                f"best group_wer={best.group_wer:.4f} (epoch {history.best_epoch}) in {elapsed:.1f}s",
                flush=True,
            )
            runs.append(
                {
                    "seed": seed,
                    "epoch_loss": history.epoch_loss,
                    "val_group_wer": [m.group_wer for m in history.val_metrics],
                    "val_group_ber": [m.group_ber for m in history.val_metrics],
                    "val_single_wer": [m.single_wer for m in history.val_metrics],
                    "val_single_ber": [m.single_ber for m in history.val_metrics],
                    "final": vars(final),
                    "best_epoch": history.best_epoch,
                    "best": vars(best),
                    "train_seconds": elapsed,
                }
            )
            if best.group_wer < overall_best["group_wer"]:
                overall_best = {"group_wer": best.group_wer, "objective": objective, "seed": seed, "epoch": history.best_epoch}
                torch.save(
                    {"state_dict": history.best_state_dict, "objective": objective, "seed": seed, "group_wer": best.group_wer},
                    checkpoint_path,
                )

        results["objectives"][objective] = {
            "runs": runs,
            "final_group_wer": _mean_std([r["final"]["group_wer"] for r in runs]),
            "final_group_ber": _mean_std([r["final"]["group_ber"] for r in runs]),
            "best_group_wer": _mean_std([r["best"]["group_wer"] for r in runs]),
            "best_group_ber": _mean_std([r["best"]["group_ber"] for r in runs]),
        }
        summary = results["objectives"][objective]
        print(
            f"{objective} summary: final group_wer={summary['final_group_wer']['mean']:.4f}"
            f"+-{summary['final_group_wer']['std']:.4f}, "
            f"best group_wer={summary['best_group_wer']['mean']:.4f}+-{summary['best_group_wer']['std']:.4f}",
            flush=True,
        )
        results["overall_best"] = overall_best
        # Written after every objective, not just at the end, so a slow or
        # interrupted run still leaves the completed objectives on disk.
        out_path.write_text(json.dumps(results, indent=2))
        print(f"saved partial results -> {out_path}", flush=True)

    print(f"overall best: {overall_best}, checkpoint -> {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
