#!/usr/bin/env python3
"""Search Detector architecture/learning-rate against the leakage-free ECC setup.

    python scripts/tune_detector_architecture.py out/detector_arch_tuning.json \
        out/background_bank.npz out/yolo_background_bank.npz

These are tiny networks (n_layers=2, base_channels=32 is ~10K params), so a
search is cheap: each search trial retrains from scratch at a reduced epoch
budget (150, vs. the full 400) on a single seed, scored on the
neurosymbolic objective (the current best performer, and the "purest"
ECC-aware signal). Optuna TPE for the same reason as tune_segmentation.py --
fewer, smarter evaluations. After the search, the best (n_layers,
base_channels, lr) is confirmed with a full run: all three objectives,
400 epochs, 3 seeds, same protocol as rerun_d37.py.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import optuna
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
SEARCH_OBJECTIVE = "neurosymbolic"
SEARCH_EPOCHS = 200
SEARCH_TRIALS = 30
CONFIRM_EPOCHS = 400
CONFIRM_SEEDS = [0, 1, 2]
CHANNEL_P = 0.2
NUM_TRAIN_MESSAGES = 256
NUM_VAL_MESSAGES = 128
GROUP_J_RANGE = (10, 20)
BATCH_SIZE = 32


def _mean_std(values: list[float]) -> dict[str, float]:
    t = torch.tensor(values, dtype=torch.float32)
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "values": values}


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <out.json> <bank1.npz> [bank2.npz ...]")
        raise SystemExit(1)
    out_path = Path(sys.argv[1])
    bank_paths = [Path(p) for p in sys.argv[2:]]

    bank = [crop for path in bank_paths for crop in load_bank(path)]
    train_crops, val_crops = split_bank_by_source(bank, val_fraction=0.3, seed=0)
    print(f"bank: {len(bank)} crops -> {len(train_crops)} train / {len(val_crops)} val crops", flush=True)

    channel = ZChannel(p=CHANNEL_P)
    codebook = codebook_tensor(NEO_20_10)
    train_cfg = SyntheticConfig(num_messages=NUM_TRAIN_MESSAGES, channel=channel, group_j_min=GROUP_J_RANGE[0], group_j_max=GROUP_J_RANGE[1], augment=True, seed=1)
    val_cfg = SyntheticConfig(num_messages=NUM_VAL_MESSAGES, channel=channel, group_j_min=GROUP_J_RANGE[0], group_j_max=GROUP_J_RANGE[1], seed=2)
    train_dataset = generate_dataset(train_crops, NEO_20_10, train_cfg)
    val_dataset = generate_dataset(val_crops, NEO_20_10, val_cfg)
    print(f"generated {len(train_dataset.message_indices)} train / {len(val_dataset.message_indices)} val groups", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        base_channels = trial.suggest_categorical("base_channels", [4, 8, 16, 32, 64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
        norm = trial.suggest_categorical("norm", ["none", "batch", "group"])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        torch.manual_seed(0)
        detector = Detector(base_channels=base_channels, n_layers=n_layers, kernel_size=kernel_size, norm=norm, dropout=dropout)
        n_params = sum(p.numel() for p in detector.parameters())
        cfg = TrainingConfig(objective=SEARCH_OBJECTIVE, epochs=SEARCH_EPOCHS, batch_size=BATCH_SIZE, lr=lr, device=device)
        trial.set_user_attr("n_params", n_params)

        # A wide/shallow architecture (large base_channels, few n_layers, so
        # feature maps stay big) can still blow GPU memory even after
        # _forward_logits's chunking. Optuna doesn't catch exceptions by
        # default, so one bad trial would otherwise kill the whole study --
        # catch it, record a poor score, and keep going. Separately: this is
        # one long-running process running ~30 trials with wildly different
        # model sizes back to back, and PyTorch's CUDA caching allocator can
        # fragment across very differently-shaped allocations rather than
        # cleanly reusing/releasing memory between trials -- explicitly drop
        # the model and empty the cache after every trial (not just failed
        # ones) so usage doesn't creep upward and eventually OOM a trial that
        # would otherwise have fit fine on its own (observed: 17GB/23.5GB in
        # use by trial 4, for models a few hundred KB in size).
        t0 = time.time()
        try:
            history = train_detector(detector, train_dataset, val_dataset, codebook, cfg)
            best_wer = history.val_metrics[history.best_epoch].group_wer
        except RuntimeError as e:
            elapsed = time.time() - t0
            trial.set_user_attr("failed", str(e)[:200])
            trial.set_user_attr("elapsed_seconds", elapsed)
            print(
                f"trial {trial.number}: n_layers={n_layers} base_channels={base_channels} lr={lr:.2e} "
                f"kernel_size={kernel_size} norm={norm} dropout={dropout:.2f} n_params={n_params} "
                f"-> FAILED ({elapsed:.1f}s): {e}",
                flush=True,
            )
            return 1.0  # worst possible group WER -- Optuna (minimize) will never pick this
        finally:
            del detector
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        trial.set_user_attr("elapsed_seconds", elapsed)
        print(
            f"trial {trial.number}: n_layers={n_layers} base_channels={base_channels} lr={lr:.2e} "
            f"kernel_size={kernel_size} norm={norm} dropout={dropout:.2f} "
            f"n_params={n_params} -> best_group_wer={best_wer:.4f} ({elapsed:.1f}s)",
            flush=True,
        )
        return best_wer

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    t0 = time.time()
    study.optimize(objective, n_trials=SEARCH_TRIALS)
    search_elapsed = time.time() - t0
    print(f"\nsearch done in {search_elapsed:.1f}s. best trial #{study.best_trial.number}: group_wer={study.best_value:.4f}", flush=True)
    print(f"best params: {study.best_params}", flush=True)
    print(f"default was: n_layers=2, base_channels=32, lr=1e-3, kernel_size=3, norm=none, dropout=0", flush=True)

    results = {
        "search": {
            "objective": SEARCH_OBJECTIVE,
            "epochs": SEARCH_EPOCHS,
            "n_trials": SEARCH_TRIALS,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "trials": [
                {"number": t.number, "value": t.value, "params": t.params, "user_attrs": t.user_attrs}
                for t in study.trials
            ],
        },
        "confirmation": {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    # Confirm the winning architecture against the full protocol: all three
    # objectives, full epoch budget, multiple seeds -- exactly like rerun_d37.py.
    best = study.best_params
    val_corrupted_t = torch.as_tensor(val_dataset.corrupted_bits, dtype=torch.float32)
    val_clean_t = torch.as_tensor(val_dataset.clean_bits, dtype=torch.float32)
    val_targets_t = torch.as_tensor(val_dataset.message_indices, dtype=torch.long)
    val_mask_t = torch.as_tensor(val_dataset.group_mask, dtype=torch.float32)
    oracle = oracle_metrics(val_corrupted_t, val_clean_t, val_targets_t, val_mask_t, codebook, channel=channel)
    results["confirmation"]["oracle"] = vars(oracle)
    print(f"\n--- confirming best architecture, oracle group_wer={oracle.group_wer:.4f} ---", flush=True)

    for objective_name in OBJECTIVES:
        print(f"--- confirm objective={objective_name} ---", flush=True)
        runs = []
        for seed in CONFIRM_SEEDS:
            torch.manual_seed(seed)
            detector = Detector(
                base_channels=best["base_channels"],
                n_layers=best["n_layers"],
                kernel_size=best["kernel_size"],
                norm=best["norm"],
                dropout=best["dropout"],
            )
            cfg = TrainingConfig(objective=objective_name, epochs=CONFIRM_EPOCHS, batch_size=BATCH_SIZE, lr=best["lr"], device=device)
            t0 = time.time()
            history = train_detector(detector, train_dataset, val_dataset, codebook, cfg)
            elapsed = time.time() - t0
            final = history.val_metrics[-1]
            best_metrics = history.val_metrics[history.best_epoch]
            print(
                f"{objective_name} seed={seed}: final group_wer={final.group_wer:.4f} "
                f"best group_wer={best_metrics.group_wer:.4f} in {elapsed:.1f}s",
                flush=True,
            )
            runs.append({"seed": seed, "final": vars(final), "best": vars(best_metrics), "train_seconds": elapsed})
            del detector
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        results["confirmation"][objective_name] = {
            "runs": runs,
            "final_group_wer": _mean_std([r["final"]["group_wer"] for r in runs]),
            "best_group_wer": _mean_std([r["best"]["group_wer"] for r in runs]),
        }
        summary = results["confirmation"][objective_name]
        print(
            f"{objective_name} confirmed: best group_wer={summary['best_group_wer']['mean']:.4f}"
            f"+-{summary['best_group_wer']['std']:.4f}",
            flush=True,
        )
        out_path.write_text(json.dumps(results, indent=2))
        print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
