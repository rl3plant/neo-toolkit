#!/usr/bin/env python3
"""Bayesian-optimize neo_toolkit.segmentation's hyperparameters against real,
human-verified ground truth (neo_project/Data).

    python scripts/tune_segmentation.py checkpoints/sam2.1_hiera_tiny.pt \
        sam2.1/sam2.1_hiera_t.yaml out/segmentation_tuning.json \
        "../neo_project/Data/osif shape rectangles" \
        "../neo_project/Data/final design rectangle Monometric Strepttavidin" \
        "../neo_project/Data/final design - rectangle + streptavidin" \
        "../neo_project/Data/rectangle+IgG-mica-dry-Bruker"

Each trial means actually running SAM2 over the validation images -- not
cheap -- so this uses Optuna's default TPE sampler (a sequential
model-based / Bayesian-optimization-style method) instead of grid or random
search, to get a good result in as few expensive evaluations as possible.

Caveat: the same real images may also appear in the ECC background bank
(scripts/build_yolo_background_bank.py) -- fine here, since tuning a
handful of geometric thresholds by IoU isn't the kind of high-capacity
fitting that memorizes individual images the way training a network would,
but worth knowing if these numbers get reused elsewhere.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neo_toolkit.segmentation.evaluation import aggregate, load_ground_truth, match_instances
from neo_toolkit.segmentation.pipeline import segment_image
from neo_toolkit.segmentation.prompting import foreground_component_points
from neo_toolkit.segmentation.sam_backend import SamSegmenter

N_TRIALS = 25
MAX_IMAGES_PER_CATEGORY = 3


def main() -> None:
    if len(sys.argv) < 5:
        print(f"usage: {sys.argv[0]} <sam2-checkpoint> <sam2-model-cfg> <out.json> <category_dir> [category_dir ...]")
        raise SystemExit(1)
    sam_checkpoint, model_cfg, out_path_str = sys.argv[1:4]
    category_dirs = sys.argv[4:]
    out_path = Path(out_path_str)

    items = []
    for category_dir in category_dirs:
        items.extend(load_ground_truth(category_dir)[:MAX_IMAGES_PER_CATEGORY])
    print(f"validation set: {len(items)} images across {len(category_dirs)} categories", flush=True)
    images = {item.image_path: cv2.imread(str(item.image_path), cv2.IMREAD_GRAYSCALE) for item in items}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = SamSegmenter(sam_checkpoint, model_cfg=model_cfg, device=device)
    print(f"device: {device}, trials: {N_TRIALS}", flush=True)

    def objective(trial: optuna.Trial) -> float:
        morph_kernel = trial.suggest_int("morph_kernel", 1, 7, step=2)
        merge_area_ratio = trial.suggest_float("merge_area_ratio", 1.2, 3.0)
        dense_grid_n = trial.suggest_int("dense_grid_n", 2, 6)
        area_ratio_min = trial.suggest_float("area_ratio_min", 0.2, 0.7)
        area_ratio_max = trial.suggest_float("area_ratio_max", 1.3, 3.0)
        nms_iou = trial.suggest_float("nms_iou", 0.05, 0.5)

        per_image = []
        for item in items:
            image = images[item.image_path]
            points = foreground_component_points(
                image, morph_kernel=morph_kernel, merge_area_ratio=merge_area_ratio, dense_grid=(dense_grid_n, dense_grid_n)
            )
            predicted = segment_image(image, sam, prompt_points=points, area_ratio=(area_ratio_min, area_ratio_max), nms_iou=nms_iou)
            per_image.append(match_instances(predicted, item.masks))

        agg = aggregate(per_image)
        trial.set_user_attr("precision", agg.precision)
        trial.set_user_attr("recall", agg.recall)
        trial.set_user_attr("mean_iou", agg.mean_iou)
        trial.set_user_attr("n_predicted", agg.n_predicted)
        trial.set_user_attr("n_ground_truth", agg.n_ground_truth)
        print(
            f"trial {trial.number}: f1={agg.f1:.4f} precision={agg.precision:.4f} recall={agg.recall:.4f} "
            f"mean_iou={agg.mean_iou:.4f} (pred={agg.n_predicted}, gt={agg.n_ground_truth})",
            flush=True,
        )
        return agg.f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    elapsed = time.time() - t0

    print(f"\nbest trial #{study.best_trial.number}: f1={study.best_value:.4f}", flush=True)
    print(f"best params: {study.best_params}", flush=True)
    print(f"defaults were: morph_kernel=3, merge_area_ratio=1.8, dense_grid_n=4, area_ratio=(0.5,2.0), nms_iou=0.2", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "n_trials": N_TRIALS,
                "n_validation_images": len(items),
                "category_dirs": category_dirs,
                "elapsed_seconds": elapsed,
                "best_value_f1": study.best_value,
                "best_params": study.best_params,
                "best_trial_attrs": study.best_trial.user_attrs,
                "trials": [
                    {"number": t.number, "value": t.value, "params": t.params, "user_attrs": t.user_attrs}
                    for t in study.trials
                ],
            },
            indent=2,
        )
    )
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
