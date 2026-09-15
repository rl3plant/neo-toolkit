#!/bin/bash
#SBATCH --partition=IMLcuda0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=d37_rerun
#SBATCH --output=out/d37_rerun_%j.out
#SBATCH --error=out/d37_rerun_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p out
python3 scripts/rerun_d37.py out/d37_rerun.json out/background_bank.npz out/yolo_background_bank.npz

# Prove the trained detector actually works chained behind real SAM2
# segmentation on a raw scan, not just on pre-cropped synthetic data.
python3 scripts/end_to_end_demo.py checkpoints/sam2.1_hiera_tiny.pt \
    sam2.1/sam2.1_hiera_t.yaml out/best_detector.pt \
    ../old_stuff/NEO_ECC/rectangles/rectangle_43.png
