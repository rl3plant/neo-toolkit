#!/bin/bash
#SBATCH --partition=IMLcuda0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=seg_tune
#SBATCH --output=out/seg_tune_%j.out
#SBATCH --error=out/seg_tune_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p out
python3 scripts/tune_segmentation.py checkpoints/sam2.1_hiera_tiny.pt \
    sam2.1/sam2.1_hiera_t.yaml out/segmentation_tuning.json \
    "../neo_project/Data/osif shape rectangles" \
    "../neo_project/Data/final design rectangle Monometric Strepttavidin" \
    "../neo_project/Data/final design - rectangle + streptavidin" \
    "../neo_project/Data/rectangle+IgG-mica-dry-Bruker"
