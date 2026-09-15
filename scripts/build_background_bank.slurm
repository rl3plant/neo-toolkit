#!/bin/bash
#SBATCH --partition=IMLcuda0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=rebuild_bank
#SBATCH --output=out/rebuild_bank_%j.out
#SBATCH --error=out/rebuild_bank_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
python3 scripts/build_background_bank.py \
  checkpoints/sam2.1_hiera_tiny.pt sam2.1/sam2.1_hiera_t.yaml out/background_bank.npz \
  ../old_stuff/NEO_ECC/rectangles/rectangle_34.png ../old_stuff/NEO_ECC/rectangles/rectangle_35.png \
  ../old_stuff/NEO_ECC/rectangles/rectangle_36.png ../old_stuff/NEO_ECC/rectangles/rectangle_37.png \
  ../old_stuff/NEO_ECC/rectangles/rectangle_38.png ../old_stuff/NEO_ECC/rectangles/rectangle_39.png \
  ../old_stuff/NEO_ECC/rectangles/rectangle_40.png ../old_stuff/NEO_ECC/rectangles/rectangle_41.png \
  ../old_stuff/NEO_ECC/rectangles/rectangle_42.png ../old_stuff/NEO_ECC/rectangles/rectangle_43.png \
  ../old_stuff/NEO_ECC/rectangles/rectangle_44.png ../old_stuff/NEO_ECC/rectangles/rectangle_45.png
