#!/bin/bash
#SBATCH --partition=IMLcuda0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=d37_trellis
#SBATCH --output=out/d37_trellis_%j.out
#SBATCH --error=out/d37_trellis_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p out
python3 scripts/rerun_d37_trellis.py out/d37_trellis.json out/background_bank.npz out/yolo_background_bank.npz
