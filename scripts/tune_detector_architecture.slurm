#!/bin/bash
#SBATCH --partition=IMLcuda0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=arch_search
#SBATCH --output=out/arch_search_%j.out
#SBATCH --error=out/arch_search_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mkdir -p out
python3 scripts/tune_detector_architecture.py out/detector_arch_tuning.json \
    out/background_bank.npz out/yolo_background_bank.npz
