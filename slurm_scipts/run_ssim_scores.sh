#!/bin/bash
#SBATCH --job-name=varnet_ssim_scores
#SBATCH --partition=a100_short,a100_long,radiology
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
# a100-4042 is a known CUDA-13 (580.x) node -- cenv3 torch(cu12) fails there with "device busy/unavailable".
#SBATCH --exclude=a100-4042

set -euo pipefail

mkdir -p logs

source /gpfs/scratch/shaana01/anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/scratch/shaana01/anaconda3/envs/cenv3

# sbatch runs this with CWD = the directory you submitted from, so put THAT (the
# repo root) on PYTHONPATH -- that is how `common.*` / `e2evarnet.*` resolve.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python e2evarnet/inference/varnet_ssim_scores.py


# HOW TO RUN -- submit from the repo root (conformal-mri-reconstruction/), NOT from
# inside slurm_scipts/, so $(pwd)/PYTHONPATH points at the package root:
#   cd /gpfs/scratch/shaana01/.../QR/conformal-mri-reconstruction
#   sbatch slurm_scipts/run_ssim_scores.sh
