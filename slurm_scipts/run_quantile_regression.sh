#!/bin/bash
#SBATCH --job-name=quantile_regression
#SBATCH --partition=a100_long,radiology
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
# a100-4042 and a100-4048 are known CUDA-13 (580.x) nodes -- cenv3 torch(cu12) fails
# there with "CUDA-capable device(s) is/are busy or unavailable" at torch.cuda.set_device.
#SBATCH --exclude=a100-4042,a100-4048

set -euo pipefail

# NOTE: --gres=gpu:N above must match `num_gpus:` in the config (the trainer uses
# devices=num_gpus). Both are 4 here.

mkdir -p logs

source /gpfs/scratch/shaana01/anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/scratch/shaana01/anaconda3/envs/cenv3

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Edit the --config line to the bound you want to train (upper=q0.95, lower=q0.05).
python quantile/train_quantile_regression.py \
--config configs/quantile_regression_batches_lower_batch_14.yaml
  # --config configs/quantile_regression_batches_upper_batch_14.yaml



# HOW TO RUN -- submit from the repo root (conformal-mri-reconstruction/), once per bound:
#   cd /gpfs/.../QR/conformal-mri-reconstruction
#   sbatch slurm_scipts/run_quantile_regression.sh
