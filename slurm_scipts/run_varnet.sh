#!/bin/bash
#SBATCH --job-name=varnet_knee
#SBATCH --partition=radiology,a100_long
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
# a100-4042 and a100-4048 are known CUDA-13 (580.x) nodes -- cenv3 torch(cu12) fails
# there with "CUDA-capable device(s) is/are busy or unavailable" at torch.cuda.set_device.
#SBATCH --exclude=a100-4042,a100-4048

set -euo pipefail



mkdir -p logs

source /gpfs/scratch/shaana01/anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/scratch/shaana01/anaconda3/envs/cenv3

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python e2evarnet/runner.py --config configs/e2evarnet_knee_random_vds_6.yaml


#   So launch like this:
#   cd /gpfs/.../QR/conformal-mri-reconstruction     # the repo root
#   sbatch slurm_scipts/run_varnet.sh

