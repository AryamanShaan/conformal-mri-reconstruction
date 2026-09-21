#!/bin/bash
#SBATCH --job-name=create_quantile_bounds
#SBATCH --partition=a100_short,a100_long,radiology
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
# a100-4042 and a100-4048 are known CUDA-13 (580.x) nodes -- cenv3 torch(cu12) fails
# there with "CUDA-capable device(s) is/are busy or unavailable" at torch.cuda.set_device.
#SBATCH --exclude=a100-4042,a100-4048

set -euo pipefail

# Split passed as the first argument (snapshotted at submit time -> no edit/race).
SPLIT="${1:?usage: sbatch slurm_scipts/run_create_quantile_bounds.sh <split>   (e.g. calibration | test)}"

DATA_ROOT=/gpfs/scratch/shaana01/knee_fastmri_cp
OUT_ROOT=/gpfs/scratch/shaana01/conformal-mri-reconstruction-logs

INPUT_DATA_DIR="${DATA_ROOT}/multicoil_${SPLIT}"
OUTPUT_PT="${OUT_ROOT}/quantile_bounds_${SPLIT}_config_14.pt"

mkdir -p logs

source /gpfs/scratch/shaana01/anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/scratch/shaana01/anaconda3/envs/cenv3

# sbatch runs this with CWD = the submit dir (repo root), so put that on PYTHONPATH
# for common.* / e2evarnet.* / quantile.* to resolve.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "SPLIT=${SPLIT}"
echo "INPUT_DATA_DIR=${INPUT_DATA_DIR}"
echo "OUTPUT_PT=${OUTPUT_PT}"

python quantile/inference/create_quantile_bounds.py \
  --input_data_dir "${INPUT_DATA_DIR}" \
  --output_pt      "${OUTPUT_PT}"


# HOW TO RUN -- submit from the repo root (conformal-mri-reconstruction/), once per split.
# No editing between runs; both can be submitted immediately:
#   cd /gpfs/.../QR/conformal-mri-reconstruction
#   sbatch slurm_scipts/run_create_quantile_bounds.sh calibration
#   sbatch slurm_scipts/run_create_quantile_bounds.sh test
#
# The .py takes --input_data_dir / --output_pt as CLI args (the checkpoints and
# sampling_rates stay hardcoded inside create_quantile_bounds.py). The split is
# baked into the sbatch command, so editing the .py between submissions is no
# longer needed (and the earlier live-read race is gone).
