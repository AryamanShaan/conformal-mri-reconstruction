#!/bin/bash
#SBATCH --job-name=varnet_knee
#SBATCH --partition=a100_long,radiology
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
# a100-4042 is a known CUDA-13 (580.x) node -- cenv3 torch(cu12) fails there with "device busy/unavailable".
#SBATCH --exclude=a100-4042

set -euo pipefail



mkdir -p logs

source /gpfs/scratch/shaana01/anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/scratch/shaana01/anaconda3/envs/cenv3

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python e2evarnet/runner.py --config configs/e2evarnet_knee_random_vds_3.yaml


#   So launch like this:
#   cd /gpfs/.../QR/conformal-mri-reconstruction     # the repo root
#   sbatch slurm_scipts/run_varnet.sh


# [shaana01@bigpurple-ln2 conformal-mri-reconstruction]$ squeue -u $USER
#              JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
#           26418054 a100_long varnet_k shaana01  R    2:02:39      1 a100-4038
#           26427935 a100_long varnet_k shaana01 PD       0:00      1 (Priority)
#           26414680 radiology varnet_k shaana01  R    6:54:03      1 rc-4001
