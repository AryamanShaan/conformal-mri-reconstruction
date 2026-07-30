#!/bin/bash
#SBATCH --partition=a100_short,a100_long,radiology
#SBATCH --job-name=ilias_varnet
#SBATCH --nodes=1
# NOTE(PL2.x/DDP): Lightning under SLURM requires one srun task per GPU rank,
# i.e. ntasks-per-node must equal the number of GPUs (Trainer devices).
# Was: --ntasks-per-node=1 (fine for the author's original single-GPU runs).
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# --- logging -------------------------------------------------------------
# Create a logs/ dir next to the submitted script (SLURM_SUBMIT_DIR = the dir
# you ran `sbatch` from; falls back to cwd for a plain `bash` run) and tee ALL
# stdout/stderr into a per-job file there. Done in-script so the dir is created
# automatically if missing -- #SBATCH --output/--error cannot create a dir or
# use these variables, which is why they stay /dev/null in the header above.
LOG_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/logs"
mkdir -p "${LOG_DIR}"
SLURM_LOG_FILE="${LOG_DIR}/slurm_${SLURM_JOB_NAME:-VN_recon}_${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "${SLURM_LOG_FILE}") 2>&1
# -------------------------------------------------------------------------

# ANATOMY=${ANATOMY:-brain}
ANATOMY=${ANATOMY:-knee}
ACCELERATION=${ACCELERATION:-8}
CENTER_FRACTION=${CENTER_FRACTION:-}
MASK_MODE=${MASK_MODE:-fixed}
FIXED_MASK_TYPE=${FIXED_MASK_TYPE:-random_vds}
NUM_LOGITS=${NUM_LOGITS:-320}
# random_vds mask sampling budget (used only when FIXED_MASK_TYPE=random_vds)
MIN_K_FRACTION=${MIN_K_FRACTION:-0.10}
MAX_K_FRACTION=${MAX_K_FRACTION:-0.30}
TRAIN_MODE=${TRAIN_MODE:-train}
TEST_MODE=${TEST_MODE:-test_val}
FINE_TUNE_CKPT=${FINE_TUNE_CKPT:-your_checkpoint.ckpt}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
PRECISION=${PRECISION:-32}

# file-level smart split
# full = no split, use all files
# odd  = 1st, 3rd, 5th, ... sorted files
# even = 2nd, 4th, 6th, ... sorted files
VN_FILE_SPLIT=${VN_FILE_SPLIT:-full}
UNC_FILE_SPLIT=${UNC_FILE_SPLIT:-odd}

RUN_TRAINING=${RUN_TRAINING:-true}
RUN_TESTING=${RUN_TESTING:-false}
RUN_EVALUATION=${RUN_EVALUATION:-false}
RUN_UNCERTAINTY_TRAINING=${RUN_UNCERTAINTY_TRAINING:-false}
RUN_UNCERTAINTY_CALIBRATION=${RUN_UNCERTAINTY_CALIBRATION:-false}
RUN_UNCERTAINTY_TESTING=${RUN_UNCERTAINTY_TESTING:-false}

# reconstruction
MODEL_NAME=${MODEL_NAME:-e2e_varnet}
# NUM_CASCADES=${NUM_CASCADES:-8}
NUM_CASCADES=${NUM_CASCADES:-12}
CHANS=${CHANS:-32}
POOLS=${POOLS:-4}
SENS_CHANS=${SENS_CHANS:-8}
SENS_POOLS=${SENS_POOLS:-4}
VN_LR=${VN_LR:-0.0003}
VN_LR_BASE=${VN_LR_BASE:-}
VN_LR_MASK=${VN_LR_MASK:-}
VN_WEIGHT_DECAY=${VN_WEIGHT_DECAY:-0.0}
# NOTE: 105000/3750/75000 was the repo's original 1-GPU default (~6 epochs
# on this dataset). Updated below for 60 epochs on 4 GPUs, batch_size=1,
# full-data combined train+val (~36,492 slices -> 9123 steps/epoch):
#   VN_MAX_STEPS=${VN_MAX_STEPS:-105000}
#   VN_RAMP_STEPS=${VN_RAMP_STEPS:-3750}
#   VN_COSINE_DECAY_START=${VN_COSINE_DECAY_START:-75000}
VN_MAX_STEPS=${VN_MAX_STEPS:-547380}
VN_RAMP_STEPS=${VN_RAMP_STEPS:-19549}
VN_COSINE_DECAY_START=${VN_COSINE_DECAY_START:-390986}

# uncertainty
UNC_ALPHA=${UNC_ALPHA:-0.1}
UNC_CAL_DELTA=${UNC_CAL_DELTA:-0.1}
UNC_CAL_LAM_START=${UNC_CAL_LAM_START:-1.0}
UNC_CAL_LAM_END=${UNC_CAL_LAM_END:-2.0}
UNC_CAL_LAM_STEPS=${UNC_CAL_LAM_STEPS:-20}
UNC_CAL_SAMPLE_RATE=${UNC_CAL_SAMPLE_RATE:-1.0}
UNC_HEAD_CHANS=${UNC_HEAD_CHANS:-32}
UNC_HEAD_POOLS=${UNC_HEAD_POOLS:-4}
UNC_DROP_PROB=${UNC_DROP_PROB:-0.0}
UNC_LR=${UNC_LR:-0.0003}
UNC_WEIGHT_DECAY=${UNC_WEIGHT_DECAY:-0.0}
UNC_MAX_STEPS=${UNC_MAX_STEPS:-105000}
UNC_RAMP_STEPS=${UNC_RAMP_STEPS:-3750}
UNC_COSINE_DECAY_START=${UNC_COSINE_DECAY_START:-75000}

# repository location
# NOTE(recon-migration): now points at THIS repo root (the dir containing common/ e2evarnet/
# quantile/ conformal/), not the old /VarNet repo. Defaults to the directory you ran `sbatch`
# from -- so run `sbatch slurm_scipts/batch_run_1.sh` from the repo root, or override with
# VARNET_DIR=/path/to/conformal-mri-reconstruction. This dir goes on PYTHONPATH (line below).
# VARNET_DIR=${VARNET_DIR:-/gpfs/home/shaana01/ilias_recon/VarNet}
VARNET_DIR=${VARNET_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}

# read-only fastMRI dataset location
# DATA_ROOT=${DATA_ROOT:-/gpfs/data/gianni02lab/Team/Datasets/FastMRI}
# DATA_DIR=${DATA_DIR:-${DATA_ROOT}/${ANATOMY}}
# VN_TRAIN_PATH=${VN_TRAIN_PATH:-${DATA_DIR}/multicoil_train}
# VN_VAL_PATH=${VN_VAL_PATH:-${DATA_DIR}/multicoil_val}
# VN_CAL_PATH=${VN_CAL_PATH:-${DATA_DIR}/multicoil_cal}
# VN_TEST_PATH=${VN_TEST_PATH:-${DATA_DIR}/multicoil_test}
VN_TRAIN_PATH=${VN_TRAIN_PATH:-/gpfs/scratch/shaana01/knee_fastmri_cp/multicoil_train}
VN_VAL_PATH=${VN_VAL_PATH:-/gpfs/scratch/shaana01/knee_fastmri_cp/multicoil_val}
VN_CAL_PATH=${VN_CAL_PATH:-/gpfs/scratch/shaana01/knee_fastmri_cp/multicoil_calibration}
VN_TEST_PATH=${VN_TEST_PATH:-/gpfs/scratch/shaana01/knee_fastmri_cp/multicoil_test}

# NOTE: DATA_DIR only feeds the optional --data_path arg, used solely as a
# prepare_data() fallback ("<data_path>/multicoil_test") when --test_path is
# not passed (e.g. during a training-only run). Not used for train/val/test
# dataloading itself -- those are the explicit VN_*_PATH vars above. Must stay
# set (this script has `set -u`) and its /multicoil_test subfolder must exist,
# so it's pointed at the shared parent of the paths above.
DATA_DIR=${DATA_DIR:-/gpfs/scratch/shaana01/knee_fastmri_cp}


# user-owned output location
# <-- your own writable output dir; everything (ckpts, TB + CSV logs, cache, slurm log) nests under here
OUTPUT_ROOT=${OUTPUT_ROOT:-/gpfs/scratch/shaana01/ilias_varnet/run_1}

if [ -z "${CENTER_FRACTION}" ]; then
    case ${ACCELERATION} in
        2) CENTER_FRACTION=0.16 ;;
        4) CENTER_FRACTION=0.08 ;;
        6) CENTER_FRACTION=0.06 ;;
        8) CENTER_FRACTION=0.04 ;;
        10) CENTER_FRACTION=0.02 ;;
        *) CENTER_FRACTION=0.08 ;;
    esac
fi

if [ "${MASK_MODE}" = "learnable" ]; then
    MASK_TAG=learnable_mask
else
    MASK_TAG=${FIXED_MASK_TYPE}
fi

if [ "${VN_FILE_SPLIT}" = "full" ]; then
    VN_SPLIT_TAG=""
else
    VN_SPLIT_TAG="_split_${VN_FILE_SPLIT}"
fi

if [ "${UNC_FILE_SPLIT}" = "full" ]; then
    UNC_SPLIT_TAG=""
else
    UNC_SPLIT_TAG="_split_${UNC_FILE_SPLIT}"
fi

VN_MODEL=Model_${ANATOMY}_${MODEL_NAME}_${ACCELERATION}x_SSIM_${MASK_TAG}${VN_SPLIT_TAG}
UNC_MODEL=${VN_MODEL}_uncertainty${UNC_SPLIT_TAG}

# Every generated file is stored below this model-specific directory.
MODEL_OUTPUT_DIR=${OUTPUT_ROOT}/${VN_MODEL}
LOG_PATH=${MODEL_OUTPUT_DIR}
RUN_ROOT_DIR=${MODEL_OUTPUT_DIR}/${MODEL_NAME}
UNC_ROOT_DIR=${RUN_ROOT_DIR}/uncertainty
CHECKPOINT_DIR=${RUN_ROOT_DIR}/checkpoints/${VN_MODEL}
UNC_CHECKPOINT_DIR=${UNC_ROOT_DIR}/checkpoints/${UNC_MODEL}
VN_PREDICTIONS_PATH=${RUN_ROOT_DIR}/reconstructions_${VN_MODEL}
VARNET_CKPT_PREFIX=fi_varnet.

mkdir -p "${MODEL_OUTPUT_DIR}"

# SLURM stdout/stderr is also stored inside the model directory.
# NOTE: superseded by the logs/ block near the top of this script.
# SLURM_LOG_FILE=${MODEL_OUTPUT_DIR}/slurm_${SLURM_JOB_NAME:-VN_recon}_${SLURM_JOB_ID:-manual}.out
# exec > >(tee -a "${SLURM_LOG_FILE}") 2>&1

# Activate the conda env so python3 (and its torch/lightning) resolve correctly.
source /gpfs/scratch/shaana01/anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/scratch/shaana01/anaconda3/envs/cenv3

# Make the repository importable while keeping the working directory inside
# MODEL_OUTPUT_DIR. This also stores dataset_cache.pkl in the model directory
# instead of the repository or the read-only dataset directory.
export PYTHONPATH="${VARNET_DIR}:${PYTHONPATH:-}"
cd "${MODEL_OUTPUT_DIR}"

resolve_preferred_ckpt() {
    local ckpt_dir="$1"
    local preferred=""
    local fallback=""
    preferred=$(find "${ckpt_dir}" -maxdepth 1 -type f -name "*.ckpt" ! -name "last.ckpt" 2>/dev/null | sort | tail -n 1 || true)
    fallback="${ckpt_dir}/last.ckpt"
    if [ -n "${preferred}" ] && [ -f "${preferred}" ]; then
        echo "${preferred}"
    elif [ -f "${fallback}" ]; then
        echo "${fallback}"
    else
        echo ""
    fi
}

MASK_ARGS=(--mask_mode "${MASK_MODE}")
if [ "${MASK_MODE}" = "fixed" ]; then
    MASK_ARGS+=(--mask_type "${FIXED_MASK_TYPE}")
    if [ "${FIXED_MASK_TYPE}" = "random_vds" ]; then
        MASK_ARGS+=(--min_k_fraction "${MIN_K_FRACTION}" --max_k_fraction "${MAX_K_FRACTION}")
    fi
else
    MASK_ARGS+=(--num_logits "${NUM_LOGITS}")
fi

TRAIN_MODE_ARGS=(--mode "${TRAIN_MODE}")
if [ "${TRAIN_MODE}" = "fine_tune" ]; then
    TRAIN_MODE_ARGS+=(--fine_tune_ckpt "${FINE_TUNE_CKPT}")
fi

VN_SPLIT_ARGS=(--file_split "${VN_FILE_SPLIT}")
UNC_SPLIT_ARGS=(--file_split "${UNC_FILE_SPLIT}")

COMMON_ARGS=(
    --data_path "${DATA_DIR}"
    --data_path_train "${VN_TRAIN_PATH}"
    --data_path_val "${VN_VAL_PATH}"
    --log_path "${LOG_PATH}"
    --accelerations "${ACCELERATION}"
    --center_fractions "${CENTER_FRACTION}"
    --varnet_type "${MODEL_NAME}"
    --model_name "${VN_MODEL}"
    --num_cascades "${NUM_CASCADES}"
    --chans "${CHANS}"
    --pools "${POOLS}"
    --sens_chans "${SENS_CHANS}"
    --sens_pools "${SENS_POOLS}"
    --precision "${PRECISION}"
    --lr "${VN_LR}"
    --weight_decay "${VN_WEIGHT_DECAY}"
    --max_steps "${VN_MAX_STEPS}"
    --ramp_steps "${VN_RAMP_STEPS}"
    --cosine_decay_start "${VN_COSINE_DECAY_START}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
)

if [ -n "${VN_LR_BASE}" ]; then
    COMMON_ARGS+=(--lr_base "${VN_LR_BASE}")
fi
if [ -n "${VN_LR_MASK}" ]; then
    COMMON_ARGS+=(--lr_mask "${VN_LR_MASK}")
fi

UNC_ARGS=(
    --data_path "${DATA_DIR}"
    --data_path_train "${VN_TRAIN_PATH}"
    --data_path_val "${VN_VAL_PATH}"
    --data_path_cal "${VN_CAL_PATH}"
    --log_path "${LOG_PATH}"
    --accelerations "${ACCELERATION}"
    --center_fractions "${CENTER_FRACTION}"
    --varnet_type "${MODEL_NAME}"
    --model_name "${UNC_MODEL}"
    --varnet_ckpt_prefix "${VARNET_CKPT_PREFIX}"
    --num_cascades "${NUM_CASCADES}"
    --chans "${CHANS}"
    --pools "${POOLS}"
    --sens_chans "${SENS_CHANS}"
    --sens_pools "${SENS_POOLS}"
    --precision "${PRECISION}"
    --alpha "${UNC_ALPHA}"
    --head_chans "${UNC_HEAD_CHANS}"
    --head_pools "${UNC_HEAD_POOLS}"
    --uncertainty_drop_prob "${UNC_DROP_PROB}"
    --lr "${UNC_LR}"
    --weight_decay "${UNC_WEIGHT_DECAY}"
    --max_steps "${UNC_MAX_STEPS}"
    --ramp_steps "${UNC_RAMP_STEPS}"
    --cosine_decay_start "${UNC_COSINE_DECAY_START}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --calibration_delta "${UNC_CAL_DELTA}"
    --calibration_lambdas_start "${UNC_CAL_LAM_START}"
    --calibration_lambdas_end "${UNC_CAL_LAM_END}"
    --calibration_lambdas_steps "${UNC_CAL_LAM_STEPS}"
    --calibration_sample_rate "${UNC_CAL_SAMPLE_RATE}"
)

echo "============================================================"
echo "VN_MODEL              : ${VN_MODEL}"
echo "UNC_MODEL             : ${UNC_MODEL}"
echo "ANATOMY               : ${ANATOMY}"
echo "ACCELERATION          : ${ACCELERATION}"
echo "CENTER_FRACTION       : ${CENTER_FRACTION}"
echo "MASK_MODE             : ${MASK_MODE}"
echo "TEST_MODE             : ${TEST_MODE}"
echo "FIXED_MASK_TYPE       : ${FIXED_MASK_TYPE}"
echo "NUM_LOGITS            : ${NUM_LOGITS}"
echo "VN_FILE_SPLIT         : ${VN_FILE_SPLIT}"
echo "UNC_FILE_SPLIT        : ${UNC_FILE_SPLIT}"
echo "BATCH_SIZE            : ${BATCH_SIZE}"
echo "NUM_CASCADES          : ${NUM_CASCADES}"
echo "CHANS                 : ${CHANS}"
echo "POOLS                 : ${POOLS}"
echo "SENS_CHANS            : ${SENS_CHANS}"
echo "SENS_POOLS            : ${SENS_POOLS}"
echo "PRECISION             : ${PRECISION}"
echo "VARNET_DIR            : ${VARNET_DIR}"
echo "DATA_DIR              : ${DATA_DIR}"
echo "MODEL_OUTPUT_DIR      : ${MODEL_OUTPUT_DIR}"
echo "CHECKPOINT_DIR        : ${CHECKPOINT_DIR}"
echo "UNC_CHECKPOINT_DIR    : ${UNC_CHECKPOINT_DIR}"
echo "VN_PREDICTIONS_PATH   : ${VN_PREDICTIONS_PATH}"
echo "SLURM_LOG_FILE        : ${SLURM_LOG_FILE}"
echo "============================================================"

if [ "${MASK_MODE}" = "learnable" ]; then
    TEST_MODE=test
fi

if [ "${RUN_TRAINING}" = "true" ]; then
    srun python3 "${VARNET_DIR}/e2evarnet/runner.py" "${COMMON_ARGS[@]}" "${MASK_ARGS[@]}" "${VN_SPLIT_ARGS[@]}" "${TRAIN_MODE_ARGS[@]}"
fi

VN_CHECKPOINT_PATH=$(resolve_preferred_ckpt "${CHECKPOINT_DIR}")

if [ "${RUN_TESTING}" = "true" ]; then
    srun python3 "${VARNET_DIR}/e2evarnet/runner.py" "${COMMON_ARGS[@]}" "${MASK_ARGS[@]}" --mode "${TEST_MODE}" --test_path "${VN_TEST_PATH}"
fi

if [ "${RUN_EVALUATION}" = "true" ]; then
    srun python3 -m e2evarnet.evaluation --target-path "${VN_TEST_PATH}" --predictions-path "${VN_PREDICTIONS_PATH}"
fi

# NOTE(recon-migration): the uncertainty stages below call runner_uncertainty.py, which was NOT
# migrated to this repo (superseded by conformal/calibrate_lambda.py + deploy_calibration.py).
# They default to false and are kept only for reference -- they will fail if enabled as-is.
if [ "${RUN_UNCERTAINTY_TRAINING}" = "true" ]; then
    srun python3 "${VARNET_DIR}/runner_uncertainty.py" "${UNC_ARGS[@]}" "${MASK_ARGS[@]}" "${UNC_SPLIT_ARGS[@]}" --mode train --varnet_ckpt "${VN_CHECKPOINT_PATH}"
fi

UNC_CHECKPOINT_PATH=$(resolve_preferred_ckpt "${UNC_CHECKPOINT_DIR}")

if [ "${RUN_UNCERTAINTY_CALIBRATION}" = "true" ]; then
    srun python3 "${VARNET_DIR}/runner_uncertainty.py" "${UNC_ARGS[@]}" "${MASK_ARGS[@]}" --mode calibrate --devices 1 --varnet_ckpt "${VN_CHECKPOINT_PATH}" --uncertainty_ckpt "${UNC_CHECKPOINT_PATH}" --output_uncertainty_ckpt "${UNC_CHECKPOINT_PATH}"
fi

if [ "${RUN_UNCERTAINTY_TESTING}" = "true" ]; then
    srun python3 "${VARNET_DIR}/runner_uncertainty.py" "${UNC_ARGS[@]}" "${MASK_ARGS[@]}" --mode "${TEST_MODE}" --varnet_ckpt "${VN_CHECKPOINT_PATH}" --uncertainty_ckpt "${UNC_CHECKPOINT_PATH}" --reconstructions_dir "${VN_PREDICTIONS_PATH}" --test_path "${VN_TEST_PATH}"
fi


# [shaana01@bigpurple-ln2 VarNet]$ squeue -u $USER
#              JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
#           25135401 a100_long e2e_knee shaana01  R 1-04:27:44      1 a100-4039
#           25135396 a100_long e2e_knee shaana01  R 1-04:31:05      1 a100-4034
#           25189608 a100_shor ilias_va shaana01 PD       0:00      1 (Priority)
#           25186995 cpu_mediu Check_DF shaana01  R    1:19:49      1 cn-0024