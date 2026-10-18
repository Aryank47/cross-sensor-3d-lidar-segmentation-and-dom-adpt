#!/bin/bash
#SBATCH --job-name=diag_repair
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/diag_repair_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/diag_repair_%j.err

set -eo pipefail
unset LUA_PATH LUA_CPATH
module purge
module load cuda/11.8 gcc/11 gnu12/12.3.0 openblas/0.3.21

PROJECT_ROOT="/csehome/m23csa510/lidar_experiments"
CODE_ROOT="${PROJECT_ROOT}/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"
source "${PROJECT_ROOT}/env/miniconda3/etc/profile.d/conda.sh"
conda activate e0_env
set -u

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export OPENBLAS_ROOT=/opt/ohpc/pub/libs/gnu12/openblas/0.3.21
export OPENBLAS_INCLUDE="${OPENBLAS_ROOT}/include"
export OPENBLAS_LIB="${OPENBLAS_ROOT}/lib"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${OPENBLAS_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${OPENBLAS_LIB}:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export ECLAIR_ROOT="${PROJECT_ROOT}/datasets/eclair"
export ECLAIR_CACHE_ROOT=/scratch/m23csa510/eclair_cache
export DALES_TRAIN_ROOT=/scratch/m23csa510/dales/dales/train
export DALES_TEST_ROOT=/scratch/m23csa510/dales/dales/test
export DALES_CACHE_ROOT_DROP_I=/scratch/m23csa510/dales/dales_cache

cd "${CODE_ROOT}"
REPAIR_SCRIPT="${CODE_ROOT}/repair_training_replay.py"
if [[ ! -f "${REPAIR_SCRIPT}" ]]; then
    REPAIR_SCRIPT="${CODE_ROOT}/diagnostic_repair_20261018/repair_training_replay.py"
fi
test -f "${REPAIR_SCRIPT}"
echo "Training replay repair only; no dataset scan or model inference."
echo "Job ${SLURM_JOB_ID:-unknown} on $(hostname)"

exec python -u "${REPAIR_SCRIPT}" \
    --parent-out /scratch/m23csa510/project_diagnosis_20261017
