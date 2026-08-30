#!/bin/bash
#SBATCH --job-name=dg_candidate_audit
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/dg_candidate_audit_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/dg_candidate_audit_%j.err

set -euo pipefail

set +u
unset LUA_PATH
unset LUA_CPATH
set -u

PROJECT_ROOT="/csehome/m23csa510/lidar_experiments"
CONDA_ROOT="${PROJECT_ROOT}/env/miniconda3"
CODE_ROOT="${PROJECT_ROOT}/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"

ECLAIR_DIR="${PROJECT_ROOT}/datasets/eclair"
DALES_ROOT="/scratch/m23csa510/dales/dales"
ECLAIR_CACHE_DIR="/scratch/m23csa510/eclair_cache"
DALES_CACHE_DIR="/scratch/m23csa510/dales/dales_cache"
LOG_ROOT="/scratch/m23csa510/e0_logs"
OUT_ROOT="/scratch/m23csa510/e0_results"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}" "${ECLAIR_CACHE_DIR}" "${DALES_CACHE_DIR}"

module purge
module load cuda/11.8 gcc/11 gnu12/12.3.0 openblas/0.3.21

export OPENBLAS_ROOT=/opt/ohpc/pub/libs/gnu12/openblas/0.3.21
export OPENBLAS_INCLUDE="${OPENBLAS_ROOT}/include"
export OPENBLAS_LIB="${OPENBLAS_ROOT}/lib"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${OPENBLAS_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${OPENBLAS_LIB}:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate e0_env
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export ECLAIR_ROOT="${ECLAIR_DIR}"
export ECLAIR_CACHE_ROOT="${ECLAIR_CACHE_DIR}"
export DALES_TRAIN_ROOT="${DALES_ROOT}/train"
export DALES_TEST_ROOT="${DALES_ROOT}/test"
export DALES_CACHE_ROOT_DROP_I="${DALES_CACHE_DIR}"

cd "${CODE_ROOT}"

AUDIT_CONFIG="${AUDIT_CONFIG:-configs/dg_candidate_audit.yaml}"
SAMPLES="${SAMPLES:-50}"
RUN_DIR="${RUN_DIR_OVERRIDE:-${OUT_ROOT}/dg_candidate_audit_${SLURM_JOB_ID}}"
mkdir -p "${RUN_DIR}"

python -c "import torch, MinkowskiEngine as ME; print('torch', torch.__version__); print('ME', ME.__file__)"
python scripts/audit_dg_candidates.py \
  --audit-config "${AUDIT_CONFIG}" \
  --samples "${SAMPLES}" \
  --out-dir "${RUN_DIR}"

echo "[done] ${RUN_DIR}/summary.json"
