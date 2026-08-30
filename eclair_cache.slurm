#!/bin/bash
#SBATCH --job-name=eclair_cache
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/eclair_cache_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/eclair_cache_%j.err

set -euo pipefail

# --- Workaround: avoid breaking Lmod's Lua environment ---
set +u
unset LUA_PATH
unset LUA_CPATH
set -u

PROJECT_ROOT="/csehome/m23csa510/lidar_experiments"
REPO_ROOT="$PROJECT_ROOT/cross-sensor-3d-lidar-segmentation-and-dom-adpt"
CODE_ROOT="$REPO_ROOT/eclair_model_train"  # where precompute script lives
CONDA_ROOT="$PROJECT_ROOT/env/miniconda3"
DATA_ROOT="$PROJECT_ROOT/datasets"

# Logs
mkdir -p /scratch/m23csa510/e0_logs

########################
# 1) MODULES / CUDA
########################
module purge
module load cuda/11.8 gcc/11 gnu12/12.3.0 openblas/0.3.21

########################
# 2) BLAS / THREADING
########################
export OPENBLAS_ROOT=/opt/ohpc/pub/libs/gnu12/openblas/0.3.21
export OPENBLAS_INCLUDE="${OPENBLAS_ROOT}/include"
export OPENBLAS_LIB="${OPENBLAS_ROOT}/lib"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${OPENBLAS_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${OPENBLAS_LIB}:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

########################
# 3) CONDA ENV
########################
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate e0_env

echo "Python: $(which python)"
python -V
echo "CUDA_HOME: ${CUDA_HOME}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

########################
# 4) PATHS
########################
export ECLAIR_DIR="${DATA_ROOT}/eclair"

# Where cache will be written
CACHE_ROOT="/scratch/m23csa510/eclair_cache"
mkdir -p "${CACHE_ROOT}"

########################
# 5) CWD
########################
cd "${CODE_ROOT}"
echo "CWD: $(pwd)"

python -m scripts.precompute_eclair_cache \
  --eclair_root "${ECLAIR_DIR}" \
  --cache_root "${CACHE_ROOT}" \
  --splits train val test

echo "[done] Raw cache at: ${CACHE_ROOT}"
echo "[sanity] train files: $(ls -1 ${CACHE_ROOT}/train | wc -l)"
echo "[sanity] val files:   $(ls -1 ${CACHE_ROOT}/val   | wc -l)"
echo "[sanity] test files:  $(ls -1 ${CACHE_ROOT}/test  | wc -l)"
