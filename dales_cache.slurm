#!/bin/bash
#SBATCH --job-name=dales_cache
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/dales_cache_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/dales_cache_%j.err

set -euo pipefail

# --- Workaround: avoid breaking Lmod's Lua environment ---
set +u
unset LUA_PATH
unset LUA_CPATH
set -u

########################
# 0) PATHS (EDIT THESE)
########################
PROJECT_ROOT="/csehome/m23csa510/lidar_experiments"
CONDA_ROOT="$PROJECT_ROOT/env/miniconda3"

# Folder that contains: scripts/precompute_dales_cache.py and src/
CODE_ROOT="$PROJECT_ROOT/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"

# DALES dataset roots
DALES_TRAIN_DIR="/scratch/m23csa510/dales/dales/train"
DALES_TEST_DIR="/scratch/m23csa510/dales/dales/test"

# Cache output root
DALES_CACHE_ROOT="/scratch/m23csa510/dales/dales_cache"

mkdir -p /scratch/m23csa510/e0_logs "${DALES_CACHE_ROOT}"

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
# 4) ENV VARS REQUIRED BY YAML
########################
export DALES_TRAIN_ROOT="${DALES_TRAIN_DIR}"
export DALES_TEST_ROOT="${DALES_TEST_DIR}"
export DALES_CACHE_ROOT_DROP_I="${DALES_CACHE_ROOT}"

########################
# 5) CWD
########################
cd "${CODE_ROOT}"
echo "CWD: $(pwd)"

# IMPORTANT: this script expects a DALES config with:
# data.dataset: dales
# data.dales_train_root: ${DALES_TRAIN_ROOT}
# data.dales_test_root: ${DALES_TEST_ROOT}
# data.cache_root: ${DALES_CACHE_ROOT}
python -m scripts.precompute_dales_cache \
  --config "configs/e0_dales_dropI.yaml"

echo "[done] DALES cache at: ${DALES_CACHE_ROOT}"
echo "[sanity] cached train files: $(find ${DALES_CACHE_ROOT} -type f -path '*/dales*/*train/*.pt' | wc -l)"
echo "[sanity] cached val files:   $(find ${DALES_CACHE_ROOT} -type f -path '*/dales*/*val/*.pt'   | wc -l)"
echo "[sanity] cached test files:  $(find ${DALES_CACHE_ROOT} -type f -path '*/dales*/*test/*.pt'  | wc -l)"
