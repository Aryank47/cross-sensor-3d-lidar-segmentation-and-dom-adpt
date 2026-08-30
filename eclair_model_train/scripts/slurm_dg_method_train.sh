#!/bin/bash
#SBATCH --job-name=dg_method
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/dg_method_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/dg_method_%j.err

set -euo pipefail
set +u
unset LUA_PATH
unset LUA_CPATH
set -u

PROJECT_ROOT="/csehome/m23csa510/lidar_experiments"
CONDA_ROOT="${PROJECT_ROOT}/env/miniconda3"
CODE_ROOT="${PROJECT_ROOT}/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"
DATA_ROOT="/scratch/m23csa510/dales/dales"
OUT_ROOT="/scratch/m23csa510/e0_results"
LOG_ROOT="/scratch/m23csa510/e0_logs"

CONFIG_SRC="${CONFIG_SRC:?submit with --export=ALL,CONFIG_SRC=...,METHOD_TAG=...}"
METHOD_TAG="${METHOD_TAG:?submit with --export=ALL,CONFIG_SRC=...,METHOD_TAG=...}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" /scratch/m23csa510/dales/dales_cache

module purge
module load cuda/11.8 gcc/11 gnu12/12.3.0 openblas/0.3.21
export OPENBLAS_ROOT=/opt/ohpc/pub/libs/gnu12/openblas/0.3.21
export OPENBLAS_INCLUDE="${OPENBLAS_ROOT}/include"
export OPENBLAS_LIB="${OPENBLAS_ROOT}/lib"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${OPENBLAS_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${OPENBLAS_LIB}:${LIBRARY_PATH:-}"
# Two ranks plus four DataLoader workers per rank share 16 allocated CPU cores.
# Avoid giving every process all 16 OpenMP/BLAS threads.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate e0_env
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export DALES_TRAIN_ROOT="${DATA_ROOT}/train"
export DALES_TEST_ROOT="${DATA_ROOT}/test"
export DALES_CACHE_ROOT_DROP_I="/scratch/m23csa510/dales/dales_cache"

cd "${CODE_ROOT}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  RUN_DIR="${RESUME_RUN_DIR}"
  RESUME_FROM="${RUN_DIR}/checkpoints/last.pt"
  if [[ ! -f "${RESUME_FROM}" ]]; then
    echo "Missing resumable checkpoint: ${RESUME_FROM}" >&2
    exit 2
  fi
else
  RUN_DIR="${OUT_ROOT}/dg_${METHOD_TAG}_${SLURM_JOB_ID}"
  RESUME_FROM=""
fi
mkdir -p "${RUN_DIR}"
CONFIG_RUN="${RUN_DIR}/config_resolved.yaml"

python scripts/preflight_dg_method.py \
  --config "${CONFIG_SRC}" \
  --expect-method "${METHOD_TAG}" \
  --full \
  --build-model | tee "${RUN_DIR}/preflight.json"

python - "${CONFIG_SRC}" "${CONFIG_RUN}" "${RUN_DIR}" "${RESUME_FROM}" <<'PY'
import sys, yaml
from pathlib import Path
from src.config_loader import load_yaml
src, dst, out_dir, resume_from = sys.argv[1:]
cfg = load_yaml(src)
cfg["run"]["out_dir"] = out_dir
if resume_from:
    cfg["run"]["resume_from"] = resume_from
else:
    cfg["run"].pop("resume_from", None)
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

git rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>/dev/null || true
git status --porcelain > "${RUN_DIR}/git_status.txt" 2>/dev/null || true
nvidia-smi

MASTER_PORT=$((10000 + (SLURM_JOB_ID % 50000)))
export MASTER_PORT
srun --unbuffered python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${SLURM_GPUS_ON_NODE:-2}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
  train.py --config "${CONFIG_RUN}"

echo "[done] full ${METHOD_TAG}: ${RUN_DIR}"
