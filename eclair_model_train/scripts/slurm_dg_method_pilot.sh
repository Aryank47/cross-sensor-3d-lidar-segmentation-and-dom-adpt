#!/bin/bash
#SBATCH --job-name=dg_pilot
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/dg_pilot_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/dg_pilot_%j.err

set -euo pipefail
set +u
unset LUA_PATH
unset LUA_CPATH
set -u

PROJECT_ROOT="/csehome/m23csa510/lidar_experiments"
CONDA_ROOT="${PROJECT_ROOT}/env/miniconda3"
CODE_ROOT="${PROJECT_ROOT}/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"
OUT_ROOT="/scratch/m23csa510/e0_results"
LOG_ROOT="/scratch/m23csa510/e0_logs"

CONFIG_SRC="${CONFIG_SRC:?submit with --export=ALL,DATASET_TAG=...,CONFIG_SRC=...,METHOD_TAG=...}"
METHOD_TAG="${METHOD_TAG:?submit with --export=ALL,DATASET_TAG=...,CONFIG_SRC=...,METHOD_TAG=...}"
DATASET_TAG="${DATASET_TAG:?submit with --export=ALL,DATASET_TAG=...,CONFIG_SRC=...,METHOD_TAG=...}"

case "${DATASET_TAG}" in
  dales)
    FROZEN_BASE="configs/m0_dales_frozen_29364.yaml"
    MINIMUM_BATCHES_PER_RANK=150
    ;;
  eclair)
    FROZEN_BASE="configs/m0_eclair_frozen_29306.yaml"
    MINIMUM_BATCHES_PER_RANK=200
    ;;
  *)
    echo "DATASET_TAG must be dales or eclair; got ${DATASET_TAG}" >&2
    exit 2
    ;;
esac

mkdir -p \
  "${OUT_ROOT}" \
  "${LOG_ROOT}" \
  /scratch/m23csa510/dales/dales_cache \
  /scratch/m23csa510/eclair_cache

module purge
module load cuda/11.8 gcc/11 gnu12/12.3.0 openblas/0.3.21
export OPENBLAS_ROOT=/opt/ohpc/pub/libs/gnu12/openblas/0.3.21
export OPENBLAS_INCLUDE="${OPENBLAS_ROOT}/include"
export OPENBLAS_LIB="${OPENBLAS_ROOT}/lib"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${OPENBLAS_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${OPENBLAS_LIB}:${LIBRARY_PATH:-}"
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

export DALES_TRAIN_ROOT="/scratch/m23csa510/dales/dales/train"
export DALES_TEST_ROOT="/scratch/m23csa510/dales/dales/test"
export DALES_CACHE_ROOT_DROP_I="/scratch/m23csa510/dales/dales_cache"
export ECLAIR_ROOT="/csehome/m23csa510/lidar_experiments/datasets/eclair"
export ECLAIR_CACHE_ROOT="/scratch/m23csa510/eclair_cache"

cd "${CODE_ROOT}"
RUN_DIR="${OUT_ROOT}/pilot_${DATASET_TAG}_${METHOD_TAG}_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
CONFIG_FINAL="${RUN_DIR}/config_resolved.yaml"
CONFIG_PHASE1="${RUN_DIR}/config_phase1.yaml"
CONFIG_PHASE2="${RUN_DIR}/config_phase2_resume.yaml"

python scripts/preflight_dg_method.py \
  --config "${CONFIG_SRC}" \
  --frozen-base "${FROZEN_BASE}" \
  --dataset "${DATASET_TAG}" \
  --expect-method "${METHOD_TAG}" \
  --pilot \
  --build-model | tee "${RUN_DIR}/preflight.json"

python - "${CONFIG_SRC}" "${CONFIG_FINAL}" "${CONFIG_PHASE1}" "${RUN_DIR}" <<'PY'
import sys
from pathlib import Path

import yaml

from src.config_loader import load_yaml

src, final_path, phase1_path, out_dir = sys.argv[1:]
cfg = load_yaml(src)
cfg["run"]["out_dir"] = out_dir
cfg["run"].pop("resume_from", None)
Path(final_path).write_text(yaml.safe_dump(cfg, sort_keys=False))

phase1 = dict(cfg)
phase1["run"] = dict(cfg["run"])
phase1["epochs"] = int(cfg["run"]["resume_split_epoch"])
Path(phase1_path).write_text(yaml.safe_dump(phase1, sort_keys=False))
PY

git rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>/dev/null || true
git status --porcelain > "${RUN_DIR}/git_status.txt" 2>/dev/null || true
git diff --binary > "${RUN_DIR}/source_worktree.patch" 2>/dev/null || true
find train.py src scripts configs -type f \
  \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_DIR}/source_manifest.sha256"
tar -czf "${RUN_DIR}/source_snapshot.tar.gz" train.py src scripts configs
nvidia-smi

MASTER_PORT=$((10000 + (SLURM_JOB_ID % 49000)))
export MASTER_PORT
srun --unbuffered python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${SLURM_GPUS_ON_NODE:-2}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
  train.py --config "${CONFIG_PHASE1}"

RESUME_FROM="${RUN_DIR}/checkpoints/last.pt"
test -f "${RESUME_FROM}"
python - "${CONFIG_FINAL}" "${CONFIG_PHASE2}" "${RESUME_FROM}" "${RUN_DIR}/diagnostic_complete.json" <<'PY'
import sys
from pathlib import Path

import yaml

from src.config_loader import load_yaml

src, dst, resume_from, stale_complete = sys.argv[1:]
cfg = load_yaml(src)
cfg["run"]["resume_from"] = resume_from
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
Path(stale_complete).unlink(missing_ok=True)
PY

MASTER_PORT=$((MASTER_PORT + 1))
export MASTER_PORT
srun --unbuffered python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${SLURM_GPUS_ON_NODE:-2}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
  train.py --config "${CONFIG_PHASE2}"

test -f "${RUN_DIR}/diagnostic_complete.json"
test -f "${RUN_DIR}/resume_state.json"
GPU_MEMORY_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -n 1)
python scripts/check_dg_pilot.py \
  --run-dir "${RUN_DIR}" \
  --dataset "${DATASET_TAG}" \
  --method "${METHOD_TAG}" \
  --minimum-batches-per-rank "${MINIMUM_BATCHES_PER_RANK}" \
  --gpu-memory-mib "${GPU_MEMORY_MIB}" \
  --wall-limit-hours 36 \
  --storage-budget-gb 10

echo "[done] ${DATASET_TAG} ${METHOD_TAG} pilot: ${RUN_DIR}"
