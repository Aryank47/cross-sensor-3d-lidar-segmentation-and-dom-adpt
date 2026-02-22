#!/bin/bash
#SBATCH --job-name=e0_ECLAIR_REPRO
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/eclair_train_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/eclair_train_%j.err

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

# This should be the folder that contains:
#   train.py, configs/e0_eclair.yaml, scripts/precompute_eclair_cache.py, src/
CODE_ROOT="$PROJECT_ROOT/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"

# ECLAIR dataset root must contain: labels.json and pointclouds/
ECLAIR_DIR="$PROJECT_ROOT/datasets/eclair"

# Scratch outputs
SCRATCH_OUTROOT="/scratch/m23csa510/e0_results"
SCRATCH_LOGROOT="/scratch/m23csa510/e0_logs"
CACHE_ROOT="/scratch/m23csa510/eclair_cache"

mkdir -p "${SCRATCH_OUTROOT}" "${SCRATCH_LOGROOT}" "${CACHE_ROOT}"

########################
# 1) MODULES / CUDA
########################
module purge
module load cuda/11.8 gcc/11 gnu12/12.3.0 openblas/0.3.21

########################
# 2) BLAS / THREADING / CUDA
########################
export OPENBLAS_ROOT=/opt/ohpc/pub/libs/gnu12/openblas/0.3.21
export OPENBLAS_INCLUDE="${OPENBLAS_ROOT}/include"
export OPENBLAS_LIB="${OPENBLAS_ROOT}/lib"

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${OPENBLAS_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${OPENBLAS_LIB}:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# NCCL sanity
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

########################
# 3) CONDA ENV
########################
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate e0_env

# Make conda env take priority (and avoid ~/.local bin shadowing)
# export PYTHONNOUSERSITE=1
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# # Optionally remove ~/.local/bin entirely for the job
# export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/.local/bin" | paste -sd: -)"
# hash -r


echo "Python: $(which python)"
python -V
echo "CUDA_HOME: ${CUDA_HOME}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

nvidia-smi

echo "which python:  $(which python)"
echo "python exec:   $(python -c 'import sys; print(sys.executable)')"
echo "which torchrun: $(which torchrun || true)"
echo "torchrun head:  $(head -n 1 $(which torchrun) 2>/dev/null || true)"

python -c "import torch; import sys; print('torch', torch.__version__, 'py', sys.executable)"
python -c "import MinkowskiEngine as ME; import sys; print('ME_OK', ME.__file__, 'py', sys.executable)"


########################
# 4) ENV VARS REQUIRED BY YAML
########################
export ECLAIR_ROOT="${ECLAIR_DIR}"
export ECLAIR_CACHE_ROOT="${CACHE_ROOT}"

########################
# 5) MOVE TO CODE
########################
cd "${CODE_ROOT}"
echo "CWD: $(pwd)"

########################
# 7) CREATE JOB RUN DIR + JOB-SPECIFIC YAML
########################
RUN_DIR="${SCRATCH_OUTROOT}/e0_eclair_repro_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

CONFIG_SRC="configs/e0_eclair_dropI.yaml"
CONFIG_RUN="${RUN_DIR}/e0_eclair_${SLURM_JOB_ID}.yaml"

python - <<PY
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("${CONFIG_SRC}").read_text())
cfg.setdefault("run", {})
cfg["run"]["out_dir"] = "${RUN_DIR}"
Path("${CONFIG_RUN}").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("[config] wrote:", "${CONFIG_RUN}")
PY

echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] Using config: ${CONFIG_RUN}"

# Optional: record git commit if this is a git repo
if command -v git >/dev/null 2>&1 && [ -d .git ]; then
  git rev-parse HEAD | tee "${RUN_DIR}/git_commit.txt" || true
  git status --porcelain | tee "${RUN_DIR}/git_status.txt" || true
fi

########################
# 8) TRAIN (2-GPU DDP)
########################
# Pick a deterministic-ish free port (avoid collisions)
MASTER_PORT=$((10000 + (${SLURM_JOB_ID} % 50000)))
export MASTER_PORT

echo "[ddp] MASTER_PORT=${MASTER_PORT}"
echo "[ddp] nproc=${SLURM_GPUS_ON_NODE:-2}"

srun --unbuffered python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${SLURM_GPUS_ON_NODE:-2}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
  train.py --config "${CONFIG_RUN}"

echo "[done] Outputs written to: ${RUN_DIR}"
echo "  - ${RUN_DIR}/metrics.csv"
echo "  - ${RUN_DIR}/checkpoints/{best.pt,last.pt,...}"
echo "  - ${RUN_DIR}/test_metrics.json"
