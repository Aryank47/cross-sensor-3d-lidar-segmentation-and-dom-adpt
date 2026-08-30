#!/bin/bash
#SBATCH --job-name=dg_diag
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/dg_diag_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/dg_diag_%j.err

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate e0_env
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export DALES_TRAIN_ROOT="${DATA_ROOT}/train"
export DALES_TEST_ROOT="${DATA_ROOT}/test"
export DALES_CACHE_ROOT_DROP_I="/scratch/m23csa510/dales/dales_cache"

cd "${CODE_ROOT}"
RUN_DIR="${OUT_ROOT}/diagnostic_${METHOD_TAG}_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
CONFIG_RUN="${RUN_DIR}/config_resolved.yaml"

python scripts/preflight_dg_method.py \
  --config "${CONFIG_SRC}" \
  --expect-method "${METHOD_TAG}" \
  --build-model | tee "${RUN_DIR}/preflight.json"

python - "${CONFIG_SRC}" "${CONFIG_RUN}" "${RUN_DIR}" <<'PY'
import sys, yaml
from pathlib import Path
from src.config_loader import load_yaml
src, dst, out_dir = sys.argv[1:]
cfg = load_yaml(src)
cfg["run"]["out_dir"] = out_dir
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

python -c "import torch, MinkowskiEngine as ME; print('torch', torch.__version__); print('ME', ME.__file__)"
python train.py --config "${CONFIG_RUN}"
test -f "${RUN_DIR}/diagnostic_complete.json"
echo "[done] diagnostic ${METHOD_TAG}: ${RUN_DIR}"
