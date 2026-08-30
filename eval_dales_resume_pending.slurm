#!/bin/bash
#SBATCH --job-name=e0_eval_dales
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/e0_eval_dales_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/e0_eval_dales_%j.err

set -euo pipefail
# --- Workaround: avoid breaking Lmod's Lua environment ---
set +u
unset LUA_PATH
unset LUA_CPATH
set -u

# -----------------------------
# 0) Paths / Env
# -----------------------------
PROJECT_ROOT="/csehome/m23csa510/lidar_experiments/cross-sensor-3d-lidar-segmentation-and-dom-adpt"
CONDA_ROOT="/csehome/m23csa510/lidar_experiments/env/miniconda3"
ENV_NAME="e0_env"

DALES_ROOT="/scratch/m23csa510/dales/dales"          # as you showed
DALES_SPLIT="all"                              # change to: all | test | train
DALES_PATH="${DALES_ROOT}/${DALES_SPLIT}"
ECLAIR_ROOT="/csehome/m23csa510/lidar_experiments/datasets/eclair"
ECLAIR_DIR="/csehome/m23csa510/lidar_experiments/datasets/eclair"
CACHE_ROOT="/scratch/m23csa510/eclair_cache"
# Your E0 config (the one used to train ECLAIR)
CONFIG="${PROJECT_ROOT}/eclair_model_train/configs/e0_eclair.yaml"

# Best checkpoint from ECLAIR training
# CHANGE THIS to your actual best.pt path
CKPT="/scratch/m23csa510/e0_results/e0_eclair_repro_23978/checkpoints/best.pt"

# Common mapping files
MAP_ECLAIR_COMMON="${PROJECT_ROOT}/eclair_model_train/configs/mapping_eclair_to_common.yaml"
MAP_DALES_COMMON="${PROJECT_ROOT}/eclair_model_train/configs/mapping_dales_to_common.yaml"
MAP_PRED_TOCOMMON="${PROJECT_ROOT}/eclair_model_train/configs/train_id_to_common_id.yaml"

# Output directory for DALES eval artifacts
RUN_ROOT="/scratch/m23csa510/dales_eval_runs"
RUN_ID="dales_common_${DALES_SPLIT}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
OUT_DIR="/scratch/m23csa510/dales_eval_runs/dales_common_all_20251230_175739_24022"

mkdir -p "${OUT_DIR}"

echo "=================================================="
echo "[info] PROJECT_ROOT = ${PROJECT_ROOT}"
echo "[info] DALES_PATH    = ${DALES_PATH}"
echo "[info] CONFIG        = ${CONFIG}"
echo "[info] CKPT          = ${CKPT}"
echo "[info] OUT_DIR       = ${OUT_DIR}"
echo "=================================================="

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

# -----------------------------
# 1) Activate env
# -----------------------------
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${PROJECT_ROOT}"

python -c "import torch; print('[info] torch:', torch.__version__, 'cuda:', torch.version.cuda, 'avail:', torch.cuda.is_available())"
python -c "import MinkowskiEngine as ME; print('[info] MinkowskiEngine:', ME.__version__)"

# -----------------------------
# 2) Compute ECLAIR reference intensity quantiles
#    (used for quantile matching)
# -----------------------------
export ECLAIR_ROOT="${ECLAIR_DIR}"
export ECLAIR_CACHE_ROOT="${CACHE_ROOT}"

REF_JSON="${OUT_DIR}/intensity_ref_eclair_train.json"

python - <<PY
from pathlib import Path
paths = [
  Path("${CONFIG}"),
  Path("${CKPT}"),
  Path("${MAP_ECLAIR_COMMON}"),
  Path("${MAP_DALES_COMMON}"),
  Path("${MAP_PRED_TOCOMMON}"),
  Path("${DALES_PATH}"),
]
for p in paths:
  print("[check]", p, "exists=", p.exists())
PY


echo "=================================================="
echo "[step] Compute ECLAIR intensity reference quantiles -> ${REF_JSON}"
echo "=================================================="

python -m eclair_model_train.src.compute_intensity_reference \
  --eclair_root "${ECLAIR_ROOT}" \
  --split train \
  --meta_filename labels.json \
  --allowed_review_categories approved rejected \
  --out "${REF_JSON}"

# -----------------------------
# 3) Run DALES eval (baseline + preproc) with comparison
# -----------------------------
echo "=================================================="
echo "[step] DALES eval baseline + preprocessed"
echo "=================================================="
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6"

# python -m eclair_model_train.src.eval_dales_common \
#   --config "${CONFIG}" \
#   --ckpt "${CKPT}" \
#   --dales_root "${DALES_PATH}" \
#   --mapping_eclair_to_common "${MAP_ECLAIR_COMMON}" \
#   --mapping_dales_to_common "${MAP_DALES_COMMON}" \
#   --out_dir "${OUT_DIR}" \
#   --run_both \
#   --intensity_ref_json "${REF_JSON}" \
#   --preproc_intensity_mode quantile_match \
#   --use_height_xy_cap \
#   --xy_cell_size_m 2.0 \
#   --height_bins 5 \
#   --caps_per_bin 4000 3000 2500 2500 2500 \
#   --dales_patch_size_m 100 \
#   --dales_patch_stride_m 100 \
#   --dales_min_patch_points 2000 \
#   --dales_max_patches_per_cloud 0


srun --cpu-bind=cores python -m eclair_model_train.src.eval_dales_preproc_sweep \
  --config "${CONFIG}" \
  --dales_root "${DALES_PATH}" \
  --ckpt "${CKPT}" \
  --mapping_dales_native_to_common "${MAP_DALES_COMMON}" \
  --mapping_eclair_native_to_common "${MAP_ECLAIR_COMMON}" \
  --include_slow_modes \
  --eclair_intensity_ref_json "${REF_JSON}" \
  --mapping_pred_to_common "${MAP_PRED_TOCOMMON}" \
  --out_dir "${OUT_DIR}" \
  --mode staged --max_runs 120 \
  --plan sweep \
  --try_import_label_maps \
  # --pending_specs_json /scratch/m23csa510/dales_eval_runs/dales_common_all_20251230_175739_24022/pending_specs.json

# --plan decisive_returns \

echo "=================================================="
echo "[done] Outputs written to: ${OUT_DIR}"
echo "[done] Summary: ${OUT_DIR}/summary.json"
echo "=================================================="
