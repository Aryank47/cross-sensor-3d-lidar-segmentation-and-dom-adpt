#!/bin/bash

#SBATCH --job-name=cross_eval_dg
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/%x_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/%x_%j.err

set -euo pipefail

# --- Workaround: avoid breaking Lmod's Lua environment ---
set +u
unset LUA_PATH
unset LUA_CPATH
set -u

########################
# 0) PATHS
########################
PROJECT_ROOT="/csehome/m23csa510/lidar_experiments"
CONDA_ROOT="${PROJECT_ROOT}/env/miniconda3"

CODE_ROOT="${PROJECT_ROOT}/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"

ECLAIR_DIR="${PROJECT_ROOT}/datasets/eclair"

DATA_ROOT="/scratch/m23csa510/dales"
DALES_TRAIN_DIR="${DATA_ROOT}/dales/train"
DALES_TEST_DIR="${DATA_ROOT}/dales/test"

SCRATCH_OUTROOT="/scratch/m23csa510/e0_results"
SCRATCH_LOGROOT="/scratch/m23csa510/e0_logs"
CROSS_EVAL_ROOT="/scratch/m23csa510/cross_eval_runs/dg_methods_120m"

ECLAIR_CACHE_ROOT="/scratch/m23csa510/eclair_cache"
DALES_CACHE_ROOT_DROP_I="/scratch/m23csa510/dales/dales_cache"

mkdir -p "${SCRATCH_OUTROOT}" "${SCRATCH_LOGROOT}" "${CROSS_EVAL_ROOT}"
mkdir -p "${ECLAIR_CACHE_ROOT}" "${DALES_CACHE_ROOT_DROP_I}"

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

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

########################
# 3) CONDA ENV
########################
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate e0_env

export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

echo "============================================================"
echo "DG cross-domain common-space pointwise evaluator"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================================"

echo "Python: $(which python)"
python -V
echo "CUDA_HOME: ${CUDA_HOME}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

nvidia-smi

echo "which python:  $(which python)"
echo "python exec:   $(python -c 'import sys; print(sys.executable)')"

python -c "import torch, sys; print('torch', torch.__version__, 'py', sys.executable)"
python -c "import MinkowskiEngine as ME, sys; print('ME_OK', ME.__file__, 'py', sys.executable)"

########################
# 4) ENV VARS REQUIRED BY YAML
########################
export ECLAIR_ROOT="${ECLAIR_DIR}"
export ECLAIR_CACHE_ROOT

export DALES_TRAIN_ROOT="${DALES_TRAIN_DIR}"
export DALES_TEST_ROOT="${DALES_TEST_DIR}"
export DALES_CACHE_ROOT_DROP_I

########################
# 5) MOVE TO CODE
########################
cd "${CODE_ROOT}"
echo "CWD: $(pwd)"

########################
# 6) SELECT METHOD / EVAL CASE
########################
# Required:
#   METHOD_TAG=ocons|bev_als|baseline
#   CASE=dales_to_dales|dales_to_eclair|eclair_to_dales|eclair_to_eclair

METHOD_TAG="${METHOD_TAG:?Set METHOD_TAG to ocons, bev_als, or baseline}"
CASE="${CASE:?Set CASE to a supported source_to_target direction}"

TARGET_SPLIT="${TARGET_SPLIT:-test}"
LIMIT_TILES="${LIMIT_TILES:-0}"
WINDOW_M="${WINDOW_M:-120}"
STRIDE_M="${STRIDE_M:-${WINDOW_M}}"
MAX_VOX="${MAX_VOX:-180000}"
MAX_SPLIT_DEPTH="${MAX_SPLIT_DEPTH:-8}"
AMP_FLAG="${AMP_FLAG:-0}"
EVAL_TAG="${EVAL_TAG:-canonical120m}"

if [[ "${WINDOW_M}" != "120" || "${STRIDE_M}" != "120" ]]; then
  echo "Refusing non-canonical evaluation: WINDOW_M=${WINDOW_M}, STRIDE_M=${STRIDE_M}"
  echo "The established cross-evaluation protocol requires 120 m / 120 m."
  exit 2
fi

if [[ "${AMP_FLAG}" == "1" ]]; then
  AMP_ARGS=(--amp)
else
  AMP_ARGS=()
fi

case "${METHOD_TAG}" in
  ocons)
    DALES_RUN_DIR_DEFAULT="${SCRATCH_OUTROOT}/dg_dales_ocons_39769"
    ECLAIR_RUN_DIR_DEFAULT="${SCRATCH_OUTROOT}/dg_eclair_ocons_39770"
    ;;
  bev_als)
    DALES_RUN_DIR_DEFAULT="${SCRATCH_OUTROOT}/dg_dales_bev_als_39797"
    ECLAIR_RUN_DIR_DEFAULT="${SCRATCH_OUTROOT}/dg_eclair_bev_als_39798"
    ;;
  baseline)
    DALES_RUN_DIR_DEFAULT="${SCRATCH_OUTROOT}/e0_dales_train_29364"
    ECLAIR_RUN_DIR_DEFAULT="${SCRATCH_OUTROOT}/e0_eclair_repro_29306"
    ;;
  *)
    echo "Unsupported METHOD_TAG=${METHOD_TAG}; expected ocons, bev_als, or baseline"
    exit 2
    ;;
esac

DALES_RUN_DIR="${DALES_RUN_DIR:-${DALES_RUN_DIR_DEFAULT}}"
ECLAIR_RUN_DIR="${ECLAIR_RUN_DIR:-${ECLAIR_RUN_DIR_DEFAULT}}"

DALES_CKPT="${DALES_CKPT:-${DALES_RUN_DIR}/checkpoints/best.pt}"
DALES_CFG="${DALES_CFG:-${DALES_RUN_DIR}/config_resolved.json}"

ECLAIR_CKPT="${ECLAIR_CKPT:-${ECLAIR_RUN_DIR}/checkpoints/best.pt}"
ECLAIR_CFG="${ECLAIR_CFG:-${ECLAIR_RUN_DIR}/config_resolved.json}"

# Canonical baseline target configurations are deliberately fixed for all methods.
ECLAIR_TARGET_CFG="${ECLAIR_TARGET_CFG:-${SCRATCH_OUTROOT}/e0_eclair_repro_29306/config_resolved.json}"
DALES_TARGET_CFG="${DALES_TARGET_CFG:-${SCRATCH_OUTROOT}/e0_dales_train_29364/config_resolved.json}"

MAP_ECLAIR_TRAIN_TO_COMMON="${MAP_ECLAIR_TRAIN_TO_COMMON:-configs/train_id_to_common_id.yaml}"
MAP_DALES_TRAIN_TO_COMMON="${MAP_DALES_TRAIN_TO_COMMON:-configs/train_id_to_common_dales.yaml}"
MAP_ECLAIR_NATIVE_TO_COMMON="${MAP_ECLAIR_NATIVE_TO_COMMON:-configs/mapping_eclair_to_common.yaml}"
MAP_DALES_NATIVE_TO_COMMON="${MAP_DALES_NATIVE_TO_COMMON:-configs/mapping_dales_to_common.yaml}"

echo "METHOD_TAG=${METHOD_TAG}"
echo "CASE=${CASE}"
echo "TARGET_SPLIT=${TARGET_SPLIT}"
echo "LIMIT_TILES=${LIMIT_TILES}"
echo "WINDOW_M=${WINDOW_M}"
echo "STRIDE_M=${STRIDE_M}"
echo "MAX_VOX=${MAX_VOX}"
echo "MAX_SPLIT_DEPTH=${MAX_SPLIT_DEPTH}"
echo "AMP_FLAG=${AMP_FLAG}"
echo "EVAL_TAG=${EVAL_TAG}"

########################
# 7) OPTIONAL DALES TEST MANIFEST
########################
DALES_TEST_MANIFEST="${DALES_TEST_MANIFEST:-}"

if [[ -z "${DALES_TEST_MANIFEST}" ]]; then
  if [[ -f "/scratch/m23csa510/dales/splits/test.txt" ]]; then
    DALES_TEST_MANIFEST="/scratch/m23csa510/dales/splits/test.txt"
  elif [[ -f "/scratch/m23csa510/dales/manifests/test.txt" ]]; then
    DALES_TEST_MANIFEST="/scratch/m23csa510/dales/manifests/test.txt"
  fi
fi

EXTRA_DALES_MANIFEST_ARGS=()
if [[ "${TARGET_SPLIT}" == "test" ]]; then
  if [[ -n "${DALES_TEST_MANIFEST}" && -f "${DALES_TEST_MANIFEST}" ]]; then
    EXTRA_DALES_MANIFEST_ARGS=(--target_manifest "${DALES_TEST_MANIFEST}")
    echo "Using DALES_TEST_MANIFEST=${DALES_TEST_MANIFEST}"
  else
    echo "No DALES test manifest found/provided; evaluator will use dales_test_root from target config."
  fi
else
  echo "TARGET_SPLIT=${TARGET_SPLIT}; not using DALES test manifest."
fi

########################
# 8) RESOLVE CASE-SPECIFIC ARGS
########################
case "${CASE}" in
  dales_to_dales)
    CKPT="${DALES_CKPT}"
    SOURCE_CFG="${DALES_CFG}"
    TARGET_CFG="${DALES_TARGET_CFG}"
    SOURCE_DOMAIN="dales"
    TARGET_DOMAIN="dales"
    SOURCE_TRAIN_TO_COMMON="${MAP_DALES_TRAIN_TO_COMMON}"
    SOURCE_NATIVE_TO_COMMON="${MAP_DALES_NATIVE_TO_COMMON}"
    TARGET_NATIVE_TO_COMMON="${MAP_DALES_NATIVE_TO_COMMON}"
    MANIFEST_ARGS=("${EXTRA_DALES_MANIFEST_ARGS[@]}")
    ;;
  dales_to_eclair)
    CKPT="${DALES_CKPT}"
    SOURCE_CFG="${DALES_CFG}"
    TARGET_CFG="${ECLAIR_TARGET_CFG}"
    SOURCE_DOMAIN="dales"
    TARGET_DOMAIN="eclair"
    SOURCE_TRAIN_TO_COMMON="${MAP_DALES_TRAIN_TO_COMMON}"
    SOURCE_NATIVE_TO_COMMON="${MAP_DALES_NATIVE_TO_COMMON}"
    TARGET_NATIVE_TO_COMMON="${MAP_ECLAIR_NATIVE_TO_COMMON}"
    MANIFEST_ARGS=()
    ;;
  eclair_to_dales)
    CKPT="${ECLAIR_CKPT}"
    SOURCE_CFG="${ECLAIR_CFG}"
    TARGET_CFG="${DALES_TARGET_CFG}"
    SOURCE_DOMAIN="eclair"
    TARGET_DOMAIN="dales"
    SOURCE_TRAIN_TO_COMMON="${MAP_ECLAIR_TRAIN_TO_COMMON}"
    SOURCE_NATIVE_TO_COMMON="${MAP_ECLAIR_NATIVE_TO_COMMON}"
    TARGET_NATIVE_TO_COMMON="${MAP_DALES_NATIVE_TO_COMMON}"
    MANIFEST_ARGS=("${EXTRA_DALES_MANIFEST_ARGS[@]}")
    ;;
  eclair_to_eclair)
    CKPT="${ECLAIR_CKPT}"
    SOURCE_CFG="${ECLAIR_CFG}"
    TARGET_CFG="${ECLAIR_TARGET_CFG}"
    SOURCE_DOMAIN="eclair"
    TARGET_DOMAIN="eclair"
    SOURCE_TRAIN_TO_COMMON="${MAP_ECLAIR_TRAIN_TO_COMMON}"
    SOURCE_NATIVE_TO_COMMON="${MAP_ECLAIR_NATIVE_TO_COMMON}"
    TARGET_NATIVE_TO_COMMON="${MAP_ECLAIR_NATIVE_TO_COMMON}"
    MANIFEST_ARGS=()
    ;;
  *)
    echo "Unsupported CASE=${CASE}"
    exit 2
    ;;
esac

RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RESULT_NAME="${EVAL_TAG}_${TARGET_SPLIT}_w${WINDOW_M}_s${STRIDE_M}_job${RUN_ID}"
OUT_DIR="${CROSS_EVAL_ROOT}/${METHOD_TAG}/${CASE}/${RESULT_NAME}"

if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output directory: ${OUT_DIR}"
  exit 4
fi
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Resolved evaluation"
echo "METHOD_TAG=${METHOD_TAG}"
echo "CASE=${CASE}"
echo "CKPT=${CKPT}"
echo "SOURCE_CFG=${SOURCE_CFG}"
echo "TARGET_CFG=${TARGET_CFG}"
echo "SOURCE_DOMAIN=${SOURCE_DOMAIN}"
echo "TARGET_DOMAIN=${TARGET_DOMAIN}"
echo "SOURCE_TRAIN_TO_COMMON=${SOURCE_TRAIN_TO_COMMON}"
echo "SOURCE_NATIVE_TO_COMMON=${SOURCE_NATIVE_TO_COMMON}"
echo "TARGET_NATIVE_TO_COMMON=${TARGET_NATIVE_TO_COMMON}"
echo "OUT_DIR=${OUT_DIR}"
echo "============================================================"

for required_path in \
  "${CKPT}" \
  "${SOURCE_CFG}" \
  "${TARGET_CFG}" \
  "${SOURCE_TRAIN_TO_COMMON}" \
  "${SOURCE_NATIVE_TO_COMMON}" \
  "${TARGET_NATIVE_TO_COMMON}"
do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}"
    exit 3
  fi
done

{
  printf 'METHOD_TAG=%q\n' "${METHOD_TAG}"
  printf 'CASE=%q\n' "${CASE}"
  printf 'TARGET_SPLIT=%q\n' "${TARGET_SPLIT}"
  printf 'WINDOW_M=%q\n' "${WINDOW_M}"
  printf 'STRIDE_M=%q\n' "${STRIDE_M}"
  printf 'CKPT=%q\n' "${CKPT}"
  printf 'SOURCE_CFG=%q\n' "${SOURCE_CFG}"
  printf 'TARGET_CFG=%q\n' "${TARGET_CFG}"
  printf 'SLURM_JOB_ID=%q\n' "${SLURM_JOB_ID:-NA}"
  printf 'GIT_COMMIT=%q\n' "$(git rev-parse HEAD 2>/dev/null || true)"
} > "${OUT_DIR}/run_identity.env"

########################
# 9) RUN EVAL
########################
python -m src.eval_cross_common \
  --ckpt "${CKPT}" \
  --source_config "${SOURCE_CFG}" \
  --target_config "${TARGET_CFG}" \
  --source_domain "${SOURCE_DOMAIN}" \
  --target_domain "${TARGET_DOMAIN}" \
  --split "${TARGET_SPLIT}" \
  --source_train_to_common "${SOURCE_TRAIN_TO_COMMON}" \
  --source_native_to_common "${SOURCE_NATIVE_TO_COMMON}" \
  --target_native_to_common "${TARGET_NATIVE_TO_COMMON}" \
  "${MANIFEST_ARGS[@]}" \
  --out_dir "${OUT_DIR}" \
  --device cuda \
  --seed 1337 \
  --window_size_m "${WINDOW_M}" \
  --window_stride_m "${STRIDE_M}" \
  --aggregation mean_logits \
  --max_voxels_per_forward "${MAX_VOX}" \
  --max_split_depth "${MAX_SPLIT_DEPTH}" \
  --force_raw_cache \
  --limit_tiles "${LIMIT_TILES}" \
  "${AMP_ARGS[@]}"

test -s "${OUT_DIR}/summary_common.csv"

echo "============================================================"
echo "Finished METHOD_TAG=${METHOD_TAG} CASE=${CASE}"
echo "Output: ${OUT_DIR}"
echo "Summary:"
cat "${OUT_DIR}/summary_common.csv"
echo "============================================================"
