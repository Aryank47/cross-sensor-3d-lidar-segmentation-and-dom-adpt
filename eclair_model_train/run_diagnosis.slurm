#!/bin/bash
#SBATCH --job-name=project_diag
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/m23csa510/e0_logs/project_diag_%j.out
#SBATCH --error=/scratch/m23csa510/e0_logs/project_diag_%j.err
#SBATCH --exclude=C07

# Uses the same partition/resources/modules as the existing cross-eval launcher.
# Submit with sbatch; the Python diagnostic itself is a single-process program.
set -eo pipefail

# Keep nounset disabled until Lmod/Conda initialization is complete.
unset LUA_PATH LUA_CPATH
module purge
module load cuda/11.8 gcc/11 gnu12/12.3.0 openblas/0.3.21

PROJECT_ROOT="/csehome/m23csa510/lidar_experiments"
CONDA_ROOT="${PROJECT_ROOT}/env/miniconda3"
CODE_ROOT="${PROJECT_ROOT}/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

export ECLAIR_ROOT="${PROJECT_ROOT}/datasets/eclair"
export ECLAIR_CACHE_ROOT=/scratch/m23csa510/eclair_cache
export DALES_TRAIN_ROOT=/scratch/m23csa510/dales/dales/train
export DALES_TEST_ROOT=/scratch/m23csa510/dales/dales/test
export DALES_CACHE_ROOT_DROP_I=/scratch/m23csa510/dales/dales_cache

cd "${CODE_ROOT}"
test -f "${CODE_ROOT}/src/eval_cross_common.py"

# Your pasted transcript shows a copy in CODE_ROOT. Also support the ZIP folder.
DIAG_SCRIPT="${CODE_ROOT}/diagnose_project.py"
if [[ ! -f "${DIAG_SCRIPT}" ]]; then
    DIAG_SCRIPT="${CODE_ROOT}/one_shot_diagnosis_20261017/diagnose_project.py"
fi
if [[ ! -f "${DIAG_SCRIPT}" ]]; then
    echo "Cannot find diagnose_project.py in CODE_ROOT or its one_shot_diagnosis_20261017 folder." >&2
    exit 2
fi

echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Code root: ${CODE_ROOT}"
echo "Diagnostic script: ${DIAG_SCRIPT}"
nvidia-smi

exec python -u "${DIAG_SCRIPT}" \
    --repo "${CODE_ROOT}" \
    --results-root /scratch/m23csa510/e0_results \
    --out "${DIAG_OUT_DIR:-/scratch/m23csa510/project_diagnosis_20261017}"
