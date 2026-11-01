#!/bin/bash
#SBATCH --job-name=return_input_study
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --exclude=cn07
#SBATCH --signal=B:USR1@900

set -euo pipefail
set +u
unset LUA_PATH LUA_CPATH
module purge
module load cuda/11.8 gcc/11 gnu12/12.3.0 openblas/0.3.21
source /csehome/m23csa510/lidar_experiments/env/miniconda3/etc/profile.d/conda.sh
conda activate e0_env
set -u
export OPENBLAS_ROOT=/opt/ohpc/pub/libs/gnu12/openblas/0.3.21
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${OPENBLAS_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${OPENBLAS_ROOT}/lib:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
cd "${FEATURE_CODE_ROOT:?Use scripts/submit_feature_contract_study.py}"
trap 'touch "${FEATURE_STUDY_DIR}/${FEATURE_STUDY_ARM}/STOP_REQUESTED"' USR1 TERM
export STUDY_DEADLINE_UNIX
STUDY_DEADLINE_UNIX="$(scontrol show job "${SLURM_JOB_ID}" -o | python -c 'import sys,re,datetime; s=sys.stdin.read(); x=re.search(r"EndTime=(\S+)",s).group(1); print(int(datetime.datetime.fromisoformat(x).timestamp()))')"
python scripts/run_feature_contract_study.py --study-dir "${FEATURE_STUDY_DIR}" --arm "${FEATURE_STUDY_ARM}" --resume-flag "${FEATURE_RESUME:-0}"
