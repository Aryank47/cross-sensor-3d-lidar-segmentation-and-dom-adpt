#!/bin/bash
#SBATCH --job-name=transfer_evidence
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export DALES_TRAIN_ROOT=/scratch/m23csa510/dales/dales/train
export DALES_TEST_ROOT=/scratch/m23csa510/dales/dales/test
export DALES_CACHE_ROOT_DROP_I=/scratch/m23csa510/dales/dales_cache
export ECLAIR_ROOT=/csehome/m23csa510/lidar_experiments/datasets/eclair
export ECLAIR_CACHE_ROOT=/scratch/m23csa510/eclair_cache
cd "${TRANSFER_CODE_ROOT:?Use scripts/submit_transfer_evidence_study.py}"
TRANSFER_DEADLINE_UNIX="$(scontrol show job "${SLURM_JOB_ID}" -o | python -B -c 'import sys,re,datetime; s=sys.stdin.read(); x=re.search(r"EndTime=(\S+)",s).group(1); print(int(datetime.datetime.fromisoformat(x).timestamp()))')"
export TRANSFER_DEADLINE_UNIX
resume=()
if [[ "${TRANSFER_RESUME:-0}" == "1" ]]; then resume=(--resume); fi
exec python -B scripts/run_transfer_evidence_study.py --output "${TRANSFER_OUTPUT}" "${resume[@]}"
