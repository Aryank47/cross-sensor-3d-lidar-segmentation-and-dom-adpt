#!/bin/bash
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=cn07

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
export PAIR_HEAD_DEADLINE_UNIX
PAIR_HEAD_DEADLINE_UNIX="$(scontrol show job "${SLURM_JOB_ID}" -o | python -B -c 'import sys,re,datetime; s=sys.stdin.read(); x=re.search(r"EndTime=(\S+)",s).group(1); print(int(datetime.datetime.fromisoformat(x).timestamp()))')"
exec python -B scripts/run_pair_head_study.py --baseline "${PAIR_HEAD_BASELINE:?}" --study-dir "${PAIR_HEAD_STUDY:?}"
