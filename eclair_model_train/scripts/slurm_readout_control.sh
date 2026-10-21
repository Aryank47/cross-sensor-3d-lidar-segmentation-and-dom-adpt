#!/bin/bash
#SBATCH --job-name=readout_control
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
cd "${READOUT_CODE_ROOT:?Use scripts/submit_readout_control.py}"
export READOUT_DEADLINE_UNIX
READOUT_DEADLINE_UNIX="$(scontrol show job "${SLURM_JOB_ID}" -o | python -B -c 'import sys,re,datetime; s=sys.stdin.read(); x=re.search(r"EndTime=(\S+)",s).group(1); print(int(datetime.datetime.fromisoformat(x).timestamp()))')"
IFS=',' read -r -a arms <<< "${READOUT_ARMS}"
IFS=',' read -r -a endpoints <<< "${READOUT_ENDPOINTS}"
resume=()
if [[ "${READOUT_RESUME:-0}" == "1" ]]; then resume=(--resume); fi
# exec delivers the batch time-limit signal directly to the resumable evaluator.
exec python -B scripts/run_readout_control.py --study-dir "${READOUT_STUDY_DIR}" \
    --output-dir "${READOUT_OUTPUT_DIR}" --arms "${arms[@]}" --endpoints "${endpoints[@]}" "${resume[@]}"
