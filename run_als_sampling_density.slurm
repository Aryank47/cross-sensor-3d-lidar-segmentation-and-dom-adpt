#!/bin/bash
#SBATCH --job-name=als_density
#SBATCH --output=als_density_%j.out
#SBATCH --error=als_density_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

# Activate the same environment used by the ECLAIR/DALES training code.
# Example:
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate e0_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/analyze_als_sampling_density.py \
  --repo-root . \
  --config configs/als_sampling_density.yaml
