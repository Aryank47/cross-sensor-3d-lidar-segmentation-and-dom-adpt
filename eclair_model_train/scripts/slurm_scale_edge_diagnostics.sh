#!/usr/bin/env bash
set -euo pipefail

study="$1"
expected_commit="$2"
root="/scratch/$USER/e0_results/transfer_evidence/scale_edge_v1"
stage="/csehome/$USER/lidar_experiments/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"
cd "$stage"
[[ "$(git rev-parse HEAD)" == "$expected_commit" ]]

case "$study" in
    scale) script=scripts/screen_eclair_scale_pairs.py ;;
    edge) script=scripts/audit_eclair_partial_neighbors.py ;;
    *) exit 2 ;;
esac

source "/csehome/$USER/lidar_experiments/env/miniconda3/etc/profile.d/conda.sh"
conda activate e0_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out="$root/${study}_${SLURM_JOB_ID:?}"
mkdir "$out"
printf '{"state":"running","commit":"%s"}\n' "$expected_commit" > "$out/STATUS.json"
trap 'printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$?" > "$out/STATUS.json"' ERR

python -B -u "$script" | awk -v tmp="$out/result.tmp" '/^\{/ {print > tmp; next} {print; fflush()}'
python -B -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["decision"])' "$out/result.tmp"
mv "$out/result.tmp" "$out/result.json"
printf '{"state":"complete","commit":"%s"}\n' "$expected_commit" > "$out/STATUS.json"
