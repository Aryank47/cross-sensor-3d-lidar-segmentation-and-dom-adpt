#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="/scratch/m23csa510"
CLEAN_ROOT="${SCRATCH_ROOT}/cross_eval_runs/clean_eval"
DG_ROOT="${SCRATCH_ROOT}/cross_eval_runs/dg_methods_120m"
LOG_ROOT="${SCRATCH_ROOT}/e0_logs"

REPO_ROOT="/csehome/m23csa510/lidar_experiments/cross-sensor-3d-lidar-segmentation-and-dom-adpt/eclair_model_train"
STAMP="$(date +%Y%m%d_%H%M%S)"

PACKAGE_NAME="cross_eval_complete_history_baseline_mix3d_ocons_bev_${STAMP}"
ARCHIVE="${REPO_ROOT}/${PACKAGE_NAME}.tar.gz"
REVIEW_DIR="${SCRATCH_ROOT}/cross_eval_runs/review_packages/${PACKAGE_NAME}"

# Files larger than this are represented in the inventory but are not copied
# into the review archive. Override only if a specifically needed result is
# reported in excluded_files.tsv.
MAX_FILE_MIB="${MAX_FILE_MIB:-64}"
MAX_FILE_BYTES=$((MAX_FILE_MIB * 1024 * 1024))

for required_dir in "${CLEAN_ROOT}" "${DG_ROOT}" "${LOG_ROOT}" "${REPO_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Missing required directory: ${required_dir}" >&2
    exit 2
  fi
done

mkdir -p "${REVIEW_DIR}"

ABS_CANDIDATES="${REVIEW_DIR}/candidate_files_absolute.txt"
ABS_UNIQUE="${REVIEW_DIR}/candidate_files_absolute_unique.txt"
INCLUDED_LIST="${REVIEW_DIR}/included_files.txt"
INVENTORY="${REVIEW_DIR}/file_inventory.tsv"
EXCLUDED="${REVIEW_DIR}/excluded_files.tsv"
JOB_IDS_FILE="${REVIEW_DIR}/job_ids.txt"
ACCOUNTING="${REVIEW_DIR}/job_accounting.txt"
PACKAGE_INFO="${REVIEW_DIR}/package_info.txt"

: > "${ABS_CANDIDATES}"
: > "${INCLUDED_LIST}"

find "${CLEAN_ROOT}" -type f -print >> "${ABS_CANDIDATES}"
find "${DG_ROOT}" \
  -path "${SCRATCH_ROOT}/cross_eval_runs/review_packages" -prune -o \
  -type f -print >> "${ABS_CANDIDATES}"

while IFS= read -r basename; do
  if [[ "${basename}" =~ job([0-9]{4,})$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  elif [[ "${basename}" =~ _([0-9]{4,})$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
done < <(find "${CLEAN_ROOT}" "${DG_ROOT}" -type d -printf '%f\n') \
  | sort -u > "${JOB_IDS_FILE}"

while IFS= read -r job_id; do
  [[ -n "${job_id}" ]] || continue
  find "${LOG_ROOT}" -maxdepth 1 -type f \
    \( -name "*_${job_id}.out" -o -name "*_${job_id}.err" \) \
    -print >> "${ABS_CANDIDATES}"
done < "${JOB_IDS_FILE}"

sort -u "${ABS_CANDIDATES}" > "${ABS_UNIQUE}"

printf 'relative_path\tsize_bytes\tsha256\tarchive_status\treason\n' > "${INVENTORY}"
printf 'relative_path\tsize_bytes\treason\n' > "${EXCLUDED}"

while IFS= read -r absolute_path; do
  [[ -f "${absolute_path}" ]] || continue

  relative_path="${absolute_path#${SCRATCH_ROOT}/}"
  size_bytes="$(stat -c '%s' "${absolute_path}")"
  reason=""

  case "${relative_path}" in
    */checkpoints/*|*.pt|*.pth|*.ckpt)
      reason="checkpoint_or_model_weight"
      ;;
    */source_snapshot.tar.gz|*.tar|*.tar.gz|*.tgz)
      reason="nested_archive"
      ;;
  esac

  if [[ -z "${reason}" && "${size_bytes}" -gt "${MAX_FILE_BYTES}" ]]; then
    reason="larger_than_${MAX_FILE_MIB}_MiB"
  fi

  if [[ -n "${reason}" ]]; then
    printf '%s\t%s\t%s\n' \
      "${relative_path}" "${size_bytes}" "${reason}" >> "${EXCLUDED}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${relative_path}" "${size_bytes}" "NOT_HASHED" "excluded" "${reason}" >> "${INVENTORY}"
    continue
  fi

  file_sha256="$(sha256sum "${absolute_path}" | awk '{print $1}')"
  printf '%s\n' "${relative_path}" >> "${INCLUDED_LIST}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${relative_path}" "${size_bytes}" "${file_sha256}" "included" "selected_result_or_log" >> "${INVENTORY}"
done < "${ABS_UNIQUE}"

if command -v sacct >/dev/null 2>&1 && [[ -s "${JOB_IDS_FILE}" ]]; then
  JOB_ID_CSV="$(paste -sd, "${JOB_IDS_FILE}")"
  sacct \
    -j "${JOB_ID_CSV}" \
    --format=JobID,JobName%45,State,ExitCode,Elapsed,MaxRSS,AllocTRES%50,NodeList \
    > "${ACCOUNTING}" || true
else
  printf 'Slurm accounting unavailable or no job IDs discovered.\n' > "${ACCOUNTING}"
fi

{
  echo "package_name=${PACKAGE_NAME}"
  echo "created_at=$(date --iso-8601=seconds)"
  echo "clean_eval_root=${CLEAN_ROOT}"
  echo "dg_eval_root=${DG_ROOT}"
  echo "log_root=${LOG_ROOT}"
  echo "max_included_file_mib=${MAX_FILE_MIB}"
  echo "protocol_window_m=120"
  echo "protocol_stride_m=120"
  echo "protocol_aggregation=mean_logits"
  echo "protocol_space=common"
  echo "comparison_methods=baseline,mix3d,ocons,bev_als"
  echo "included_file_count=$(wc -l < "${INCLUDED_LIST}")"
  echo "excluded_file_count=$(( $(wc -l < "${EXCLUDED}") - 1 ))"
  echo "discovered_job_count=$(wc -l < "${JOB_IDS_FILE}")"
} > "${PACKAGE_INFO}"

for metadata_file in \
  "${INVENTORY}" \
  "${EXCLUDED}" \
  "${JOB_IDS_FILE}" \
  "${ACCOUNTING}" \
  "${PACKAGE_INFO}"
do
  printf '%s\n' "${metadata_file#${SCRATCH_ROOT}/}" >> "${INCLUDED_LIST}"
done

sort -u -o "${INCLUDED_LIST}" "${INCLUDED_LIST}"

echo "Source directory sizes:"
du -sh "${CLEAN_ROOT}" "${DG_ROOT}"

echo "Largest candidate files:"
while IFS= read -r candidate; do
  [[ -f "${candidate}" ]] || continue
  stat -c '%s\t%n' "${candidate}"
done < "${ABS_UNIQUE}" | sort -nr | sed -n '1,20p'

cd "${SCRATCH_ROOT}"
tar -czf "${ARCHIVE}" --files-from="${INCLUDED_LIST}"

sha256sum "${ARCHIVE}" | tee "${ARCHIVE}.sha256"

echo "Package information:"
cat "${PACKAGE_INFO}"

echo "Excluded files:"
cat "${EXCLUDED}"

echo "Final archive:"
ls -lh "${ARCHIVE}" "${ARCHIVE}.sha256"

echo "Archive verification:"
tar -tzf "${ARCHIVE}" >/dev/null
echo "OK"
