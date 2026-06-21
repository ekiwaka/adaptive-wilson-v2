#!/usr/bin/env bash
set -euo pipefail

# Run split-sweep chunks directly inside an already-started interactive OAR job.
# This script does NOT call oarsub.
# It writes experiment outputs to the SAME standard output structure used by the oarsub scripts:
#   outputs/active/grid/chunked/online_split_sweep/<split>/<dataset>/<regime>/<RUN_STAMP>/seeds_X_Y/
#
# Exact run_one_chunk signature used:
#   bash jobs/run_one_chunk.sh START_SEED CONFIG OUTROOT_BASE TOTAL_SEEDS CHUNK_SIZE
#
# Recommended first use inside an interactive job:
#   MODE=train0p30_all RUN_FOR_HOURS=14 MAX_PARALLEL=1 DRY_RUN=true bash jobs/run_interactive_oarstyle_chunks.sh
#   MODE=train0p30_all RUN_FOR_HOURS=14 MAX_PARALLEL=1 DRY_RUN=false bash jobs/run_interactive_oarstyle_chunks.sh

REPO_DIR="${REPO_DIR:-/home/krawat/compactdisk/users/krawat/adaptive-wilson/adaptive-wilson}"
MODE="${MODE:-train0p30_all}"
DRY_RUN="${DRY_RUN:-true}"
CONFIG_ROOT="${CONFIG_ROOT:-configs/active/grid/online_split_sweep}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/active/grid/chunked/online_split_sweep}"
FIRST_SEED="${FIRST_SEED:-0}"
TOTAL_RUNS="${TOTAL_RUNS:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-5}"
RUN_FOR_HOURS="${RUN_FOR_HOURS:-14}"
MIN_REMAINING_SECONDS="${MIN_REMAINING_SECONDS:-3600}"
SKIP_COMPLETED="${SKIP_COMPLETED:-true}"
SKIP_IF_ANY_COMPLETED="${SKIP_IF_ANY_COMPLETED:-true}"
DATASETS="${DATASETS:-roman_empire amazon_ratings pubmed}"
REGIMES="${REGIMES:-normalized unweighted}"

# If HARD_DEADLINE=true, a chunk is forcibly killed when the run window ends.
# Safer default is false: the script only avoids starting new chunks too close to the deadline.
HARD_DEADLINE="${HARD_DEADLINE:-false}"

case "${MODE}" in
  train0p20_missing)
    DEFAULT_RUN_STAMP="train0p20_$(date +%Y%m%d_%H%M%S)"
    ;;
  train0p30_all)
    DEFAULT_RUN_STAMP="train0p30_$(date +%Y%m%d_%H%M%S)"
    ;;
  both)
    DEFAULT_RUN_STAMP="split_sweep_$(date +%Y%m%d_%H%M%S)"
    ;;
  *)
    DEFAULT_RUN_STAMP="interactive_oarstyle_$(date +%Y%m%d_%H%M%S)"
    ;;
esac
RUN_STAMP="${RUN_STAMP:-${DEFAULT_RUN_STAMP}}"

cd "${REPO_DIR}"

if [[ ! -f jobs/run_one_chunk.sh ]]; then
  echo "ERROR: jobs/run_one_chunk.sh not found under REPO_DIR=${REPO_DIR}" >&2
  exit 2
fi

mkdir -p logs/interactive_oarstyle
MANIFEST="logs/interactive_oarstyle/${RUN_STAMP}.manifest.tsv"
SUMMARY="logs/interactive_oarstyle/${RUN_STAMP}.summary.log"

START_EPOCH="$(date +%s)"
DEADLINE_EPOCH=$((START_EPOCH + RUN_FOR_HOURS * 3600))

export REPO_DIR
export PARALLEL_WORKERS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$(pwd)/src"

cleanup() {
  echo "Interrupted. Terminating child jobs..." >&2
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup INT TERM

is_yes() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "y" ]]
}

find_config() {
  local split="$1"
  local dataset="$2"
  local regime="$3"
  local exact="${CONFIG_ROOT}/${split}/${regime}/${dataset}__${split}_all_samplers_fixed_batch.yaml"

  if [[ -f "${exact}" ]]; then
    printf '%s\n' "${exact}"
    return 0
  fi

  local found
  found=$(find "${CONFIG_ROOT}/${split}/${regime}" -maxdepth 1 -type f -name "*${dataset}*${split}*.yaml" 2>/dev/null | sort | head -1 || true)
  if [[ -n "${found}" && -f "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi

  echo "ERROR: Could not find config for split=${split}, dataset=${dataset}, regime=${regime}" >&2
  echo "Searched exact: ${exact}" >&2
  return 1
}

wait_for_slot() {
  while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_PARALLEL )); do
    sleep 15
  done
}

remaining_seconds() {
  local now
  now=$(date +%s)
  echo $((DEADLINE_EPOCH - now))
}

should_stop_launching() {
  local rem
  rem=$(remaining_seconds)
  (( rem <= MIN_REMAINING_SECONDS ))
}

chunk_done_in_target() {
  local outroot_base="$1"
  local start_seed="$2"
  local total_seeds="$3"
  local end_seed=$((start_seed + total_seeds - 1))
  local chunk_dir="${outroot_base}/seeds_${start_seed}_${end_seed}"

  is_yes "${SKIP_COMPLETED}" || return 1
  [[ -d "${chunk_dir}" ]] || return 1
  find "${chunk_dir}" -name "benchmark_detailed_runs.csv" -print -quit | grep -q .
}

chunk_done_anywhere() {
  local split="$1"
  local dataset="$2"
  local regime="$3"
  local start_seed="$4"
  local total_seeds="$5"
  local end_seed=$((start_seed + total_seeds - 1))
  local base="${OUTPUT_ROOT}/${split}/${dataset}/${regime}"

  is_yes "${SKIP_IF_ANY_COMPLETED}" || return 1
  [[ -d "${base}" ]] || return 1
  find "${base}" -path "*/seeds_${start_seed}_${end_seed}/*" -name "benchmark_detailed_runs.csv" -print -quit | grep -q .
}

run_direct_chunk() {
  local split="$1"
  local dataset="$2"
  local regime="$3"
  local start_seed="$4"
  local total_seeds="$5"

  local config
  config=$(find_config "${split}" "${dataset}" "${regime}")

  local end_seed=$((start_seed + total_seeds - 1))
  local outroot_base="${OUTPUT_ROOT}/${split}/${dataset}/${regime}/${RUN_STAMP}"
  local label="${dataset}_${regime}_${split}_seeds_${start_seed}_${end_seed}"
  local log_file="logs/interactive_oarstyle/${RUN_STAMP}.${label}.log"

  mkdir -p "${outroot_base}"

  if chunk_done_in_target "${outroot_base}" "${start_seed}" "${total_seeds}"; then
    echo -e "SKIP_TARGET_DONE\t${split}\t${dataset}\t${regime}\t${start_seed}\t${end_seed}\t${config}\t${outroot_base}" | tee -a "${MANIFEST}"
    return 0
  fi

  if chunk_done_anywhere "${split}" "${dataset}" "${regime}" "${start_seed}" "${total_seeds}"; then
    echo -e "SKIP_ALREADY_DONE_ANYWHERE\t${split}\t${dataset}\t${regime}\t${start_seed}\t${end_seed}\t${config}\t${outroot_base}" | tee -a "${MANIFEST}"
    return 0
  fi

  if should_stop_launching; then
    echo -e "STOP_TIME_LIMIT\t${split}\t${dataset}\t${regime}\t${start_seed}\t${end_seed}\t${config}\t${outroot_base}" | tee -a "${MANIFEST}"
    return 100
  fi

  echo -e "QUEUED\t${split}\t${dataset}\t${regime}\t${start_seed}\t${end_seed}\t${config}\t${outroot_base}" | tee -a "${MANIFEST}"

  if is_yes "${DRY_RUN}"; then
    echo "DRY_RUN: bash jobs/run_one_chunk.sh '${start_seed}' '${config}' '${outroot_base}' '${total_seeds}' '${total_seeds}'" | tee -a "${SUMMARY}"
    return 0
  fi

  wait_for_slot

  (
    echo "============================================================"
    echo "Starting ${label} at $(date)"
    echo "Host: $(hostname)"
    echo "OAR_JOB_ID: ${OAR_JOB_ID:-none}"
    echo "RUN_STAMP=${RUN_STAMP}"
    echo "CONFIG=${config}"
    echo "OUTROOT_BASE=${outroot_base}"
    echo "Remaining seconds before launch: $(remaining_seconds)"
    echo "MAX_PARALLEL=${MAX_PARALLEL}"
    echo "PARALLEL_WORKERS=${PARALLEL_WORKERS}"
    echo "Command: bash jobs/run_one_chunk.sh '${start_seed}' '${config}' '${outroot_base}' '${total_seeds}' '${total_seeds}'"
    echo "============================================================"

    if is_yes "${HARD_DEADLINE}" && command -v timeout >/dev/null 2>&1; then
      local_timeout=$(remaining_seconds)
      if (( local_timeout < 60 )); then
        echo "Not enough time left for ${label}; skipping."
        exit 124
      fi
      timeout --kill-after=120s "${local_timeout}s" \
        bash jobs/run_one_chunk.sh \
          "${start_seed}" \
          "${config}" \
          "${outroot_base}" \
          "${total_seeds}" \
          "${total_seeds}"
    else
      bash jobs/run_one_chunk.sh \
        "${start_seed}" \
        "${config}" \
        "${outroot_base}" \
        "${total_seeds}" \
        "${total_seeds}"
    fi

    echo "Finished ${label} at $(date)"
  ) > >(tee -a "${log_file}") 2>&1 &
}

add_range() {
  local split="$1"
  local dataset="$2"
  local regime="$3"
  local first="$4"
  local last_start="$5"
  local step="$6"

  local start
  for start in $(seq "${first}" "${step}" "${last_start}"); do
    local last_seed=$((FIRST_SEED + TOTAL_RUNS - 1))
    local remaining=$((last_seed - start + 1))
    local total="${step}"
    if (( remaining < step )); then
      total="${remaining}"
    fi
    if (( total <= 0 )); then
      continue
    fi

    run_direct_chunk "${split}" "${dataset}" "${regime}" "${start}" "${total}" || {
      code=$?
      if [[ "${code}" == "100" ]]; then
        echo "Time limit reached; not launching more chunks." | tee -a "${SUMMARY}"
        return 100
      fi
      return "${code}"
    }
  done
}

run_train0p20_missing() {
  # Previously accepted in OAR: pubmed_unweighted train0p20 seeds 0-4 and 5-9.
  # Missing as observed: pubmed_unweighted seeds 10-99 and roman_empire_unweighted seeds 0-99.
  add_range train0p20 pubmed unweighted 10 95 "${CHUNK_SIZE}" || return $?
  add_range train0p20 roman_empire unweighted 0 95 "${CHUNK_SIZE}" || return $?
}

run_train0p30_all() {
  local dataset regime
  for regime in ${REGIMES}; do
    for dataset in ${DATASETS}; do
      add_range train0p30 "${dataset}" "${regime}" "${FIRST_SEED}" "$((FIRST_SEED + TOTAL_RUNS - CHUNK_SIZE))" "${CHUNK_SIZE}" || return $?
    done
  done
}

cat <<RUNTIME_HEADER | tee "${SUMMARY}"
============================================================
Interactive OAR-style split-sweep chunk runner
Started at: $(date)
Will stop launching new chunks after: $(date -d "@${DEADLINE_EPOCH}" 2>/dev/null || date -r "${DEADLINE_EPOCH}" 2>/dev/null || echo "epoch ${DEADLINE_EPOCH}")
Host: $(hostname)
OAR_JOB_ID: ${OAR_JOB_ID:-none}
REPO_DIR=${REPO_DIR}
MODE=${MODE}
DRY_RUN=${DRY_RUN}
RUN_STAMP=${RUN_STAMP}
CONFIG_ROOT=${CONFIG_ROOT}
OUTPUT_ROOT=${OUTPUT_ROOT}
FIRST_SEED=${FIRST_SEED}
TOTAL_RUNS=${TOTAL_RUNS}
CHUNK_SIZE=${CHUNK_SIZE}
MAX_PARALLEL=${MAX_PARALLEL}
PARALLEL_WORKERS=${PARALLEL_WORKERS}
RUN_FOR_HOURS=${RUN_FOR_HOURS}
MIN_REMAINING_SECONDS=${MIN_REMAINING_SECONDS}
HARD_DEADLINE=${HARD_DEADLINE}
SKIP_COMPLETED=${SKIP_COMPLETED}
SKIP_IF_ANY_COMPLETED=${SKIP_IF_ANY_COMPLETED}
DATASETS=${DATASETS}
REGIMES=${REGIMES}
MANIFEST=${MANIFEST}
============================================================
RUNTIME_HEADER

echo -e "status\tsplit\tdataset\tregime\tstart_seed\tend_seed\tconfig\toutroot_base" > "${MANIFEST}"

case "${MODE}" in
  train0p20_missing)
    run_train0p20_missing || true
    ;;
  train0p30_all)
    run_train0p30_all || true
    ;;
  both)
    run_train0p20_missing || true
    run_train0p30_all || true
    ;;
  *)
    echo "ERROR: Unknown MODE=${MODE}. Use train0p20_missing, train0p30_all, or both." >&2
    exit 2
    ;;
esac

if is_yes "${DRY_RUN}"; then
  echo "Dry run complete. Review: ${MANIFEST}" | tee -a "${SUMMARY}"
  exit 0
fi

wait

echo "All launched interactive chunks finished at $(date)" | tee -a "${SUMMARY}"
echo "Manifest: ${MANIFEST}" | tee -a "${SUMMARY}"
echo "Standard output location used: ${OUTPUT_ROOT}/<split>/<dataset>/<regime>/${RUN_STAMP}/seeds_X_Y" | tee -a "${SUMMARY}"
