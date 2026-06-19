#!/usr/bin/env bash
set -euo pipefail

cd ~/adaptive-wilson/adaptive-wilson

START_SEED="${1:-${START_SEED:-}}"

if [[ -z "${START_SEED}" ]]; then
  echo "ERROR: START_SEED must be provided."
  echo "Usage: bash jobs/run_roman_norm_one_chunk.sh START_SEED"
  exit 1
fi

END_SEED=$((START_SEED + 9))

CONFIG="${CONFIG:-configs/active/grid/roazhon13/online/normalized/roman_empire__all_samplers.yaml}"
OUTROOT_BASE="${OUTROOT_BASE:-outputs/active/grid/roazhon13/chunked/normalized}"
OUTROOT="${OUTROOT_BASE}/seeds_${START_SEED}_${END_SEED}"

TOTAL_SEEDS="${TOTAL_SEEDS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-10}"

mkdir -p "${OUTROOT}"
mkdir -p "${OUTROOT}/logs"

source jobs/env.sh

export PARALLEL_WORKERS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$(pwd)/src"

CHUNK_LOG="${OUTROOT}/logs/roman_norm_seeds_${START_SEED}_${END_SEED}.job_${OAR_JOB_ID:-no_oar_id}.log"
exec > >(tee -a "${CHUNK_LOG}") 2>&1

echo "========================================"
echo "Roman Empire normalized chunk"
echo "Seeds: ${START_SEED}-${END_SEED}"
echo "Host: $(hostname)"
echo "OAR_JOB_ID: ${OAR_JOB_ID:-none}"
echo "Started at: $(date)"
echo "CONFIG=${CONFIG}"
echo "OUTROOT=${OUTROOT}"
echo "TOTAL_SEEDS=${TOTAL_SEEDS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "START_SEED=${START_SEED}"
echo "END_SEED=${END_SEED}"
echo "PARALLEL_WORKERS=${PARALLEL_WORKERS}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "========================================"

bash scripts/run_roazhon13_chunked_online.sh \
  "${CONFIG}" \
  "${TOTAL_SEEDS}" \
  "${CHUNK_SIZE}" \
  "${START_SEED}" \
  "${OUTROOT}"

echo "Finished seeds ${START_SEED}-${END_SEED} at: $(date)"