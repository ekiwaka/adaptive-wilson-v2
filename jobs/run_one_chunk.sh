#!/usr/bin/env bash
set -euo pipefail

START_SEED="${1:?Usage: bash jobs/run_one_chunk.sh START_SEED CONFIG OUTROOT_BASE [TOTAL_SEEDS] [CHUNK_SIZE]}"
CONFIG="${2:?Usage: bash jobs/run_one_chunk.sh START_SEED CONFIG OUTROOT_BASE [TOTAL_SEEDS] [CHUNK_SIZE]}"
OUTROOT_BASE="${3:?Usage: bash jobs/run_one_chunk.sh START_SEED CONFIG OUTROOT_BASE [TOTAL_SEEDS] [CHUNK_SIZE]}"
TOTAL_SEEDS="${4:-10}"
CHUNK_SIZE="${5:-10}"

END_SEED=$((START_SEED + TOTAL_SEEDS - 1))

cd ~/adaptive-wilson/adaptive-wilson

OUTROOT="${OUTROOT_BASE}/seeds_${START_SEED}_${END_SEED}"
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

CHUNK_LOG="${OUTROOT}/logs/seeds_${START_SEED}_${END_SEED}.job_${OAR_JOB_ID:-no_oar_id}.log"
exec > >(tee -a "${CHUNK_LOG}") 2>&1

echo "========================================"
echo "Online chunk job"
echo "Seeds: ${START_SEED}-${END_SEED}"
echo "Host: $(hostname)"
echo "OAR_JOB_ID: ${OAR_JOB_ID:-none}"
echo "Started at: $(date)"
echo "CONFIG=${CONFIG}"
echo "OUTROOT=${OUTROOT}"
echo "TOTAL_SEEDS=${TOTAL_SEEDS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
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