#!/usr/bin/env bash
set -euo pipefail

cd ~/adaptive-wilson/adaptive-wilson

mkdir -p logs/oar

CONFIG="configs/active/grid/roazhon13/online/normalized/roman_empire__all_samplers.yaml"
OUTROOT_BASE="outputs/active/grid/roazhon13/chunked/normalized"

TOTAL_SEEDS_PER_JOB=10
CHUNK_SIZE=10
PARALLEL_WORKERS=10
CORES=12
WALLTIME="12:00:00"

for START_SEED in 0 10 20 30 40 50 60 70 80 90; do
  END_SEED=$((START_SEED + 9))
  JOB_NAME="roman_norm_seeds_${START_SEED}_${END_SEED}"

  echo "Submitting ${JOB_NAME}"

  oarsub -q production \
    -p "cluster='roazhon13'" \
    -l host=1/core=${CORES},walltime=${WALLTIME} \
    -n "${JOB_NAME}" \
    -O "logs/oar/${JOB_NAME}.%jobid%.out" \
    -E "logs/oar/${JOB_NAME}.%jobid%.err" \
    "cd ~/adaptive-wilson/adaptive-wilson && \
     CONFIG='${CONFIG}' \
     OUTROOT_BASE='${OUTROOT_BASE}' \
     TOTAL_SEEDS='${TOTAL_SEEDS_PER_JOB}' \
     CHUNK_SIZE='${CHUNK_SIZE}' \
     PARALLEL_WORKERS='${PARALLEL_WORKERS}' \
     bash jobs/run_roman_norm_one_chunk.sh ${START_SEED}"
done

echo "Submitted all chunks."
echo "Check status with: oarstat -u"