#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage:"
  echo "  bash jobs/submit_10chunks_cluster.sh CONFIG OUTROOT_BASE JOB_PREFIX [TOTAL_RUNS=100] [CHUNK_SIZE=10] [START_SEED=0]"
  echo ""
  echo "Optional environment variables:"
  echo "  CLUSTER=roazhon13"
  echo "  CORES=12"
  echo "  WALLTIME=12:00:00"
  echo "  PARALLEL_WORKERS=10"
  exit 2
fi

cd ~/adaptive-wilson/adaptive-wilson

CONFIG="$1"
OUTROOT_BASE="$2"
JOB_PREFIX="$3"
TOTAL_RUNS="${4:-100}"
CHUNK_SIZE="${5:-10}"
FIRST_SEED="${6:-0}"

CLUSTER="${CLUSTER:-roazhon13}"
CORES="${CORES:-12}"
WALLTIME="${WALLTIME:-12:00:00}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-10}"

mkdir -p logs/oar

LAST_SEED=$((FIRST_SEED + TOTAL_RUNS - 1))

echo "CONFIG=${CONFIG}"
echo "OUTROOT_BASE=${OUTROOT_BASE}"
echo "JOB_PREFIX=${JOB_PREFIX}"
echo "TOTAL_RUNS=${TOTAL_RUNS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "FIRST_SEED=${FIRST_SEED}"
echo "CLUSTER=${CLUSTER}"
echo "CORES=${CORES}"
echo "WALLTIME=${WALLTIME}"
echo "PARALLEL_WORKERS=${PARALLEL_WORKERS}"

for START_SEED in $(seq "${FIRST_SEED}" "${CHUNK_SIZE}" "${LAST_SEED}"); do
  REMAINING=$((FIRST_SEED + TOTAL_RUNS - START_SEED))

  if (( REMAINING < CHUNK_SIZE )); then
    THIS_TOTAL="${REMAINING}"
  else
    THIS_TOTAL="${CHUNK_SIZE}"
  fi

  END_SEED=$((START_SEED + THIS_TOTAL - 1))
  JOB_NAME="${JOB_PREFIX}_seeds_${START_SEED}_${END_SEED}"

  echo "Submitting ${JOB_NAME}"

  oarsub -q production \
    -p "cluster='${CLUSTER}'" \
    -l host=1/core=${CORES},walltime=${WALLTIME} \
    -n "${JOB_NAME}" \
    -O "logs/oar/${JOB_NAME}.%jobid%.out" \
    -E "logs/oar/${JOB_NAME}.%jobid%.err" \
    "cd ~/adaptive-wilson/adaptive-wilson && \
     export PARALLEL_WORKERS='${PARALLEL_WORKERS}' && \
     bash jobs/run_one_chunk.sh \
       '${START_SEED}' \
       '${CONFIG}' \
       '${OUTROOT_BASE}' \
       '${THIS_TOTAL}' \
       '${THIS_TOTAL}'"
done

echo "Submitted chunks for ${JOB_PREFIX}."
echo "Check status with: oarstat -u"
