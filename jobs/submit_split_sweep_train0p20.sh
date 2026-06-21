#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/krawat/compactdisk/users/krawat/adaptive-wilson/adaptive-wilson}"
SPLIT_LABEL="train0p20"

cd "${REPO_DIR}"
source jobs/env.sh
export PYTHONPATH="$(pwd)/src"

TOTAL_RUNS="${TOTAL_RUNS:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
FIRST_SEED="${FIRST_SEED:-0}"
CLUSTER="${CLUSTER:-roazhon13}"
CORES="${CORES:-12}"
WALLTIME="${WALLTIME:-19:00:00}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-5}"
TIMESTAMP_OUTPUTS="${TIMESTAMP_OUTPUTS:-true}"
RUN_STAMP="${RUN_STAMP:-${SPLIT_LABEL}_$(date +%Y%m%d_%H%M%S)}"

CONFIG_ROOT="configs/active/grid/online_split_sweep/${SPLIT_LABEL}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-outputs/active/grid/chunked/online_split_sweep/${SPLIT_LABEL}}"

if [[ ! -d "${CONFIG_ROOT}" ]]; then
  echo "Missing config root: ${CONFIG_ROOT}" >&2
  exit 1
fi

mapfile -t CONFIGS < <(find "${CONFIG_ROOT}" -name "*.yaml" | sort)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "No configs found under ${CONFIG_ROOT}" >&2
  exit 1
fi

cat <<EOF
Submitting split sweep jobs
REPO_DIR=${REPO_DIR}
SPLIT_LABEL=${SPLIT_LABEL}
CONFIG_ROOT=${CONFIG_ROOT}
OUTPUT_ROOT_BASE=${OUTPUT_ROOT_BASE}
TOTAL_RUNS=${TOTAL_RUNS}
CHUNK_SIZE=${CHUNK_SIZE}
FIRST_SEED=${FIRST_SEED}
CLUSTER=${CLUSTER}
CORES=${CORES}
WALLTIME=${WALLTIME}
PARALLEL_WORKERS=${PARALLEL_WORKERS}
TIMESTAMP_OUTPUTS=${TIMESTAMP_OUTPUTS}
RUN_STAMP=${RUN_STAMP}
EOF

for cfg in "${CONFIGS[@]}"; do
  regime="$(basename "$(dirname "${cfg}")")"
  base="$(basename "${cfg}" .yaml)"
  dataset="${base%%__*}"

  outroot="${OUTPUT_ROOT_BASE}/${dataset}/${regime}"
  job_prefix="${dataset}_${regime}_${SPLIT_LABEL}"

  echo
  echo "============================================================"
  echo "Submitting ${dataset} ${regime} ${SPLIT_LABEL}"
  echo "Config: ${cfg}"
  echo "Outroot: ${outroot}"
  echo "============================================================"

  REPO_DIR="${REPO_DIR}"   RUN_STAMP="${RUN_STAMP}"   TIMESTAMP_OUTPUTS="${TIMESTAMP_OUTPUTS}"   CLUSTER="${CLUSTER}"   CORES="${CORES}"   WALLTIME="${WALLTIME}"   PARALLEL_WORKERS="${PARALLEL_WORKERS}"   bash jobs/submit_10chunks_cluster.sh     "${cfg}"     "${outroot}"     "${job_prefix}"     "${TOTAL_RUNS}"     "${CHUNK_SIZE}"     "${FIRST_SEED}"
done

echo
echo "Submitted all jobs for ${SPLIT_LABEL}."
