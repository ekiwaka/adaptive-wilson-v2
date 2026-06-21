#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/krawat/compactdisk/users/krawat/adaptive-wilson/adaptive-wilson}"
SPLIT_LABEL="${SPLIT_LABEL:-}"

cd "${REPO_DIR}"
source jobs/env.sh

export PYTHONPATH="$(pwd)/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -n "${SPLIT_LABEL}" ]]; then
  CONFIG_ROOT="configs/active/grid/online_split_sweep/${SPLIT_LABEL}"
else
  CONFIG_ROOT="configs/active/grid/online_split_sweep"
fi

if [[ ! -d "${CONFIG_ROOT}" ]]; then
  echo "Missing config root: ${CONFIG_ROOT}" >&2
  exit 1
fi

mkdir -p logs/q_precompute_split_sweep
mapfile -t CONFIGS < <(find "${CONFIG_ROOT}" -name "*.yaml" | sort)

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "No configs found under ${CONFIG_ROOT}" >&2
  exit 1
fi

echo "Precomputing q tables under ${CONFIG_ROOT}"
printf '  %s\n' "${CONFIGS[@]}"

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "============================================================"
  echo "Precomputing q table for: ${cfg}"
  echo "============================================================"
  /usr/bin/time -v python scripts/precompute_online_wilson_q_table.py \
    --config "${cfg}" \
    --force
done

echo
echo "All requested split-sweep q tables completed."
