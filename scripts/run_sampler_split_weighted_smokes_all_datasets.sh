#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_VALUE="${PYTHONPATH:-src}"
CONFIG_DIR="configs/benchmark/sampler_split_weighted_smoke"

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Missing weighted smoke config directory: ${CONFIG_DIR}" >&2
  exit 1
fi

for cfg in "${CONFIG_DIR}"/*_sampler_split_static_weighted_smoke.yaml; do
  [[ "${cfg}" == *"all_datasets"* ]] && continue
  echo "============================================================"
  echo "Weighted smoke benchmark: ${cfg}"
  echo "============================================================"
  PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" scripts/run_benchmark.py --config "${cfg}"
done
