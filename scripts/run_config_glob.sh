#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <entrypoint> <config-glob>" >&2
  echo "Example: $0 scripts/run_benchmark.py 'configs/active/local/static/unweighted/*.yaml'" >&2
  exit 2
fi

ENTRYPOINT="$1"
CONFIG_GLOB="$2"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
shopt -s nullglob
CONFIGS=( $CONFIG_GLOB )
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs matched: ${CONFIG_GLOB}" >&2
  exit 1
fi
for CONFIG in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running ${ENTRYPOINT} --config ${CONFIG}"
  echo "============================================================"
  python "${ENTRYPOINT}" --config "${CONFIG}"
done
