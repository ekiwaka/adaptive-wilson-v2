#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CHUNK_ROOT DATASET [OUT_ROOT=outputs/active/grid/roazhon13/manual_plots] [STEPS_PER_EPOCH=50]" >&2
  echo "Example:" >&2
  echo "  $0 outputs/active/grid/roazhon13/chunked roman-empire" >&2
  exit 2
fi

CHUNK_ROOT="$1"
DATASET="$2"
OUT_ROOT="${3:-outputs/active/grid/roazhon13/manual_plots}"
STEPS_PER_EPOCH="${4:-50}"

export PYTHONPATH="${PYTHONPATH:-$PWD/src}"

python scripts/plot_chunked_seed_results.py \
  --root "$CHUNK_ROOT" \
  --dataset "$DATASET" \
  --out "$OUT_ROOT" \
  --steps-per-epoch "$STEPS_PER_EPOCH"
