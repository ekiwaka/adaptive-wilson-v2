#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG [TOTAL_RUNS=100] [CHUNK_SIZE=25] [START_SEED=0] [OUTPUT_ROOT=outputs/active/grid/roazhon13/chunked]" >&2
  echo "Example:" >&2
  echo "  $0 configs/active/grid/roazhon13/online/unweighted/roman_empire__all_samplers.yaml 100 25 0" >&2
  exit 2
fi

CONFIG="$1"
TOTAL_RUNS="${2:-100}"
CHUNK_SIZE="${3:-25}"
START_SEED="${4:-0}"
OUTPUT_ROOT="${5:-outputs/active/grid/roazhon13/chunked}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"

python scripts/run_chunked_seed_benchmark.py \
  --config "$CONFIG" \
  --total-runs "$TOTAL_RUNS" \
  --chunk-size "$CHUNK_SIZE" \
  --start-seed "$START_SEED" \
  --output-root "$OUTPUT_ROOT" \
  --parallel-workers "$PARALLEL_WORKERS"
