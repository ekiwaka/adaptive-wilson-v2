#!/usr/bin/env bash
set -euo pipefail

cd ~/adaptive-wilson/adaptive-wilson

source jobs/env.sh

export PYTHONPATH="$(pwd)/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIGS=(
  "configs/active/grid/online_static_informed/normalized/roman_empire__all_samplers_fixed_batch.yaml"
  "configs/active/grid/online_static_informed/unweighted/roman_empire__all_samplers_fixed_batch.yaml"
  "configs/active/grid/online_static_informed/normalized/amazon_ratings__all_samplers_fixed_batch.yaml"
  "configs/active/grid/online_static_informed/unweighted/amazon_ratings__all_samplers_fixed_batch.yaml"
  "configs/active/grid/online_static_informed/normalized/pubmed__all_samplers_fixed_batch.yaml"
  "configs/active/grid/online_static_informed/unweighted/pubmed__all_samplers_fixed_batch.yaml"
)

mkdir -p logs/q_precompute

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "============================================================"
  echo "Precomputing q table for: ${cfg}"
  echo "============================================================"

  python scripts/precompute_online_wilson_q_table.py \
    --config "${cfg}" \
    --force
done

echo
echo "All online expected-budget q tables completed."
