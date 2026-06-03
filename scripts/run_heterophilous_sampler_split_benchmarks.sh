#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_VALUE="${PYTHONPATH:-src}"

CONFIGS=(
  # configs/benchmark/sampler_split/actor_budget_sweep_static.yaml
  # configs/benchmark/sampler_split/webkb_cornell_budget_sweep_static.yaml
  # configs/benchmark/sampler_split/webkb_texas_budget_sweep_static.yaml
  # configs/benchmark/sampler_split/webkb_wisconsin_budget_sweep_static.yaml
  # configs/benchmark/sampler_split/wikipedia_chameleon_budget_sweep_static.yaml
  # configs/benchmark/sampler_split/wikipedia_squirrel_budget_sweep_static.yaml
  configs/benchmark/sampler_split/roman_empire_budget_sweep_static.yaml
  configs/benchmark/sampler_split/amazon_ratings_budget_sweep_static.yaml
  configs/benchmark/sampler_split/minesweeper_budget_sweep_static.yaml
  configs/benchmark/sampler_split/tolokers_budget_sweep_static.yaml
  # configs/benchmark/sampler_split/questions_budget_sweep_static.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Heterophilous full benchmark: ${cfg}"
  echo "============================================================"
  PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" scripts/run_benchmark.py --config "${cfg}"
done
