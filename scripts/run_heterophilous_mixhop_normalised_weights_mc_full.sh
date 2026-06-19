#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_VALUE="${PYTHONPATH:-src}"

echo "Running heterophilous MixHop normalised-weight Monte Carlo full benchmarks"
CONFIGS=(
  "configs/benchmark/sampler_split_normalised_weights_mc/actor_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/webkb_cornell_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/webkb_texas_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/webkb_wisconsin_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/wikipedia_chameleon_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/wikipedia_squirrel_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/roman_empire_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/amazon_ratings_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/minesweeper_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/tolokers_budget_sweep_static_normalised_weights_mc.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc/questions_budget_sweep_static_normalised_weights_mc.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running ${cfg}"
  echo "============================================================"
  PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" scripts/run_benchmark.py --config "${cfg}"
done
