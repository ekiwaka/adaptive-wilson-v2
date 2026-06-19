#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_VALUE="${PYTHONPATH:-src}"

echo "Running normalised-weight Monte Carlo smoke benchmarks for all datasets"
CONFIGS=(
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/cora_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/citeseer_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/pubmed_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/flickr_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/amazon_computers_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/amazon_photo_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/coauthor_cs_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/coauthor_physics_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/actor_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/webkb_cornell_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/webkb_texas_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/webkb_wisconsin_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/wikipedia_chameleon_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/wikipedia_squirrel_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/roman_empire_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/amazon_ratings_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/minesweeper_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/tolokers_static_normalised_weights_mc_smoke.yaml"
  "configs/benchmark/sampler_split_normalised_weights_mc_smoke/questions_static_normalised_weights_mc_smoke.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running ${cfg}"
  echo "============================================================"
  PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" scripts/run_benchmark.py --config "${cfg}"
done
