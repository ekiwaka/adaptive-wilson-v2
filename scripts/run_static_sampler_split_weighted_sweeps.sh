#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_VALUE="${PYTHONPATH:-src}"

cat <<'EOF'
Before running this full sweep, it is recommended to run:

  bash scripts/run_sampler_split_smokes_all_datasets.sh
  bash scripts/run_sampler_split_weighted_smokes_all_datasets.sh

EOF

CONFIGS=(
  configs/benchmark/sampler_split/cora_budget_sweep_static.yaml
  configs/benchmark/sampler_split/citeseer_budget_sweep_static.yaml
  configs/benchmark/sampler_split/pubmed_budget_sweep_static.yaml
  configs/benchmark/sampler_split/actor_budget_sweep_static.yaml
  configs/benchmark/sampler_split/flickr_budget_sweep_static.yaml
  configs/benchmark/sampler_split/amazon_computers_budget_sweep_static.yaml
  configs/benchmark/sampler_split/amazon_photo_budget_sweep_static.yaml
  configs/benchmark/sampler_split/coauthor_cs_budget_sweep_static.yaml
  configs/benchmark/sampler_split/coauthor_physics_budget_sweep_static.yaml
  configs/benchmark/sampler_split/webkb_cornell_budget_sweep_static.yaml
  configs/benchmark/sampler_split/webkb_texas_budget_sweep_static.yaml
  configs/benchmark/sampler_split/webkb_wisconsin_budget_sweep_static.yaml
  configs/benchmark/sampler_split/wikipedia_chameleon_budget_sweep_static.yaml
  configs/benchmark/sampler_split/wikipedia_squirrel_budget_sweep_static.yaml
  configs/benchmark/sampler_split/roman_empire_budget_sweep_static.yaml
  configs/benchmark/sampler_split/amazon_ratings_budget_sweep_static.yaml
  configs/benchmark/sampler_split/minesweeper_budget_sweep_static.yaml
  configs/benchmark/sampler_split/tolokers_budget_sweep_static.yaml
  configs/benchmark/sampler_split/questions_budget_sweep_static.yaml
)

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "Missing config: ${cfg}" >&2
    exit 1
  fi
  echo "============================================================"
  echo "Running ${cfg}"
  echo "============================================================"
  PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" scripts/run_benchmark.py --config "${cfg}"
done
