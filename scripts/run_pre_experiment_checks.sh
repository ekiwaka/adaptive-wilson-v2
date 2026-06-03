#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_wilson_checks_all.sh
bash scripts/run_dataset_checks.sh
bash scripts/run_graph_visualizations.sh
