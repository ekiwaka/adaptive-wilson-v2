#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_config_glob.sh scripts/run_dataset_metrics.py 'configs/active/local/checks/dataset_metrics/*.yaml'
