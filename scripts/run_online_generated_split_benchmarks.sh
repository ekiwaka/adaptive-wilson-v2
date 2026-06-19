#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_config_glob.sh scripts/run_benchmark.py 'configs/active/local/online/unweighted/*.yaml'
