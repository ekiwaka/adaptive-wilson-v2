#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_config_glob.sh scripts/run_benchmark.py 'configs/active/local/online/unweighted/*__adaptive_wilson.yaml'
bash scripts/run_config_glob.sh scripts/run_benchmark.py 'configs/active/local/online/normalized/*__adaptive_wilson.yaml'
