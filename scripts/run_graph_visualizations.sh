#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_config_glob.sh scripts/run_graph_visualization.py 'configs/active/local/checks/graph_visualization/*.yaml'
