#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
python -m pytest -q tests/test_chunked_seed_runner_resume.py
