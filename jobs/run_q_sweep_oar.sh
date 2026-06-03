set -euo pipefail
CONFIG_PATH="${1:?Usage: run_q_sweep_oar.sh <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
source "$PROJECT_ROOT/jobs/env.sh"
mkdir -p "$PROJECT_ROOT/logs"
echo "Running q_sweep with config: $CONFIG_PATH"
echo "Hostname: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found; runtime may fall back to CPU."
fi
python scripts/run_q_sweep.py --config "$CONFIG_PATH"
