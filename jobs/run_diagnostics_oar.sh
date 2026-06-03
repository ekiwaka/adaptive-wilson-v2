set -euo pipefail

CONFIG_PATH="${1:?Usage: run_diagnostics_oar.sh <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

source "$PROJECT_ROOT/jobs/env.sh"
mkdir -p "$PROJECT_ROOT/logs"

echo "Running diagnostics with config: $CONFIG_PATH"
echo "Hostname: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found; runtime may fall back to CPU."
fi
python scripts/run_diagnostics.py --config "$CONFIG_PATH"
