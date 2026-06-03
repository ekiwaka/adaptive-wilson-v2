set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs
for cfg in configs/search/grid_*.yaml; do
  echo "Submitting $cfg"
  oarsub -q "${G5K_GPU_QUEUE:-abaca}" -S "./jobs/run_search_oar.sh $cfg"
done
