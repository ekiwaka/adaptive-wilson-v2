set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs
configs=(
  configs/benchmark/grid_smoke.yaml
  configs/benchmark/grid_all_datasets.yaml
)
for cfg in "${configs[@]}"; do
  echo "Submitting $cfg"
  oarsub -q "${G5K_GPU_QUEUE:-abaca}" -S "./jobs/run_benchmark_oar.sh $cfg"
done
