set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

CONFIG_DIR="configs/workflows/dataset_metrics"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Could not find config directory: $CONFIG_DIR"
  exit 1
fi

echo "Running dataset metrics for all YAML configs in: $CONFIG_DIR"
echo

for cfg in "$CONFIG_DIR"/*.yaml; do
  if [ ! -f "$cfg" ]; then
    echo "No YAML configs found in $CONFIG_DIR"
    exit 1
  fi

  echo "============================================================"
  echo "Running dataset metrics config: $cfg"
  echo "============================================================"

  PYTHONPATH=src python scripts/run_dataset_metrics.py --config "$cfg"

  echo
  echo "Finished: $cfg"
  echo
done

echo "All dataset metrics workflows completed successfully."