set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gnn-pub}"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"

module load mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "HOSTNAME=$(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi
python --version
which python
