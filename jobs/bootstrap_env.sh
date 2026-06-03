set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gnn-pub}"

cd "$PROJECT_ROOT"
module load mamba
mamba create -y -n "$CONDA_ENV_NAME" python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Environment '$CONDA_ENV_NAME' is ready on host: $(hostname)"
