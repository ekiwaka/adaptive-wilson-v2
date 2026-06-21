#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/krawat/compactdisk/users/krawat/adaptive-wilson/adaptive-wilson}"
SPLIT_LABEL="${SPLIT_LABEL:-train0p10}"
DATASET="${DATASET:-roman_empire}"
REGIME="${REGIME:-unweighted}"
BUDGET="${BUDGET:-64}"

cd "${REPO_DIR}"
source jobs/env.sh
export PYTHONPATH="$(pwd)/src"

SOURCE_CONFIG="configs/active/grid/online_split_sweep/${SPLIT_LABEL}/${REGIME}/${DATASET}__${SPLIT_LABEL}_all_samplers_fixed_batch.yaml"
SMOKE_CONFIG="configs/smoke/online_split_sweep/${SPLIT_LABEL}/${REGIME}/${DATASET}_budget${BUDGET}_smoke.yaml"
SMOKE_ROOT="outputs/smoke/grid/online_split_sweep/${SPLIT_LABEL}/${DATASET}/${REGIME}"

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "Missing source config: ${SOURCE_CONFIG}" >&2
  exit 1
fi

python - <<PY
from pathlib import Path
import yaml

source = Path('${SOURCE_CONFIG}')
out = Path('${SMOKE_CONFIG}')
cfg = yaml.safe_load(source.read_text())

cfg['study_name'] = '${DATASET}_${REGIME}_${SPLIT_LABEL}_q_usage_smoke'
cfg.setdefault('output', {})['root'] = '${SMOKE_ROOT}'
cfg['output']['experiment_name'] = '${DATASET}_${REGIME}_${SPLIT_LABEL}_q_usage_smoke'
cfg['output']['overwrite'] = True
cfg['output']['append_timestamp'] = False

cfg.setdefault('optimization', {})['epochs'] = 2
cfg['optimization']['min_epochs'] = 1
cfg['optimization']['early_stopping_patience'] = 100

cfg.setdefault('sampling', {})['steps_per_epoch'] = 2
cfg['sampling']['budget_values'] = [int('${BUDGET}')]
cfg['sampling']['batch_sizes'] = [64]

cfg.setdefault('benchmark', {})['runs'] = 1
cfg['benchmark']['parallel_workers'] = 1
cfg['benchmark']['save_plots'] = False

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
print('q_table_path:', cfg['sampling']['wilson']['q_table_path'])
print('split_file:', cfg['data']['split']['split_file'])
PY

RUN_STAMP="${RUN_STAMP:-smoke_${SPLIT_LABEL}_${DATASET}_${REGIME}_$(date +%Y%m%d_%H%M%S)}" \
TIMESTAMP_OUTPUTS=true \
CLUSTER="${CLUSTER:-roazhon13}" \
CORES="${CORES:-4}" \
WALLTIME="${WALLTIME:-02:00:00}" \
PARALLEL_WORKERS=1 \
REPO_DIR="${REPO_DIR}" \
bash jobs/submit_10chunks_cluster.sh \
  "${SMOKE_CONFIG}" \
  "${SMOKE_ROOT}" \
  "smoke_${DATASET}_${REGIME}_${SPLIT_LABEL}" \
  1 \
  1 \
  0
