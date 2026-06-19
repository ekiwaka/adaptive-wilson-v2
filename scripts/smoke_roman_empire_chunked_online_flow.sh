#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/active/local/smoke/roman_empire_online_chunked_smoke.yaml}"
OUTPUT_ROOT="${2:-outputs/smoke/roman_empire_chunked_online}"
PLOT_ROOT="${3:-outputs/smoke/roman_empire_manual_plots}"
TOTAL_RUNS="${TOTAL_RUNS:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
START_SEED="${START_SEED:-9000}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-2}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"

if [[ "${KEEP_SMOKE_OUTPUTS:-0}" != "1" ]]; then
  rm -rf "$OUTPUT_ROOT" "$PLOT_ROOT"
fi

echo "[smoke] 1/4: checking chunk tracker helper logic"
python -m pytest -q tests/test_chunked_seed_runner_resume.py

echo "[smoke] 2/4: running only the first chunk"
python scripts/run_chunked_seed_benchmark.py \
  --config "$CONFIG" \
  --total-runs "$CHUNK_SIZE" \
  --chunk-size "$CHUNK_SIZE" \
  --start-seed "$START_SEED" \
  --output-root "$OUTPUT_ROOT" \
  --parallel-workers "$PARALLEL_WORKERS"

echo "[smoke] 3/4: rerunning with TOTAL_RUNS=$TOTAL_RUNS to exercise resume/skipping"
python scripts/run_chunked_seed_benchmark.py \
  --config "$CONFIG" \
  --total-runs "$TOTAL_RUNS" \
  --chunk-size "$CHUNK_SIZE" \
  --start-seed "$START_SEED" \
  --output-root "$OUTPUT_ROOT" \
  --parallel-workers "$PARALLEL_WORKERS"

python - "$OUTPUT_ROOT" "$TOTAL_RUNS" "$CHUNK_SIZE" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
total_runs = int(sys.argv[2])
chunk_size = int(sys.argv[3])
expected_chunks = (total_runs + chunk_size - 1) // chunk_size
status_files = sorted((root / "_chunk_status").glob("*.json"))
if not status_files:
    raise SystemExit(f"No tracker JSON found under {root / '_chunk_status'}")
status = json.loads(status_files[0].read_text(encoding="utf-8"))
chunks = status.get("chunks", {})
done = [key for key, record in chunks.items() if record.get("status") == "done"]
if len(done) != expected_chunks:
    raise SystemExit(
        f"Expected {expected_chunks} done chunks, found {len(done)} in {status_files[0]}: {chunks}"
    )
print(f"[smoke] tracker OK: {status_files[0]} has {len(done)} done chunks")
PY

echo "[smoke] 4/4: combining chunk tables and generating manual plots"
bash scripts/plot_roazhon13_chunked_online.sh \
  "$OUTPUT_ROOT" \
  roman-empire \
  "$PLOT_ROOT" \
  "$STEPS_PER_EPOCH"

python - "$PLOT_ROOT" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

plot_root = Path(sys.argv[1])
manifest = plot_root / "roman_empire" / "combined_manifest.json"
if not manifest.exists():
    raise SystemExit(f"Missing combined manifest: {manifest}")
payload = json.loads(manifest.read_text(encoding="utf-8"))
if not payload.get("plot_extra_metrics_is_called"):
    raise SystemExit("Manifest says plot_extra_metrics.py was not called")
if not payload.get("original_run_benchmark_plots_generated"):
    raise SystemExit("Manifest says original run_benchmark-style plots were not generated")
pngs = sorted(plot_root.rglob("*.png"))
if not pngs:
    raise SystemExit(f"No PNG plots found under {plot_root}")
print(f"[smoke] plotting OK: wrote {len(pngs)} PNG files under {plot_root}")
PY

echo "[smoke] finished successfully"
echo "[smoke] chunk outputs: $OUTPUT_ROOT"
echo "[smoke] manual plots:  $PLOT_ROOT"
