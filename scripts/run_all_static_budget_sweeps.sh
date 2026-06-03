set -uo pipefail

PYTHONPATH_VALUE="${PYTHONPATH:-src}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "scripts/run_local_cpu_study.py" ]]; then
  echo "Error: run this script from the repository root (scripts/run_local_cpu_study.py not found)." >&2
  exit 1
fi

if [[ ! -f "scripts/plot_budget_curves.py" ]]; then
  echo "Error: scripts/plot_budget_curves.py not found." >&2
  exit 1
fi

studies=(
  "cora_gcn_static_budget_sweep"
  "citeseer_gcn_static_budget_sweep"
  "webkb_cornell_gcn_static_budget_sweep"
  "webkb_texas_gcn_static_budget_sweep"
  "webkb_wisconsin_gcn_static_budget_sweep"
  "wikipedia_chameleon_gcn_static_budget_sweep"
  "wikipedia_squirrel_gcn_static_budget_sweep"
  "sbm_balanced_gcn_static_budget_sweep"
  "sbm_imbalanced_gcn_static_budget_sweep"
)

declare -a run_ok=()
declare -a run_failed=()
declare -a plot_ok=()
declare -a plot_failed=()

echo "============================================================"
echo "Running all static GCN budget-sweep studies"
echo "PYTHONPATH=${PYTHONPATH_VALUE}"
echo "Python=${PYTHON_BIN}"
echo "============================================================"

for study in "${studies[@]}"; do
  study_cfg="configs/refactor/study/budget_sweeps/${study}.yaml"
  summary_csv="outputs/budget_benchmark/${study}/tables/benchmark_summary.csv"
  plot_dir="outputs/budget_benchmark/${study}/budget_curves"

  echo
  echo "------------------------------------------------------------"
  echo "Study: ${study}"
  echo "Config: ${study_cfg}"
  echo "------------------------------------------------------------"

  if [[ ! -f "${study_cfg}" ]]; then
    echo "Missing study config: ${study_cfg}"
    run_failed+=("${study} (missing config)")
    continue
  fi

  if PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" scripts/run_local_cpu_study.py --config "${study_cfg}"; then
    echo "Study finished: ${study}"
    run_ok+=("${study}")
  else
    echo "Study failed: ${study}"
    run_failed+=("${study}")
    continue
  fi

  if [[ ! -f "${summary_csv}" ]]; then
    echo "Missing benchmark summary for plotting: ${summary_csv}"
    plot_failed+=("${study} (missing summary)")
    continue
  fi

  mkdir -p "${plot_dir}"

  if PYTHONPATH="${PYTHONPATH_VALUE}" "${PYTHON_BIN}" scripts/plot_budget_curves.py \
      --summary "${summary_csv}" \
      --outdir "${plot_dir}"; then
    echo "Plots generated: ${plot_dir}"
    plot_ok+=("${study}")
  else
    echo "Plotting failed: ${study}"
    plot_failed+=("${study}")
  fi
done

echo
echo "============================================================"
echo "Final summary"
echo "============================================================"

echo "Successful study runs: ${#run_ok[@]}"
for x in "${run_ok[@]}"; do
  echo "  [OK] ${x}"
done

echo
echo "Failed study runs: ${#run_failed[@]}"
for x in "${run_failed[@]}"; do
  echo "  [FAIL] ${x}"
done

echo
echo "Successful plot generations: ${#plot_ok[@]}"
for x in "${plot_ok[@]}"; do
  echo "  [OK] ${x}"
done

echo
echo "Failed plot generations: ${#plot_failed[@]}"
for x in "${plot_failed[@]}"; do
  echo "  [FAIL] ${x}"
done

if [[ ${#run_failed[@]} -gt 0 || ${#plot_failed[@]} -gt 0 ]]; then
  exit 1
fi

exit 0
