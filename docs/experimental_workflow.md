# Experimental workflow

## Setup

Install the project in an environment with PyTorch, PyTorch Geometric, NetworkX, SciPy, NumPy, Matplotlib, scikit-learn, and PyYAML. From the repository root, export the source path:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

The native Wilson extension is optional. Set `sampling.wilson.backend: python` for portable CPU runs. Use `backend: native` only after building the extension according to the repository's existing native-backend instructions.

## Pre-experiment checks

```bash
bash scripts/run_wilson_checks_all.sh
bash scripts/run_dataset_checks.sh
bash scripts/run_graph_visualizations.sh
```

## Smoke tests

```bash
bash scripts/run_static_smoke.sh
bash scripts/run_online_smoke.sh
```

## Main benchmarks

```bash
bash scripts/run_static_benchmarks.sh
bash scripts/run_static_normalized_benchmarks.sh
bash scripts/run_online_benchmarks.sh
bash scripts/run_online_normalized_benchmarks.sh
bash scripts/run_adaptive_wilson_online.sh
bash scripts/run_online_alpha_baseline.sh
```

Grid-sized configs live under `configs/active/grid/` and can be submitted with the same entrypoints or wrapped in the cluster's scheduler. The scripts fail fast, print the active config, set `PYTHONPATH`, and keep outputs separated by regime and weighting mode.

## Progress tracking

`experiments/progress/status.csv` is a lightweight progress ledger. Use one row per config. The output path can include a date-specific folder when repeating a run. For continuation runs, set the config to reuse generated split files and cache directories so the split and preprocessed graph can be recovered.
