# Graph sampling for GNN training

This repository studies how graph sampling affects node-classification pipelines for graph neural networks. The implementation supports static sampled-subgraph training, online mini-batch training, uniform and degree sampling, Wilson/q-forest sampling, and the first Adaptive Wilson baseline based on `q_i = alpha`.

The active configuration tree is `configs/active/`. The previous configuration tree is preserved under `configs/legacy/`.

## Main execution modes

Static sampling samples once before optimization and reuses the sampled training graph. Online sampling samples during training, usually once per mini-batch. Switch modes with YAML only:

```yaml
experiment:
  sampling_regime: static   # or online
```

This flag maps to the lower-level `sampling.mode` or `sampling.modes` fields.

## Datasets

Active configs cover Cora, CiteSeer, Roman-empire, Amazon Photo, Amazon Computers, Amazon Ratings, PubMed, Flickr, and Minesweeper. The registry remains extensible through `src/gnn_graph_sampling/data/registry.py`.

## Split selection

For online experiments, the active configs use explicit generated splits:

```yaml
data:
  split:
    mode: generated
    train_ratio: 0.6
    val_ratio: 0.2
    test_ratio: 0.2
    seed: 0
    split_file: outputs/splits/cora/generated_seed0.pt
    save_generated_split: true
    reuse_existing_split: true
```

Use `mode: public` only when a public split is intended. Use `mode: custom` with `split_file` for externally supplied masks or node lists. Generated splits are saved and reused when configured, making online experiments reproducible and resumable.

## Normalized weighted loss

Use the new general flag:

```yaml
optimization:
  use_normalized_weights: true
  weight_normalization: mean_one
```

The legacy static-only flag `use_static_weight_normalization` is still accepted. Uniform weights reduce to the original unweighted loss. Degree weights are finite mean-one inverse-probability-style weights. Static Wilson weights can be exact or Monte Carlo. Online Wilson and Adaptive Wilson currently use a documented mean-one proxy because exact online inclusion probabilities for changing batches are not available in the current loop.

## Wilson options

Wilson q is fixed with `sampling.wilson.q`; if q is null, the historical `mean_degree` approximation remains the default. `q_estimation` is an alias for the q-null strategy. Static Wilson weights use `weight_estimation: exact` or `monte_carlo_forests`.

Budget targeting is disabled by default. Enable it explicitly:

```yaml
sampling:
  wilson:
    budget_targeting_enabled: true
    target_budget_tolerance: 0.03
    max_budget_search_steps: 20
    avoid_padding: true
    avoid_capping: true
```

When enabled, Wilson searches q to target the requested budget and avoids padding/capping.

## Adaptive Wilson

The first supported Adaptive Wilson mode is the supervisor-recommended baseline:

```yaml
sampling:
  name: adaptive_wilson
  adaptive_wilson:
    q_i_mode: alpha
    alpha: 1.0
    alpha_grid: [0.25, 0.5, 1.0, 2.0]
```

This gives a clean baseline for sweeping q/batch size before adding loss-, uncertainty-, or coverage-adaptive score updates.

## Checks and workflows

Run pre-experiment checks first:

```bash
bash scripts/run_pre_experiment_checks.sh
```

Then run smoke tests:

```bash
bash scripts/run_static_smoke.sh
bash scripts/run_online_smoke.sh
```

Run main benchmarks:

```bash
bash scripts/run_static_benchmarks.sh
bash scripts/run_static_normalized_benchmarks.sh
bash scripts/run_online_benchmarks.sh
bash scripts/run_online_normalized_benchmarks.sh
bash scripts/run_adaptive_wilson_online.sh
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

## Outputs and reproducibility

Outputs are separated by static/online regime, normalized/unweighted loss, smoke/benchmark/check workflow, dataset, sampler, seed, and budget. Generated split files and preprocessing caches can be reused for continuation runs. Set `runtime.seed`, `data.split.seed`, and `output.append_timestamp` according to whether a run should be exactly resumed or repeated in a date-specific folder.

## Documentation

See `docs/static_paradigm_experiments.md`, `docs/yaml_flags.md`, `docs/experimental_workflow.md`, `docs/original_file_inventory.txt`, and `docs/unresolved_dependencies.md`.
