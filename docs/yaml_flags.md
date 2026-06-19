# YAML flag reference

The repository is controlled by YAML dataclasses in `src/gnn_graph_sampling/config/schema.py`. Existing flags remain valid; the update adds aliases and new fields for explicit static/online switching, split management, normalized weighting, Wilson budget targeting, and Adaptive Wilson.

| YAML flag | Type | Default | Used by | Meaning | Valid values | Notes |
|---|---:|---:|---|---|---|---|
| `experiment.sampling_regime` | string/null | null | train, benchmark | Overrides the active sampling regime without code changes. | `static`, `online`, null | Maps to `sampling.mode` or `sampling.modes`. |
| `dataset.split.mode` / `data.split.mode` | string | `predefined` | dataset loading | Selects split source. | `predefined`, `public`, `custom`, `generated`, `sampler` | Online active configs use `generated`; public splits must be explicit. |
| `dataset.split.name` | string/null | null | custom split loading | Selects a named split inside a split file. | any key | Required only for nested custom split payloads. |
| `dataset.split.train_ratio` | float | 0.6 | generated split | Fraction of nodes assigned to train. | 0-1 | Ratios must sum to 1. |
| `dataset.split.val_ratio` | float | 0.2 | generated split | Fraction of nodes assigned to validation. | 0-1 | Ratios must sum to 1. |
| `dataset.split.test_ratio` | float | 0.2 | generated split | Fraction of nodes assigned to test. | 0-1 | Remaining mask is test. |
| `dataset.split.seed` | int/null | 0 | generated split | Seed for reproducible split creation. | integer | Falls back to runtime seed if null. |
| `dataset.split.split_file` | string/null | null | custom/generated split | Path to load or save split masks/nodes. | path | Generated splits can be reused. |
| `dataset.split.save_generated_split` | bool | true | generated split | Save generated masks to `split_file`. | true/false | Creates parent directories. |
| `dataset.split.reuse_existing_split` | bool | true | generated split | Reuse existing split file instead of regenerating. | true/false | Enables continuation-safe split reuse. |
| `sampling.name` | string | `uniform` | training | Center-node sampler. | `uniform`, `degree`, `wilson`, `adaptive_wilson`, `none` | `adaptive_wilson` is online-capable baseline alpha mode. |
| `sampling.mode` | string | `online` | training | Single-run sampling regime. | `static`, `online` | Can be overridden by `experiment.sampling_regime`. |
| `sampling.modes` | list | `[online]` | benchmark | Benchmark regimes to sweep. | list of `static`/`online` | Can be overridden by `experiment.sampling_regime`. |
| `sampling.online_schedule` | string | `per_batch` | training | Online refresh cadence. | `per_batch`, `per_epoch` | Per-batch is the minibatch path. |
| `sampling.budget_unit` | string | `count` | sampling | Budget interpretation. | `count`, `ratio` | Ratio is relative to candidate train nodes. |
| `sampling.budget_value` / `sampling.budget_values` | number/list | null/[] | train/benchmark | Requested center-node budget. | positive count or ratio | Benchmark falls back to `batch_sizes` when empty. |
| `sampling.wilson.q` | float/null | null | Wilson | Fixed q. | positive/null | Null uses q estimation. |
| `sampling.wilson.auto_q_method` | string | `mean_degree` | Wilson | Existing q=null strategy. | `mean_degree`, `empirical_binary_search`, `precomputed_table` | Preserved for compatibility. |
| `sampling.wilson.q_estimation` | string/null | null | Wilson | Alias for q-null strategy. | same as `auto_q_method` | If set, updates `auto_q_method`. |
| `sampling.wilson.weight_estimation` | string | `exact` | static weights | Static Wilson probability estimation. | `exact`, `monte_carlo_forests` | Exact uses solves, MC samples q-forests. |
| `sampling.wilson.num_weight_forests` | int | 256 | Wilson MC weights | Number of forests for MC static weights. | positive int | Seeded by `weight_estimation_seed`. |
| `sampling.wilson.budget_targeting_enabled` | bool | false | Wilson | Enables ± tolerance q budget search. | true/false | Disabled by default. |
| `sampling.wilson.target_budget_tolerance` | float | 0.03 | Wilson targeting | Relative target tolerance. | 0-1 | 0.03 means ±3%. |
| `sampling.wilson.max_budget_search_steps` | int | 20 | Wilson targeting | Maximum q-search iterations. | positive int | Used only when enabled. |
| `sampling.wilson.avoid_padding` | bool | false | Wilson | Disables uniform fill when true. | true/false | Targeting configs set true. |
| `sampling.wilson.avoid_capping` | bool | false | Wilson | Requires q-target search to avoid overshoot capping. | true/false | Fails clearly if incompatible. |
| `sampling.adaptive_wilson.q_i_mode` | string | `alpha` | Adaptive Wilson | Per-node q mode. | `alpha` | Current supported baseline. |
| `sampling.adaptive_wilson.alpha` | float | 1.0 | Adaptive Wilson | Scalar q value for `q_i = alpha`. | positive float | Supervisor-recommended baseline. |
| `sampling.adaptive_wilson.alpha_grid` | list | [] | experiments | Candidate alpha sweep values. | positive floats | Stored in config for sweeps. |
| `optimization.use_static_weight_normalization` | bool | false | legacy static | Legacy static sampler-split normalized loss flag. | true/false | Preserved. |
| `optimization.use_normalized_weights` | bool | false | static/online | General sampler-normalized weighting flag. | true/false | Preferred new flag. |
| `optimization.normalize_weights` | bool | true | weighting | Keeps weights on mean-one scale. | true/false | Current implementation supports mean-one. |
| `optimization.weight_normalization` | string | `mean_one` | weighting | Normalization rule. | `mean_one` | Guards invalid configs. |
| `wilson_checks.q` | float/null | 1.0 | checks | q used in Wilson diagnostics. | positive/null | Falls back to Wilson q if null. |
| `wilson_checks.num_forests` | int | 100 | checks | Forests for Monte Carlo diagnostics. | positive int | Drives log-log convergence. |
| `wilson_checks.max_exact_nodes` | int | 3000 | checks | Dense exact-kernel guard. | positive int | Increase explicitly for larger graphs. |
| `wilson_checks.max_pairs` | int | 64 | checks | Number of node pairs checked. | positive int | Pair determinant and negative correlation. |
| `metrics.max_exact_path_nodes` | int | 2000 | dataset checks | Graph metric exact path cutoff. | positive int | Existing dataset metrics workflow. |
| `visualization.label_nodes` | bool | false | graph visualization | Draws integer labels on selected nodes. | true/false | Added to avoid points-only plots. |
| `visualization.label_with` | string | `label` | graph visualization | Label text. | `label`, `node_id` | Labels are integer class labels when available. |
| `runtime.seed` | int | 0 | all | Reproducibility seed. | integer | Also controls generated splits unless split seed is set. |
| `output.root` | string | outputs | all | Base output directory. | path | Active configs separate static/online and normalized/unweighted outputs. |
