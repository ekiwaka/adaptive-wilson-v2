# Static and online sampling experiments

This guide describes the current implementation after the update. It is written for the active configuration tree under `configs/active/`; older configs are preserved under `configs/legacy/`.

## Static paradigm

Static experiments build one sampled training graph before optimization and reuse it. When `data.split.mode: sampler` is used, the sampler itself chooses the training nodes and the remaining nodes are partitioned into validation and test sets using `val_ratio_of_remaining`. Static normalized weighting is enabled with `optimization.use_normalized_weights: true` or the legacy `optimization.use_static_weight_normalization: true`.

For uniform static sampling, normalized weights are all one, so the loss is equivalent to the unweighted loss. For degree sampling, inverse-probability-style weights are computed from center-node degree probabilities. For Wilson sampling, static weights can be exact or Monte Carlo. Exact Wilson weights use the q-root marginal formula and linear solves; Monte Carlo weights estimate inclusion probabilities from repeated q-forest draws.

## Online paradigm

Online experiments sample during training. The per-batch path samples new center nodes for every optimization step, extracts the exact k-hop subgraph, computes logits for the local centers, and applies the configured loss. Online active configs use `data.split.mode: generated` with an explicit `split_file`, so the training split is not silently inherited from a public benchmark split. Set `data.split.mode: public` only when the public split is intentionally desired.

Online normalized weighting is enabled with `optimization.use_normalized_weights: true`. The implementation supports finite mean-one weights for uniform and degree batches. Wilson and Adaptive Wilson use a documented mean-one proxy in the online path because exact per-step Wilson inclusion probabilities for changing candidate sets are not available in the current loop.

## Adaptive Wilson and the alpha baseline

Adaptive Wilson is exposed as `sampling.name: adaptive_wilson`. The currently implemented mode is `sampling.adaptive_wilson.q_i_mode: alpha`, where every candidate node uses the scalar baseline `q_i = alpha`. This matches the supervisor guidance to start with the easiest q sweep and compare it against uniform, degree, and standard Wilson before adding richer dynamic score policies.

## Wilson budget targeting

Wilson budget targeting is off by default. Default Wilson behavior preserves the legacy path unless `sampling.wilson.budget_targeting_enabled: true` is set. Opt-in configs also set `avoid_padding: true` and `avoid_capping: true`, so the final set remains a Wilson-root set rather than a padded or truncated set. The q search targets `target_budget_tolerance`, which defaults to `0.03`.

## Pre-experiment checks

Run Wilson checks and dataset checks before main experiments:

```bash
bash scripts/run_pre_experiment_checks.sh
```

Wilson checks save exact expected root counts, Monte Carlo convergence curves, per-node marginals, pairwise determinant checks, negative-correlation gaps, symmetry-invariance errors, CSV files, JSON metrics, and log-log plots. Dataset checks use the existing dataset metrics workflow for degree distribution, path length, sparsity, clustering, components, spectral communities, data distribution, homophily, and gradient-variance-oriented diagnostics where supported. Graph visualization configs now draw integer labels when `visualization.label_nodes: true`.

## Active datasets

The active local/grid configs cover Cora, CiteSeer, Roman-empire, Amazon Photo, Amazon Computers, Amazon Ratings, PubMed, Flickr, and Minesweeper. Filenames use slash-safe dataset slugs such as `amazon_photo` and `amazon_computers`.

## Comparing results

Use the output roots encoded in the YAML files. Static and online outputs are separated. Normalized and unweighted outputs are separated. Wilson checks, dataset checks, smoke runs, and benchmark runs are also separated. The benchmark CSV and JSON files are the safest source for comparing accuracy, runtime, convergence metadata, coverage metadata, and sampler metadata.

## Troubleshooting

A missing custom split file raises a clear error. Generated split ratios must sum to 1. Wilson `avoid_capping: true` requires budget targeting. Exact Wilson checks may fail on very large graphs unless `wilson_checks.max_exact_nodes` is increased intentionally. PyTorch Geometric must be installed for real dataset loading.
