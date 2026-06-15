# Online static-informed fixed-batch changes

This revision adds a clean online mini-batch experimental path without changing the existing static experiment path.

## Static-informed initial train splits

The recommended online train splits were selected from the uploaded static Wilson results using this rule: choose the smallest Wilson static train fraction whose mean validation macro-F1 is at least 98% of the best Wilson validation macro-F1 for that dataset and weighting regime. This avoids blindly using the largest static split while keeping the initial labelled pool close to the static Wilson performance plateau.

Normalized:

- amazon-ratings: train_ratio 0.20, val_ratio 0.20, test_ratio 0.60; fixed batch budgets 64, 128, 256, 512.
- pubmed: train_ratio 0.30, val_ratio 0.20, test_ratio 0.50; fixed batch budgets 64, 128, 256, 512.
- roman-empire: train_ratio 0.25, val_ratio 0.20, test_ratio 0.55; fixed batch budgets 64, 128, 256, 512.

Unweighted:

- amazon-ratings: train_ratio 0.10, val_ratio 0.20, test_ratio 0.70; fixed batch budgets 32, 64, 128, 256.
- pubmed: train_ratio 0.10, val_ratio 0.20, test_ratio 0.70; fixed batch budgets 32, 64, 128, 256.
- roman-empire: train_ratio 0.25, val_ratio 0.20, test_ratio 0.55; fixed batch budgets 64, 128, 256, 512.

## New configs

The new configs are under:

- `configs/active/grid/online_static_informed/normalized/`
- `configs/active/grid/online_static_informed/unweighted/`

They use:

- `sampling.budget_unit: count`
- fixed budget counts instead of large train-ratio budgets
- `sampling.adaptive_wilson.budget_grid_mode: fixed_budget`
- `redraw_on_shortfall: true` and `allow_uniform_fill: true` so samplers realize comparable center-node counts
- separate split files such as `generated_static_informed_train0p25_seed0.pt` so old 0.6 train splits are not accidentally reused

## New online diagnostics

Online per-batch training now records:

- sampling time
- subgraph construction time
- online weight computation time
- device transfer time
- forward/loss time
- backward time
- optimizer step time
- total sampled-batch time
- memory snapshot in MB
- budget relative error
- reweighting target metadata

History plots now include:

- validation error over iterations
- test error over iterations
- mini-batch loss-estimator variance over iterations
- mini-batch loss std over iterations
- mean squared mini-batch loss residual over iterations
- sampled-batch time over iterations
- sampling/subgraph/model-compute time over iterations
- memory snapshot over iterations

## Gradient reweighting option

`optimization.reweighting_target` now supports:

- `loss`: default existing weighted-loss interpretation
- `gradient`: gradient-estimator interpretation

Because sampling weights are detached, weighted-loss backpropagation and weighted per-example gradient aggregation are mathematically equivalent. The option is therefore recorded as an experimental interpretation/diagnostic switch without changing static defaults.
