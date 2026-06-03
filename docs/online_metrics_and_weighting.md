# Online Adaptive Wilson supervisor metrics and normalized weighting

This repository now records the diagnostics discussed for online Adaptive Wilson comparisons.

## Convergence under a fixed iteration budget

For online `per_batch` training, the effective training budget is

```text
epochs × steps_per_epoch
```

The code now records these explicitly as cumulative mini-batch updates and optimizer updates. With `accumulation_steps: 1`, one mini-batch update is one optimizer update. The main fields are:

- `cumulative_minibatch_updates` in `history.csv`
- `cumulative_optimizer_updates` in `history.csv`
- `best_cumulative_minibatch_updates` in `metrics.json` and benchmark tables
- `first_cumulative_minibatch_updates_to_95pct_best_monitor` in `metrics.json` and benchmark tables
- `total_minibatch_updates` and `total_optimizer_updates`

For strict fixed-iteration comparisons, disable practical early stopping by setting `min_epochs` equal to `epochs` and using a very large `early_stopping_patience`.

## Loss as the expectation of mini-batch loss

Each online batch produces a random mini-batch loss. The epoch mean is the empirical estimator of the expected mini-batch loss under the sampler. The code now writes this explicitly as:

- `expected_minibatch_loss`
- `epoch_minibatch_loss_mean`

These appear in `history.csv`, `batch_history.csv`, `metrics.json`, and benchmark summary columns after aggregation.

## Variance of loss minus expectation

For a mini-batch loss random variable `L_t`, the variance suggested in the discussion is

```text
Var(L_t - E[L_t])
```

which is the same as `Var(L_t)`. The code estimates this within each epoch using the batch losses observed in that epoch and writes:

- `epoch_minibatch_loss_variance`
- `epoch_minibatch_loss_std`
- `minibatch_loss_minus_expectation`
- `minibatch_loss_minus_expectation_squared`
- `minibatch_loss_minus_expectation_squared_mean`

`batch_history.csv` contains one row per sampled mini-batch, so residuals can be plotted at the iteration level.

## Online normalized weighted loss

The online loss is now computed as a self-normalized inverse-inclusion-probability weighted estimator:

```text
weighted_loss = sum_i w_i loss_i / sum_i w_i
w_i ∝ 1 / pi_i
```

where `pi_i` is the probability that center node `i` is included by the sampler.

Current implementations:

- Uniform: equal inclusion probability, so all weights are equal.
- Degree: uses an inclusion-probability approximation `pi_i ≈ 1 - (1 - p_i)^k`, where `p_i` is the degree-proportional per-draw probability and `k` is the center budget.
- Wilson / Adaptive Wilson: uses Wilson root marginal probabilities. For small graphs and `weight_estimation: exact`, this is based on the diagonal of `q(L+qI)^(-1)`. For larger graphs, or when configured with `weight_estimation: monte_carlo_forests`, it estimates root marginals from repeated q-forest draws and caches them per run.

Important caveat: if Wilson sampling is configured with redraws, padding, capping, or target-search behavior, these root marginals are the principled weighting signal but may not exactly equal the final post-processed center inclusion distribution. The code records this with `wilson_online_probability_matches_final_centers_exactly`.
