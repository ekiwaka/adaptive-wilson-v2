# Online expected-budget Wilson mini-batching changes

This revision replaces the previous Adaptive Wilson fixed-budget post-processing path with a cleaner expected-budget Wilson path.

## Motivation

A Wilson root set should not be capped/truncated or padded with uniform nodes if the experiment is meant to evaluate the Wilson sampling distribution. Cap/fill gives an exact final batch size, but the final batch is no longer a pure Wilson draw. For online mini-batch experiments, the cleaner comparison is:

- uniform: exactly B center nodes
- degree: exactly B center nodes
- adaptive_wilson: expected B Wilson roots, with the realised root count allowed to fluctuate naturally

The realised Wilson count is tracked in the outputs and should be reported alongside the requested budget.

## New configuration fields

`sampling.wilson.budget_targeting` now supports:

- `legacy`: old redraw/fill/cap behavior
- `q_target_search`: existing search/redraw behavior without padding/capping
- `expected_budget`: one q-Wilson draw using q calibrated for the requested budget; no truncation, no cap, no uniform fill

`sampling.adaptive_wilson.q_mode` now supports:

- `manual_alpha`: old q_i = alpha scalar-q behavior
- `precomputed_budget_q`: Adaptive Wilson ignores manual alpha/alpha_grid and resolves q from `sampling.wilson.q_table_path` for each requested budget

## New precomputation script

Use:

```bash
python scripts/precompute_online_wilson_q_table.py \
  --config configs/active/grid/online_static_informed/unweighted/roman_empire__all_samplers_fixed_batch.yaml \
  --force
```

The script writes the q table to the config's `sampling.wilson.q_table_path` unless `--out` is supplied.

The JSON table has the form expected by `WilsonCenterSampler`:

```json
{
  "kind": "online_expected_budget_wilson_q_table",
  "datasets": {
    "roman-empire": {
      "q_by_budget_count": {
        "64": {"q": 0.123, "validation_mean_roots": 63.9}
      }
    }
  }
}
```

## New fixed-batch configs

The configs under `configs/active/grid/online_static_informed/...` now use:

```yaml
sampling:
  budget_unit: count
  wilson:
    q: null
    auto_q_method: precomputed_table
    budget_targeting: expected_budget
    redraw_on_shortfall: false
    allow_uniform_fill: false
  adaptive_wilson:
    q_mode: precomputed_budget_q
    alpha_grid: []
    budget_grid_mode: fixed_budget
```

This means Adaptive Wilson uses the same target budgets as uniform/degree, but preserves the Wilson draw by matching the budget in expectation rather than by post-processing.

## Static scheme impact

The static scheme is not changed. Defaults preserve legacy behavior:

```yaml
sampling:
  wilson:
    budget_targeting: legacy
  adaptive_wilson:
    q_mode: manual_alpha
    budget_grid_mode: legacy_alpha_cap
```
