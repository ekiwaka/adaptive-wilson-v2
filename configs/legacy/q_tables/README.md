# Wilson q table directory

The benchmark configs in `configs/benchmark/sampler_split/*_qtable.yaml` use per-dataset q table paths under `outputs/q_tables/`.
Generate the actual tables before running the benchmark:

```bash
PYTHONPATH=src python scripts/calibrate_wilson_q.py \
  --config configs/benchmark/sampler_split/citeseer_budget_sweep_static_qtable.yaml
```

The lookup table is keyed by resolved integer budget count, not ratio, because the sampler receives budget counts at runtime.
