# Roman-empire Mac smoke tests for the chunked workflow

These smoke tests are meant to check the new Roazhon13-style chunked workflow on a Mac before submitting long grid jobs.

They use a tiny Roman-empire online config with:

- CPU only
- MixHop
- `uniform`, `degree`, and `adaptive_wilson`
- no plain `wilson` in online mode
- `benchmark.save_plots: false`, so chunks only train and write tables
- manual plotting only after chunks finish
- two tiny seed chunks by default

## Fast tracker-only test

This does not train a model. It only checks that the chunk tracker resumes from the first unfinished chunk.

```bash
bash scripts/smoke_chunked_seed_tracker_logic.sh
```

## End-to-end Roman-empire smoke test

This runs a very small Roman-empire online experiment, resumes from the existing tracker, combines chunk outputs, and generates both plot families manually.

```bash
bash scripts/smoke_roman_empire_chunked_online_flow.sh
```

The default outputs are:

```text
outputs/smoke/roman_empire_chunked_online
outputs/smoke/roman_empire_manual_plots
```

The shell script deliberately runs the first chunk alone, then runs the full two-chunk job. On the second call, the first chunk should be skipped because the tracker marks it as done.

## Useful overrides

```bash
TOTAL_RUNS=3 CHUNK_SIZE=1 START_SEED=9100 \
  bash scripts/smoke_roman_empire_chunked_online_flow.sh
```

To keep earlier smoke outputs instead of deleting them first:

```bash
KEEP_SMOKE_OUTPUTS=1 bash scripts/smoke_roman_empire_chunked_online_flow.sh
```

## Smoke config

The config used by default is:

```text
configs/active/local/smoke/roman_empire_online_chunked_smoke.yaml
```

It is intentionally much smaller than the Roazhon13 grid configs, so do not use it for final results.
