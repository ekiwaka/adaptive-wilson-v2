# Roazhon13 chunked online benchmark workflow

This workflow is intended for an OAR allocation like:

```bash
oarsub -q production -p "cluster='roazhon13'" \
  -l host=1/core=12,walltime=24:00:00 \
  -I
```

Roazhon13 is a CPU cluster, so the active grid YAMLs now use:

```yaml
runtime:
  device: cpu
  torch_num_threads: 1
```

The Roazhon13 online configs are under:

```text
configs/active/grid/roazhon13/online/unweighted/
configs/active/grid/roazhon13/online/normalized/
```

Each dataset-specific online YAML runs the online samplers currently being compared:

```yaml
sampling:
  samplers:
  - uniform
  - degree
  - adaptive_wilson
```

The plain `wilson` sampler is intentionally omitted from the online Roazhon13 configs for now, because Adaptive Wilson is the Wilson-family online method being tested in this workflow. Static configs still keep `wilson` where appropriate.

Roman-empire dataset-specific active configs use MixHop:

```yaml
model:
  family: mixhop
```

For mixed all-dataset YAMLs, prefer the dataset-specific YAMLs when you need Roman-empire to use MixHop while homophilous datasets use GCN, because the current benchmark schema has one model block per YAML.

## Running 100 Roman-empire seeds in chunks

From inside the repository, after entering the OAR interactive job:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src"

bash scripts/run_roazhon13_chunked_online.sh \
  configs/active/grid/roazhon13/online/unweighted/roman_empire__all_samplers.yaml \
  100 \
  25 \
  0 \
  outputs/active/grid/roazhon13/chunked
```

This runs seeds `0..99` as four chunks of 25 seeds. Each chunk gets a separate generated YAML, log file, and output folder. Progress is tracked in:

```text
outputs/active/grid/roazhon13/chunked/_chunk_status/
```

If a tracker file already exists, the runner reads it before starting. Chunks marked `done` are skipped. The first chunk that is missing, `running`, `failed`, or otherwise not marked `done` is rerun. This means that if chunks 1 and 2 finished but chunk 3 was interrupted when the OAR session ended, the next invocation starts again from chunk 3. Use `--force` directly with `scripts/run_chunked_seed_benchmark.py` if you intentionally want to rerun completed chunks.

## Manual plotting after experiments

Plotting is disabled during the Roazhon13 benchmark runs through:

```yaml
benchmark:
  save_plots: false
```

After the chunks finish, combine the chunk folders and generate plots manually:

```bash
bash scripts/plot_roazhon13_chunked_online.sh \
  outputs/active/grid/roazhon13/chunked \
  roman-empire \
  outputs/active/grid/roazhon13/manual_plots \
  50
```

The manual plotting script reads all chunk `benchmark_detailed_runs.csv` files, writes combined `benchmark_detailed_runs.csv` and `benchmark_summary.csv` tables, and then generates two plot families after the experiments finish:

1. The original `run_benchmark.py` plot family, regenerated from the combined tables. These are written under:

```text
outputs/active/grid/roazhon13/manual_plots/<dataset>/_combined_for_plots/<weighting_regime>/<dataset>/plots/
```

2. The extra supervisor/online plots from:

```bash
python scripts/plot_extra_metrics.py
```

These are written under:

```text
outputs/active/grid/roazhon13/manual_plots/<dataset>/plots/
```

So the expensive benchmark chunks do not plot during training, but after all chunks finish you still get both the original benchmark plots and the `plot_extra_metrics.py` plots from the aggregated seed data.

## Mac smoke tests before grid submission

Before launching a long Roazhon13 job, test the same chunk/resume/plotting flow locally with the tiny Roman-empire smoke config:

```bash
bash scripts/smoke_roman_empire_chunked_online_flow.sh
```

For a tracker-only test that does not train or load Roman-empire data:

```bash
bash scripts/smoke_chunked_seed_tracker_logic.sh
```

The end-to-end smoke script intentionally runs one seed chunk first, then reruns with two chunks. The second call should skip the first completed chunk and run only the missing chunk. It then combines the chunk outputs and generates both the original benchmark plot family and the `plot_extra_metrics.py` plot family manually.

The smoke config is:

```text
configs/active/local/smoke/roman_empire_online_chunked_smoke.yaml
```

The smoke outputs are written under:

```text
outputs/smoke/roman_empire_chunked_online
outputs/smoke/roman_empire_manual_plots
```
