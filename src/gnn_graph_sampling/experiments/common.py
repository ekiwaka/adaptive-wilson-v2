from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence
import matplotlib.pyplot as plt

from ..config.schema import BenchmarkConfig, DatasetConfig, DiagnosticsConfig, SamplingConfig, SearchConfig, TrainConfig
from ..exceptions import ConfigurationError
from ..utils.io import create_experiment_root, dataclass_to_dict, ensure_dir, slugify, write_yaml
from ..utils.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)


def parse_config_arg(description: str) -> argparse.Namespace:
    """Parse the shared --config CLI argument."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, required=True, help='YAML configuration file.')
    return parser.parse_args()


def prepare_experiment_root(
    output_root: str,
    experiment_name: str,
    run_name: str,
    *,
    overwrite: bool,
    append_timestamp: bool = False,
    timestamp_format: str = "%Y-%m-%d-%H-%M",
) -> Path:
    """Create an experiment root and initialize standard subdirectories."""

    root = create_experiment_root(
        output_root,
        experiment_name,
        run_name,
        overwrite=overwrite,
        append_timestamp=append_timestamp,
        timestamp_format=timestamp_format,
    )
    ensure_dir(root / 'logs')
    ensure_dir(root / 'plots')
    ensure_dir(root / 'tables')
    ensure_dir(root / 'runs')
    return root


def save_resolved_config(root: Path, config: Any) -> None:
    """Save the resolved experiment config for reproducibility."""

    write_yaml(root / 'resolved_config.yaml', dataclass_to_dict(config))


def _format_float_slug(value: float | None) -> str:
    if value is None:
        return 'na'
    return f'{float(value):0.4f}'.replace('.', 'p').replace('-', 'm')


def _format_budget_slug(value: float | None) -> str:
    if value is None:
        return 'na'
    return f'{float(value):0.4f}'.replace('.', 'p').replace('-', 'm')


def budget_grid_values(sampling_config: Any) -> list[float]:
    values = getattr(sampling_config, 'budget_values', None)
    if values:
        return [float(value) for value in values]
    batch_sizes = getattr(sampling_config, 'batch_sizes', None)
    if batch_sizes:
        return [float(value) for value in batch_sizes]
    return [float(getattr(sampling_config, 'batch_size', 128))]


def build_run_directory(
    root: Path,
    *,
    dataset_name: str,
    feature_mode: str,
    sampler_name: str,
    model_family: str,
    seed: int,
    split_idx: int = 0,
    split_mode: str = 'predefined',
    batch_size: int,
    num_hops: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    aggr: str,
    jk_mode: str,
    head_hidden_dim: int,
    q: float | None,
    sampling_mode: str = 'online',
    online_schedule: str = 'per_batch',
    budget_unit: str = 'count',
    budget_value: float | None = None,
) -> Path:
    """Build a collision-resistant run directory for one concrete job."""

    parts = [
        slugify(dataset_name),
        slugify(feature_mode),
        slugify(sampler_name),
        slugify(model_family),
        f'splitmode_{slugify(split_mode)}',
        f'mode_{slugify(sampling_mode)}',
        f'sched_{slugify(online_schedule)}',
        f'budget_{slugify(budget_unit)}_{_format_budget_slug(budget_value if budget_value is not None else float(batch_size))}',
        f'bs_{int(batch_size)}',
        f'hops_{int(num_hops)}',
        f'hidden_{int(hidden_dim)}',
        f'layers_{int(num_layers)}',
        f'dropout_{_format_float_slug(dropout)}',
        f'aggr_{slugify(aggr)}',
        f'jk_{slugify(jk_mode)}',
        f'head_{int(head_hidden_dim)}',
        f'q_{_format_float_slug(q)}',
        f'split_{int(split_idx):02d}',
        f'seed_{int(seed):03d}',
    ]
    run_dir = root / 'runs'
    for part in parts:
        run_dir = run_dir / part
    return run_dir


def validate_unique_run_roots(jobs: Sequence[dict[str, Any]]) -> None:
    """Raise if any prepared jobs would write into the same run directory."""

    seen: dict[str, dict[str, Any]] = {}
    collisions: list[str] = []
    for job in jobs:
        run_root = str(job['run_root'])
        if run_root in seen:
            collisions.append(run_root)
        else:
            seen[run_root] = job
    if collisions:
        raise ConfigurationError(
            'Run-root schema collision detected. The following output directories would be reused: '
            + ', '.join(sorted(set(collisions)))
        )


def warn_if_test_metrics_are_tracked(*, track_test_metrics_during_training: bool, context: str) -> None:
    """Emit a research-hygiene warning for per-epoch test tracking."""

    if track_test_metrics_during_training:
        LOGGER.warning(
            '%s is configured to track test metrics during training. This is discouraged for tuning studies.',
            context,
        )


from typing import Any


def flatten_run_result(
    result: dict[str, Any],
    *,
    batch_size: int,
    num_hops: int,
    hidden_dim: int,
    num_layers: int,
    model_family: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Flatten a nested run result into one table row."""

    seed = result.get('seed')
    if seed is None:
        cfg = result.get('config', {})
        if isinstance(cfg, dict):
            runtime_cfg = cfg.get('runtime', {})
            if isinstance(runtime_cfg, dict):
                seed = runtime_cfg.get('seed')

    row: dict[str, Any] = {
        'dataset_name': result['dataset_name'],
        'split_idx': extra.get('split_idx'),
        'planetoid_split': extra.get('planetoid_split'),
        'dataset_kind': result['dataset_kind'],
        'feature_mode': result['feature_mode'],
        'model_family': result.get('model_family', model_family),
        'sampler_name': result['sampler_name'],
        'sampling_mode': result.get('sampling_mode', extra.get('sampling_mode', 'online')),
        'online_schedule': result.get('online_schedule', extra.get('online_schedule', 'per_batch')),
        'budget_unit': result.get('budget_unit', extra.get('budget_unit', 'count')),
        'budget_value': result.get('budget_value', extra.get('budget_value')),
        'split_mode': result.get('split_mode', extra.get('split_mode', 'predefined')),
        'training_objective': result.get('training_objective', extra.get('training_objective')),
        'resolved_budget': int(result.get('resolved_budget', 0)),
        'total_nodes': int(result.get('total_nodes', 0)),
        'train_node_count': int(result.get('train_node_count', 0)),
        'val_node_count': int(result.get('val_node_count', 0)),
        'test_node_count': int(result.get('test_node_count', 0)),
        'split_val_ratio_of_remaining': result.get('split_val_ratio_of_remaining'),
        'preprocessed_cache_path': result.get('preprocessed_cache_path'),
        'seed': int(seed) if seed is not None else None,
        'requested_device': result['requested_device'],
        'resolved_device': result['resolved_device'],
        'best_epoch': result['best_epoch'],
        'best_cumulative_minibatch_updates': result.get('best_cumulative_minibatch_updates'),
        'best_cumulative_optimizer_updates': result.get('best_cumulative_optimizer_updates'),
        'first_epoch_to_95pct_best_monitor': result.get('first_epoch_to_95pct_best_monitor'),
        'first_cumulative_minibatch_updates_to_95pct_best_monitor': result.get(
            'first_cumulative_minibatch_updates_to_95pct_best_monitor'
        ),
        'total_minibatch_updates': result.get('total_minibatch_updates'),
        'total_optimizer_updates': result.get('total_optimizer_updates'),
        'stopped_early': int(result['stopped_early']),
        'training_time_per_epoch': float(result['training_time_per_epoch']),
        'total_training_time': float(result['total_training_time']),
        'inference_latency': float(result['inference_latency']),
        'average_sampled_nodes': float(result['average_sampled_nodes']),
        'average_sampled_edges': float(result['average_sampled_edges']),
        'average_center_nodes': float(result['average_center_nodes']),
        'budget_match_ratio': float(result['budget_match_ratio']),
        'best_val_loss': float(result['best_val_loss']),
        'best_monitor_value': float(result['best_monitor_value']),
        'monitor_metric': result['monitor_metric'],
        'final_lr': float(result['final_lr']),
        'batch_size': int(batch_size),
        'num_hops': int(num_hops),
        'hidden_dim': int(hidden_dim),
        'num_layers': int(num_layers),
        'evaluation_mode': result.get('evaluation_mode'),
        'evaluation_device': result.get('evaluation_device'),
        'evaluation_batch_size': result.get('evaluation_batch_size'),
        'run_root': result.get('run_root'),
        'resolved_config_path': result.get('resolved_config_path'),
        'metrics_path': result.get('metrics_path'),
        'history_path': result.get('history_path'),
        'batch_history_path': result.get('batch_history_path'),
        'is_baseline': int(str(result['sampler_name']).startswith('baseline_')),
    }

    device_warnings = result.get('device_warnings', [])
    if isinstance(device_warnings, list):
        row['device_warnings'] = ' | '.join(str(item) for item in device_warnings)
        row['device_warning_count'] = len(device_warnings)

    for prefix in ('train_metrics', 'val_metrics', 'test_metrics'):
        metrics = result.get(prefix, {})
        if not isinstance(metrics, dict):
            continue

        short_prefix = prefix.replace('_metrics', '')
        for key, value in metrics.items():
            if isinstance(value, bool):
                row[f'{short_prefix}_{key}'] = int(value)
            elif isinstance(value, (int, float)):
                row[f'{short_prefix}_{key}'] = float(value)
            elif value is None:
                row[f'{short_prefix}_{key}'] = None

    sampler_diagnostics = result.get('sampler_diagnostics', {})
    if isinstance(sampler_diagnostics, dict):
        for key, value in sampler_diagnostics.items():
            if isinstance(value, bool):
                row[key] = int(value)
            elif isinstance(value, (int, float)):
                row[key] = float(value)
            elif value is None:
                row[key] = None

    split_metadata = result.get('split_metadata', {})
    if isinstance(split_metadata, dict):
        for key, value in split_metadata.items():
            column = f'splitmeta_{key}'
            if isinstance(value, bool):
                row[column] = int(value)
            elif isinstance(value, (int, float, str)) or value is None:
                row[column] = value

    for key, value in extra.items():
        if isinstance(value, bool):
            row[key] = int(value)
        elif isinstance(value, (int, float, str)) or value is None:
            row[key] = value
        else:
            row[key] = str(value)

    return row


def aggregate_rows(rows: Sequence[dict[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    """Aggregate scalar metrics over repeated runs.

    This version is robust to rows that do not share exactly the same set of
    numeric keys, which happens when trivial baselines are mixed with graph
    training results.
    """

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    numeric_keys: list[str] = []
    if rows:
        seen: set[str] = set()
        for row in rows:
            for key, value in row.items():
                if key in group_keys or key in seen:
                    continue
                if isinstance(value, (int, float)):
                    seen.add(key)
                    numeric_keys.append(key)

    summaries: list[dict[str, Any]] = []
    for key, records in groups.items():
        summary = {group_keys[index]: key[index] for index in range(len(group_keys))}
        summary["num_runs"] = len(records)

        for metric_key in numeric_keys:
            values = [
                float(record[metric_key])
                for record in records
                if metric_key in record and isinstance(record[metric_key], (int, float))
            ]
            if not values:
                continue

            mean_value = float(sum(values) / len(values))
            summary[f"mean_{metric_key}"] = mean_value
            if len(values) > 1:
                summary[f"std_{metric_key}"] = float(
                    (sum((value - mean_value) ** 2 for value in values) / len(values)) ** 0.5
                )
            else:
                summary[f"std_{metric_key}"] = 0.0

        summaries.append(summary)

    summaries.sort(key=lambda item: tuple(str(item[key]) for key in group_keys))
    return summaries


def sort_rows_by_metric(rows: Sequence[dict[str, Any]], metric_key: str, *, descending: bool = True) -> list[dict[str, Any]]:
    """Sort summary rows by one scalar metric."""

    return sorted(rows, key=lambda row: float(row[metric_key]), reverse=descending)


def _scatter_rows(rows: Sequence[dict[str, Any]], out_path: Path, *, x_key: str, y_key: str, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for sampler_name in sorted({str(row['sampler_name']) for row in rows}):
        subset = [row for row in rows if str(row['sampler_name']) == sampler_name]
        plt.scatter(
            [float(row[x_key]) for row in subset],
            [float(row[y_key]) for row in subset],
            label=sampler_name,
            alpha=0.8,
        )
    plt.xlabel(x_key.replace('_', ' '))
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def plot_runtime_scatter(rows: Sequence[dict[str, Any]], out_path: Path, *, y_key: str, title: str) -> None:
    """Create a runtime-vs-metric scatter plot."""

    _scatter_rows(rows, out_path, x_key='total_training_time', y_key=y_key, title=title)


def plot_metric_vs_runtime_by_batch_size(rows: Sequence[dict[str, Any]], out_path: Path, *, metric_key: str) -> None:
    """Create a batch-size-conditioned runtime trade-off plot."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for batch_size in sorted({int(row['batch_size']) for row in rows}):
        subset = [row for row in rows if int(row['batch_size']) == batch_size]
        plt.scatter(
            [float(row['mean_total_training_time']) for row in subset],
            [float(row[f'mean_{metric_key}']) for row in subset],
            label=f'bs={batch_size}',
            alpha=0.8,
        )
    plt.xlabel('Mean total training time (s)')
    plt.ylabel(metric_key)
    plt.title(f'{metric_key} vs runtime by batch size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def plot_rows_separated_by_dataset_kind(
    rows: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    plotter,
    filename: str,
    **plot_kwargs: Any,
) -> None:
    """Generate separate real/synthetic plots when both dataset kinds are present."""

    if not rows:
        return
    kinds = sorted({str(row.get('dataset_kind', 'unknown')) for row in rows})
    for kind in kinds:
        subset = [row for row in rows if str(row.get('dataset_kind', 'unknown')) == kind]
        if subset:
            kind_dir = out_dir / kind
            kind_dir.mkdir(parents=True, exist_ok=True)
            plotter(subset, kind_dir / filename, **plot_kwargs)


def train_config_from_benchmark(
    benchmark_config: BenchmarkConfig,
    *,
    dataset_name: str,
    feature_mode: str,
    sampler_name: str,
    batch_size: int,
    num_hops: int,
    seed: int,
    split_idx: int | None = None,
    sampling_mode: str = 'online',
    online_schedule: str = 'per_batch',
    budget_value: float | None = None,
) -> TrainConfig:
    """Project one benchmark grid point into a concrete training config."""

    dataset = DatasetConfig(
        name=dataset_name,
        root=benchmark_config.data.root,
        split_idx=benchmark_config.data.split_idx if split_idx is None else int(split_idx),
        planetoid_split=benchmark_config.data.planetoid_split,
        feature_mode=feature_mode,
        standardize_features=benchmark_config.data.standardize_features,
        split=benchmark_config.data.split,
        synthetic=benchmark_config.data.synthetic,
    )
    sampling = SamplingConfig(
        name=sampler_name,
        mode=sampling_mode,
        online_schedule=online_schedule,
        budget_unit=benchmark_config.sampling.budget_unit,
        budget_value=budget_value,
        batch_size=batch_size,
        steps_per_epoch=benchmark_config.sampling.steps_per_epoch,
        eval_batch_size=benchmark_config.sampling.eval_batch_size,
        num_hops=num_hops,
        sample_inference=benchmark_config.sampling.sample_inference,
        inference_max_batches=benchmark_config.sampling.inference_max_batches,
        cache_preprocessed_graph=benchmark_config.sampling.cache_preprocessed_graph,
        reuse_cached_preprocessed_graph=benchmark_config.sampling.reuse_cached_preprocessed_graph,
        cache_dirname=benchmark_config.sampling.cache_dirname,
        wilson=benchmark_config.sampling.wilson,
        adaptive_wilson=benchmark_config.sampling.adaptive_wilson,
    )
    run_name = (
        f'{dataset_name}_{feature_mode}_{sampler_name}_splitmode{dataset.split.mode}_split{dataset.split_idx}_seed{seed}'
        f'_{benchmark_config.model.family}'
        f'_mode{sampling_mode}_sched{online_schedule}'
        f'_budget{_format_budget_slug(budget_value if budget_value is not None else float(batch_size))}_{benchmark_config.sampling.budget_unit}'
        f'_bs{batch_size}_hops{num_hops}_hd{benchmark_config.model.hidden_dim}'
        f'_L{benchmark_config.model.num_layers}_drop{_format_float_slug(benchmark_config.model.dropout)}'
        f'_aggr{benchmark_config.model.aggr}_jk{benchmark_config.model.jk_mode}'
        f'_head{int(benchmark_config.model.head_hidden_dim or 0)}'
        f'_q{_format_float_slug(benchmark_config.sampling.wilson.q)}'
    )
    return TrainConfig(
        run_name=run_name,
        experiment=benchmark_config.experiment,
        runtime=replace(benchmark_config.runtime, seed=seed),
        output=benchmark_config.output,
        logging=benchmark_config.logging,
        dataset=dataset,
        model=benchmark_config.model,
        optimization=benchmark_config.optimization,
        sampling=sampling,
        evaluation=benchmark_config.evaluation,
    )


def configure_study_logging(root: Path, level: str, *, log_to_stdout: bool) -> None:
    """Configure logging for a study root."""

    configure_logging(level, root / 'logs' / 'study.log', log_to_stdout=log_to_stdout, reset_handlers=True)


def maybe_parallel_map(function, jobs: Iterable[Any], *, parallel_workers: int) -> list[Any]:
    """Execute jobs sequentially or with a process pool depending on configuration."""

    jobs = list(jobs)
    if parallel_workers <= 1:
        return [function(job) for job in jobs]
    results: list[Any] = []
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {executor.submit(function, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def train_config_from_search(
    search_config: SearchConfig,
    *,
    dataset_name: str,
    feature_mode: str,
    sampler_name: str,
    batch_size: int,
    num_hops: int,
    seed: int,
    model,
    wilson_q: float | None,
    sampling_mode: str = 'online',
    online_schedule: str = 'per_batch',
    budget_value: float | None = None,
) -> TrainConfig:
    """Project one search grid point into a concrete training config."""

    dataset = DatasetConfig(
        name=dataset_name,
        root=search_config.data.root,
        split_idx=search_config.data.split_idx,
        feature_mode=feature_mode,
        standardize_features=search_config.data.standardize_features,
        split=search_config.data.split,
        synthetic=search_config.data.synthetic,
    )
    sampling = SamplingConfig(
        name=sampler_name,
        mode=sampling_mode,
        online_schedule=online_schedule,
        budget_unit=search_config.sampling.budget_unit,
        budget_value=budget_value,
        batch_size=batch_size,
        steps_per_epoch=search_config.sampling.steps_per_epoch,
        eval_batch_size=search_config.sampling.eval_batch_size,
        num_hops=num_hops,
        sample_inference=search_config.sampling.sample_inference,
        inference_max_batches=search_config.sampling.inference_max_batches,
        cache_preprocessed_graph=search_config.sampling.cache_preprocessed_graph,
        reuse_cached_preprocessed_graph=search_config.sampling.reuse_cached_preprocessed_graph,
        cache_dirname=search_config.sampling.cache_dirname,
        wilson=replace(search_config.sampling.wilson, q=wilson_q),
        adaptive_wilson=search_config.sampling.adaptive_wilson,
    )
    run_name = (
        f'{dataset_name}_{feature_mode}_{sampler_name}_splitmode{dataset.split.mode}_split{dataset.split_idx}_seed{seed}'
        f'_{model.family}'
        f'_mode{sampling_mode}_sched{online_schedule}'
        f'_budget{_format_budget_slug(budget_value if budget_value is not None else float(batch_size))}_{search_config.sampling.budget_unit}'
        f'_bs{batch_size}_hops{num_hops}_hd{model.hidden_dim}_L{model.num_layers}'
        f'_drop{_format_float_slug(model.dropout)}_aggr{model.aggr}_jk{model.jk_mode}'
        f'_head{int(model.head_hidden_dim or 0)}_q{_format_float_slug(wilson_q)}'
    )
    return TrainConfig(
        run_name=run_name,
        runtime=replace(search_config.runtime, seed=seed),
        output=search_config.output,
        logging=search_config.logging,
        dataset=dataset,
        model=model,
        optimization=search_config.optimization,
        sampling=sampling,
        evaluation=search_config.evaluation,
    )
