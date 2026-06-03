from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..config.loader import load_config
from ..config.schema import BenchmarkConfig, DatasetConfig
from ..utils.io import write_csv, write_json
from ..utils.logging import get_logger
from .baselines import compute_trivial_baselines
from .common import (
    aggregate_rows,
    budget_grid_values,
    build_run_directory,
    configure_study_logging,
    flatten_run_result,
    maybe_parallel_map,
    parse_config_arg,
    plot_metric_vs_runtime_by_batch_size,
    plot_rows_separated_by_dataset_kind,
    plot_runtime_scatter,
    prepare_experiment_root,
    save_resolved_config,
    train_config_from_benchmark,
    validate_unique_run_roots,
    warn_if_test_metrics_are_tracked,
)
from .train import run_training_experiment

LOGGER = get_logger(__name__)


def _run_one(job: dict[str, Any]) -> dict[str, Any]:
    train_config = job['train_config']
    run_root = Path(job['run_root'])
    result, _ = run_training_experiment(train_config, run_root=run_root, configure_root_logging=True)
    return {'result': result.to_dict(), 'job': job['metadata']}


def _baseline_rows(config: BenchmarkConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if config.data.split.mode != 'predefined':
        return [], []

    rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    for dataset_name in config.data.datasets:
        for feature_mode in config.data.feature_modes:
            dataset_config = DatasetConfig(
                name=dataset_name,
                root=config.data.root,
                split_idx=config.data.split_idx,
                feature_mode=feature_mode,
                standardize_features=config.data.standardize_features,
                split=config.data.split,
                synthetic=config.data.synthetic,
            )
            for run_index in range(config.benchmark.runs):
                seed = config.benchmark.base_seed + run_index
                for result in compute_trivial_baselines(
                    dataset_config,
                    seed=seed,
                    monitor_metric=config.optimization.monitor_metric,
                ):
                    detailed.append(result)
                    rows.append(
                        flatten_run_result(
                            result,
                            batch_size=0,
                            num_hops=0,
                            hidden_dim=0,
                            num_layers=0,
                            model_family='baseline',
                            dropout=0.0,
                            aggr='none',
                            jk_mode='none',
                            head_hidden_dim=0,
                            q=None,
                            adaptive_alpha=None,
                            sampling_mode='baseline',
                            online_schedule='none',
                            budget_unit='count',
                            budget_value=None,
                            split_mode=config.data.split.mode,
                            training_objective='baseline',
                        )
                    )
    return rows, detailed


def _mode_label(row: dict[str, Any]) -> str:
    if row.get('sampling_mode') == 'static':
        return 'static'
    return f"online_{row.get('online_schedule', 'per_batch')}"


def _format_alpha_slug(value: float | None) -> str:
    if value is None:
        return 'na'
    text = f'{float(value):0.6f}'.replace('.', 'p').replace('-', 'm')
    return text.rstrip('0').rstrip('p')


def _adaptive_alpha_values(config: BenchmarkConfig, sampler_name: str) -> list[float | None]:
    if sampler_name != 'adaptive_wilson':
        return [None]

    grid = list(config.sampling.adaptive_wilson.alpha_grid)
    if grid:
        return [float(value) for value in grid]

    return [float(config.sampling.adaptive_wilson.alpha)]


def _budget_values_for_sampler(
    config: BenchmarkConfig,
    sampler_name: str,
    budget_values: list[float],
) -> list[float | None]:
    if sampler_name == 'none':
        return [None]

    # For Adaptive Wilson alpha sweeps, alpha is intended to control the
    # realized number of Wilson roots. The sampler still needs a budget/cap,
    # so for ratio budgets we use a loose cap of 1.0. The plot then uses the
    # actual mean sampled center-node count as the x-axis.
    if sampler_name == 'adaptive_wilson' and config.sampling.adaptive_wilson.alpha_grid:
        if config.sampling.budget_unit == 'ratio':
            return [1.0]
        if budget_values:
            return [max(float(value) for value in budget_values)]

    return list(budget_values)


def _budget_x_value(row: dict[str, Any]) -> float:
    for key in (
        'mean_average_center_nodes',
        'average_center_nodes',
        'mean_resolved_budget',
        'resolved_budget',
    ):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _plot_metric_vs_budget(summary_rows: list[dict[str, Any]], out_path: Path, *, metric_key: str) -> None:
    plotted = [row for row in summary_rows if not str(row.get('sampler_name', '')).startswith('baseline_')]
    if not plotted:
        return

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plotted:
        grouped[(str(row['dataset_name']), str(row.get('split_mode', 'predefined')), _mode_label(row))].append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for (dataset_name, split_mode, mode_label), group in grouped.items():
        plt.figure(figsize=(8, 5))

        for sampler_name in sorted({str(item['sampler_name']) for item in group}):
            subset = [item for item in group if str(item['sampler_name']) == sampler_name]
            subset.sort(key=_budget_x_value)

            xs = [_budget_x_value(item) for item in subset]
            ys = [float(item.get(f'mean_{metric_key}', 0.0) or 0.0) for item in subset]

            plt.plot(xs, ys, marker='o', label=sampler_name)

            if sampler_name == 'adaptive_wilson':
                for x, y, item in zip(xs, ys, subset):
                    alpha = item.get('adaptive_alpha')
                    if isinstance(alpha, (int, float)):
                        plt.annotate(
                            f'α={float(alpha):g}',
                            (x, y),
                            textcoords='offset points',
                            xytext=(5, 5),
                            fontsize=8,
                        )

        plt.xlabel('Mean sampled center nodes')
        plt.ylabel(metric_key)
        plt.title(f'{dataset_name} | {split_mode} | {mode_label} | {metric_key} vs actual budget')
        plt.grid(True, alpha=0.3)
        plt.legend()

        safe_dataset = dataset_name.replace('/', '_')
        group_dir = out_path.parent / safe_dataset / split_mode / mode_label
        group_dir.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(group_dir / out_path.name, dpi=180, bbox_inches='tight')
        plt.close()


def write_benchmark_plots_from_rows(
    *,
    flat_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    root: Path,
) -> None:
    """Write the same plot family normally produced by run_benchmark_study.

    This is deliberately factored out so chunked grid experiments can disable
    plotting during each chunk and regenerate the benchmark plots once, after
    all seed chunks have finished and their tables have been combined.
    """
    (root / 'plots').mkdir(parents=True, exist_ok=True)
    if flat_rows:
        plot_runtime_scatter(
            flat_rows,
            root / 'plots' / 'train_accuracy_vs_runtime.png',
            y_key='train_accuracy',
            title='Train accuracy vs runtime',
        )
        plot_runtime_scatter(
            flat_rows,
            root / 'plots' / 'test_accuracy_vs_runtime.png',
            y_key='test_accuracy',
            title='Test accuracy vs runtime',
        )
        plot_rows_separated_by_dataset_kind(
            flat_rows,
            root / 'plots',
            plotter=plot_runtime_scatter,
            filename='train_accuracy_vs_runtime.png',
            y_key='train_accuracy',
            title='Train accuracy vs runtime',
        )
        plot_rows_separated_by_dataset_kind(
            flat_rows,
            root / 'plots',
            plotter=plot_runtime_scatter,
            filename='test_accuracy_vs_runtime.png',
            y_key='test_accuracy',
            title='Test accuracy vs runtime',
        )

    if summary_rows:
        for metric_key in (
            'train_accuracy',
            'val_accuracy',
            'test_accuracy',
            'train_macro_f1',
            'val_macro_f1',
            'test_macro_f1',
            'train_balanced_accuracy',
            'val_balanced_accuracy',
            'test_balanced_accuracy',
            'test_weighted_f1',
        ):
            plot_metric_vs_runtime_by_batch_size(
                summary_rows,
                root / 'plots' / f'{metric_key}_vs_runtime_by_batch_size.png',
                metric_key=metric_key,
            )
            plot_rows_separated_by_dataset_kind(
                summary_rows,
                root / 'plots',
                plotter=plot_metric_vs_runtime_by_batch_size,
                filename=f'{metric_key}_vs_runtime_by_batch_size.png',
                metric_key=metric_key,
            )
            _plot_metric_vs_budget(
                summary_rows,
                root / 'plots' / f'{metric_key}_vs_budget.png',
                metric_key=metric_key,
            )


def run_benchmark_study(config: BenchmarkConfig) -> Path:
    root = prepare_experiment_root(
        config.output.root,
        config.output.experiment_name,
        config.study_name,
        overwrite=config.output.overwrite,
        append_timestamp=config.output.append_timestamp,
        timestamp_format=config.output.timestamp_format,
    )
    configure_study_logging(root, config.logging.level, log_to_stdout=config.logging.log_to_stdout)
    save_resolved_config(root, config)
    warn_if_test_metrics_are_tracked(
        track_test_metrics_during_training=config.evaluation.track_test_metrics_during_training,
        context='benchmark study',
    )

    jobs: list[dict[str, Any]] = []
    budget_values = budget_grid_values(config.sampling)
    split_indices = list(config.data.split_indices) if config.data.split_indices else [int(config.data.split_idx)]

    for dataset_name in config.data.datasets:
        for feature_mode in config.data.feature_modes:
            for split_idx in split_indices:
                for sampler_name in config.sampling.samplers:
                    sampler_budget_values = _budget_values_for_sampler(config, sampler_name, budget_values)
                    adaptive_alpha_values = _adaptive_alpha_values(config, sampler_name)

                    for adaptive_alpha in adaptive_alpha_values:
                        for mode in config.sampling.modes:
                            schedules = ['none'] if mode == 'static' else list(config.sampling.online_schedules)

                            for online_schedule in schedules:
                                for budget_value in sampler_budget_values:
                                    if budget_value is None:
                                        batch_size = int(config.sampling.batch_sizes[0])
                                    else:
                                        batch_size = (
                                            int(round(budget_value))
                                            if config.sampling.budget_unit == 'count'
                                            else int(config.sampling.batch_sizes[0])
                                        )

                                    for num_hops in config.sampling.num_hops:
                                        for run_index in range(config.benchmark.runs):
                                            seed = config.benchmark.base_seed + run_index

                                            train_config = train_config_from_benchmark(
                                                config,
                                                dataset_name=dataset_name,
                                                feature_mode=feature_mode,
                                                sampler_name=sampler_name,
                                                batch_size=int(batch_size),
                                                num_hops=int(num_hops),
                                                seed=int(seed),
                                                split_idx=int(split_idx),
                                                sampling_mode=mode,
                                                online_schedule=online_schedule if mode != 'static' else 'per_batch',
                                                budget_value=(None if budget_value is None else float(budget_value)),
                                            )

                                            if sampler_name == 'adaptive_wilson' and adaptive_alpha is not None:
                                                train_config.sampling.adaptive_wilson = replace(
                                                    train_config.sampling.adaptive_wilson,
                                                    alpha=float(adaptive_alpha),
                                                    alpha_grid=[float(adaptive_alpha)],
                                                )
                                                train_config.run_name = (
                                                    f'{train_config.run_name}_alpha{_format_alpha_slug(float(adaptive_alpha))}'
                                                )

                                            run_root = build_run_directory(
                                                root,
                                                dataset_name=dataset_name,
                                                feature_mode=feature_mode,
                                                sampler_name=sampler_name,
                                                model_family=config.model.family,
                                                seed=int(seed),
                                                split_idx=int(split_idx),
                                                split_mode=config.data.split.mode,
                                                batch_size=int(batch_size),
                                                num_hops=int(num_hops),
                                                hidden_dim=int(config.model.hidden_dim),
                                                num_layers=int(config.model.num_layers),
                                                dropout=float(config.model.dropout),
                                                aggr=config.model.aggr,
                                                jk_mode=config.model.jk_mode,
                                                head_hidden_dim=int(config.model.head_hidden_dim or 0),
                                                q=config.sampling.wilson.q,
                                                sampling_mode=mode,
                                                online_schedule=online_schedule if mode != 'static' else 'per_batch',
                                                budget_unit=config.sampling.budget_unit,
                                                budget_value=(None if budget_value is None else float(budget_value)),
                                            )

                                            if sampler_name == 'adaptive_wilson' and adaptive_alpha is not None:
                                                run_root = run_root / f'adaptive_alpha_{_format_alpha_slug(float(adaptive_alpha))}'

                                            metadata = {
                                                'model_family': config.model.family,
                                                'batch_size': int(batch_size),
                                                'num_hops': int(num_hops),
                                                'hidden_dim': int(config.model.hidden_dim),
                                                'num_layers': int(config.model.num_layers),
                                                'dropout': float(config.model.dropout),
                                                'aggr': config.model.aggr,
                                                'jk_mode': config.model.jk_mode,
                                                'head_hidden_dim': int(config.model.head_hidden_dim or 0),
                                                'q': config.sampling.wilson.q,
                                                'adaptive_alpha': (
                                                    None if adaptive_alpha is None else float(adaptive_alpha)
                                                ),
                                                'split_idx': int(split_idx),
                                                'planetoid_split': config.data.planetoid_split,
                                                'split_mode': config.data.split.mode,
                                                'training_objective': (
                                                    'full_graph_semi_supervised'
                                                    if config.data.split.mode == 'sampler'
                                                    else 'sampled_subgraph'
                                                ),
                                                'sampling_mode': mode,
                                                'online_schedule': online_schedule if mode != 'static' else 'per_batch',
                                                'budget_unit': config.sampling.budget_unit,
                                                'budget_value': (
                                                    None if budget_value is None else float(budget_value)
                                                ),
                                            }

                                            jobs.append(
                                                {
                                                    'train_config': train_config,
                                                    'run_root': str(run_root),
                                                    'metadata': metadata,
                                                }
                                            )

    validate_unique_run_roots(jobs)
    detailed_results = maybe_parallel_map(_run_one, jobs, parallel_workers=config.benchmark.parallel_workers)

    flat_rows = [
        flatten_run_result(
            item['result'],
            batch_size=item['job']['batch_size'],
            num_hops=item['job']['num_hops'],
            model_family=item['job']['model_family'],
            hidden_dim=item['job']['hidden_dim'],
            num_layers=item['job']['num_layers'],
            dropout=item['job']['dropout'],
            aggr=item['job']['aggr'],
            jk_mode=item['job']['jk_mode'],
            head_hidden_dim=item['job']['head_hidden_dim'],
            q=item['job']['q'],
            adaptive_alpha=item['job'].get('adaptive_alpha'),
            split_idx=item['job'].get('split_idx'),
            planetoid_split=item['job'].get('planetoid_split'),
            sampling_mode=item['job']['sampling_mode'],
            online_schedule=item['job']['online_schedule'],
            budget_unit=item['job']['budget_unit'],
            budget_value=item['job']['budget_value'],
            split_mode=item['job']['split_mode'],
            training_objective=item['job']['training_objective'],
        )
        for item in detailed_results
    ]

    baseline_rows, baseline_detailed = _baseline_rows(config)
    flat_rows.extend(baseline_rows)
    detailed_results.extend({'result': item, 'job': {'is_baseline': True}} for item in baseline_detailed)

    summary_rows = aggregate_rows(
        flat_rows,
        group_keys=[
            'dataset_name',
            'dataset_kind',
            'feature_mode',
            'planetoid_split',
            'split_idx',
            'split_mode',
            'training_objective',
            'model_family',
            'sampler_name',
            'adaptive_alpha',
            'sampling_mode',
            'online_schedule',
            'budget_unit',
            'budget_value',
            'batch_size',
            'num_hops',
            'hidden_dim',
            'num_layers',
            'dropout',
            'aggr',
            'jk_mode',
            'head_hidden_dim',
            'q',
        ],
    )

    write_json(root / 'tables' / 'benchmark_detailed_runs.json', {'detailed_results': detailed_results})
    write_csv(root / 'tables' / 'benchmark_detailed_runs.csv', flat_rows)
    write_json(root / 'tables' / 'benchmark_summary.json', {'summary_rows': summary_rows})
    write_csv(root / 'tables' / 'benchmark_summary.csv', summary_rows)

    if config.benchmark.save_plots:
        write_benchmark_plots_from_rows(flat_rows=flat_rows, summary_rows=summary_rows, root=root)

    LOGGER.info('Benchmark study complete. Outputs written to %s', root)
    return root


def main() -> None:
    args = parse_config_arg('Run a multi-dataset benchmark study.')
    config = load_config(BenchmarkConfig, args.config)
    run_benchmark_study(config)


if __name__ == '__main__':  # pragma: no cover
    main()
