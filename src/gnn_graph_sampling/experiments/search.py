from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt

from ..config.loader import load_config
from ..config.schema import DatasetConfig, ModelConfig, SearchConfig
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
    plot_rows_separated_by_dataset_kind,
    prepare_experiment_root,
    save_resolved_config,
    sort_rows_by_metric,
    train_config_from_search,
    validate_unique_run_roots,
    warn_if_test_metrics_are_tracked,
)

LOGGER = get_logger(__name__)



def _iter_model_configs(config: SearchConfig) -> list[ModelConfig]:
    models: list[ModelConfig] = []
    for family in config.model_search.families:
        for hidden_dim in config.model_search.hidden_dims:
            for num_layers in config.model_search.num_layers:
                for dropout in config.model_search.dropouts:
                    for aggr in config.model_search.aggrs:
                        for jk_mode in config.model_search.jk_modes:
                            for head_hidden_dim in config.model_search.head_hidden_dims:
                                models.append(
                                    ModelConfig(
                                        family=family,
                                        hidden_dim=int(hidden_dim),
                                        num_layers=int(num_layers),
                                        dropout=float(dropout),
                                        aggr=aggr,
                                        use_batch_norm=config.model_search.use_batch_norm,
                                        residual=config.model_search.residual,
                                        jk_mode=jk_mode,
                                        head_hidden_dim=int(head_hidden_dim),
                                    )
                                )
    return models



def _q_values_for_sampler(config: SearchConfig, sampler_name: str) -> list[float | None]:
    if sampler_name == 'wilson':
        return [float(value) for value in sorted(set(config.sampling.wilson_q_values))]
    return [None]



def _run_one(job: dict[str, Any]) -> dict[str, Any]:
    from .train import run_training_experiment

    train_config = job['train_config']
    run_root = Path(job['run_root'])
    result, _ = run_training_experiment(train_config, run_root=run_root, configure_root_logging=True)
    return {'result': result.to_dict(), 'job': job['metadata']}



def _plot_runtime_tradeoff(summary_rows: list[dict[str, Any]], out_path: Path, *, metric_key: str) -> None:
    plt.figure(figsize=(8, 6))
    for sampler_name in sorted({str(row['sampler_name']) for row in summary_rows}):
        subset = [row for row in summary_rows if str(row['sampler_name']) == sampler_name]
        plt.scatter(
            [float(row['mean_total_training_time']) for row in subset],
            [float(row[metric_key]) for row in subset],
            label=sampler_name,
            alpha=0.8,
        )
    plt.xlabel('mean total training time (s)')
    plt.ylabel(metric_key)
    plt.title(f'{metric_key} versus runtime')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()



def _baseline_rows(config: SearchConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
                synthetic=config.data.synthetic,
            )
            for run_index in range(config.search.runs):
                seed = config.search.base_seed + run_index
                for result in compute_trivial_baselines(dataset_config, seed=seed, monitor_metric=config.optimization.monitor_metric):
                    detailed.append(result)
                    rows.append(
                        flatten_run_result(
                            result,
                            q=None,
                            batch_size=0,
                            num_hops=0,
                            hidden_dim=0,
                            num_layers=0,
                            model_family='baseline',
                            dropout=0.0,
                            aggr='none',
                            jk_mode='none',
                            head_hidden_dim=0,
                            monitor_metric=config.optimization.monitor_metric,
                            sampling_mode='baseline',
                            online_schedule='none',
                            budget_unit='count',
                            budget_value=None,
                        )
                    )
    return rows, detailed



def run_search_study(config: SearchConfig) -> Path:
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
        context='search study',
    )

    jobs: list[dict[str, Any]] = []
    model_configs = _iter_model_configs(config)
    budget_values = budget_grid_values(config.sampling)
    for dataset_name in config.data.datasets:
        for feature_mode in config.data.feature_modes:
            for sampler_name in config.sampling.samplers:
                for mode in config.sampling.modes:
                    schedules = ['none'] if mode == 'static' else list(config.sampling.online_schedules)
                    for online_schedule in schedules:
                        for budget_value in budget_values:
                            batch_size = int(round(budget_value)) if config.sampling.budget_unit == 'count' else int(config.sampling.batch_sizes[0])
                            for model_config in model_configs:
                                num_hops = int(model_config.num_layers)
                                if num_hops not in {int(value) for value in config.sampling.num_hops}:
                                    raise ValueError(
                                        f'No matching sampling.num_hops value for searched model depth {num_hops}. '
                                        'sampling.num_hops must exactly match model_search.num_layers.'
                                    )
                                for wilson_q in _q_values_for_sampler(config, sampler_name):
                                    for run_index in range(config.search.runs):
                                        seed = config.search.base_seed + run_index
                                        train_config = train_config_from_search(
                                            config,
                                            dataset_name=dataset_name,
                                            feature_mode=feature_mode,
                                            sampler_name=sampler_name,
                                            batch_size=int(batch_size),
                                            num_hops=num_hops,
                                            seed=int(seed),
                                            model=model_config,
                                            wilson_q=wilson_q,
                                            sampling_mode=mode,
                                            online_schedule=online_schedule if mode != 'static' else 'per_batch',
                                            budget_value=float(budget_value),
                                        )
                                        run_root = build_run_directory(
                                            root,
                                            dataset_name=dataset_name,
                                            feature_mode=feature_mode,
                                            sampler_name=sampler_name,
                                            model_family=model_config.family,
                                            seed=int(seed),
                                            batch_size=int(batch_size),
                                            num_hops=num_hops,
                                            hidden_dim=int(model_config.hidden_dim),
                                            num_layers=int(model_config.num_layers),
                                            dropout=float(model_config.dropout),
                                            aggr=model_config.aggr,
                                            jk_mode=model_config.jk_mode,
                                            head_hidden_dim=int(model_config.head_hidden_dim or 0),
                                            q=wilson_q,
                                            sampling_mode=mode,
                                            online_schedule=online_schedule if mode != 'static' else 'per_batch',
                                            budget_unit=config.sampling.budget_unit,
                                            budget_value=float(budget_value),
                                        )
                                        metadata = {
                                            'dataset_name': dataset_name,
                                            'model_family': model_config.family,
                                            'feature_mode': feature_mode,
                                            'sampler_name': sampler_name,
                                            'q': wilson_q,
                                            'batch_size': int(batch_size),
                                            'num_hops': num_hops,
                                            'hidden_dim': int(model_config.hidden_dim),
                                            'num_layers': int(model_config.num_layers),
                                            'dropout': float(model_config.dropout),
                                            'aggr': model_config.aggr,
                                            'jk_mode': model_config.jk_mode,
                                            'head_hidden_dim': int(model_config.head_hidden_dim or 0),
                                            'monitor_metric': config.optimization.monitor_metric,
                                            'sampling_mode': mode,
                                            'online_schedule': online_schedule if mode != 'static' else 'per_batch',
                                            'budget_unit': config.sampling.budget_unit,
                                            'budget_value': float(budget_value),
                                        }
                                        jobs.append({'train_config': train_config, 'run_root': str(run_root), 'metadata': metadata})

    validate_unique_run_roots(jobs)
    LOGGER.info('Prepared %d search jobs.', len(jobs))
    detailed_results = maybe_parallel_map(_run_one, jobs, parallel_workers=config.search.parallel_workers)

    flat_rows = [
        flatten_run_result(
            item['result'],
            q=item['job']['q'],
            batch_size=item['job']['batch_size'],
            num_hops=item['job']['num_hops'],
            model_family=item['job']['model_family'],
            hidden_dim=item['job']['hidden_dim'],
            num_layers=item['job']['num_layers'],
            dropout=item['job']['dropout'],
            aggr=item['job']['aggr'],
            jk_mode=item['job']['jk_mode'],
            head_hidden_dim=item['job']['head_hidden_dim'],
            monitor_metric=item['job']['monitor_metric'],
            sampling_mode=item['job']['sampling_mode'],
            online_schedule=item['job']['online_schedule'],
            budget_unit=item['job']['budget_unit'],
            budget_value=item['job']['budget_value'],
        )
        for item in detailed_results
    ]
    baseline_rows, baseline_detailed = _baseline_rows(config)
    flat_rows.extend(baseline_rows)
    detailed_results.extend({'result': item, 'job': {'is_baseline': True}} for item in baseline_detailed)

    summary_rows = aggregate_rows(
        flat_rows,
        group_keys=['dataset_name', 'dataset_kind', 'feature_mode', 'model_family', 'sampler_name', 'sampling_mode', 'online_schedule', 'budget_unit', 'budget_value', 'q', 'batch_size', 'num_hops', 'hidden_dim', 'num_layers', 'dropout', 'aggr', 'jk_mode', 'head_hidden_dim', 'monitor_metric'],
    )
    leaderboard_rows = sort_rows_by_metric(summary_rows, config.search.selection_metric)

    write_json(root / 'tables' / 'search_detailed_runs.json', {'detailed_results': detailed_results})
    write_csv(root / 'tables' / 'search_detailed_runs.csv', flat_rows)
    write_json(root / 'tables' / 'search_summary.json', {'summary_rows': summary_rows})
    write_csv(root / 'tables' / 'search_summary.csv', summary_rows)
    write_json(root / 'tables' / 'search_leaderboard.json', {'leaderboard': leaderboard_rows})
    write_csv(root / 'tables' / 'search_leaderboard.csv', leaderboard_rows)

    if summary_rows:
        for metric_key in ('mean_test_accuracy', 'mean_test_macro_f1', 'mean_test_balanced_accuracy', 'mean_test_weighted_f1'):
            _plot_runtime_tradeoff(summary_rows, root / 'plots' / f'{metric_key}_vs_runtime.png', metric_key=metric_key)
            plot_rows_separated_by_dataset_kind(summary_rows, root / 'plots', plotter=_plot_runtime_tradeoff, filename=f'{metric_key}_vs_runtime.png', metric_key=metric_key)

    LOGGER.info('Search study complete. Outputs written to %s', root)
    return root



def main() -> None:
    args = parse_config_arg('Run an end-to-end model/sampler search study.')
    config = load_config(SearchConfig, args.config)
    run_search_study(config)


if __name__ == '__main__':  # pragma: no cover
    main()
