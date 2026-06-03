from __future__ import annotations
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
import matplotlib.pyplot as plt

from ..config.loader import load_config
from ..config.schema import DatasetConfig, QSweepConfig, TrainConfig
from ..utils.io import write_csv, write_json
from ..utils.logging import get_logger
from .baselines import compute_trivial_baselines
from .common import (
    aggregate_rows,
    build_run_directory,
    configure_study_logging,
    flatten_run_result,
    maybe_parallel_map,
    parse_config_arg,
    prepare_experiment_root,
    save_resolved_config,
    sort_rows_by_metric,
    validate_unique_run_roots,
    warn_if_test_metrics_are_tracked,
)

LOGGER = get_logger(__name__)



def build_q_values(q_values: Iterable[float], *, start: float, stop: float, step: float) -> list[float]:
    explicit = sorted({float(value) for value in q_values})
    if explicit:
        return explicit
    start_dec = Decimal(str(start))
    stop_dec = Decimal(str(stop))
    step_dec = Decimal(str(step))
    if step_dec <= 0:
        raise ValueError('q_step must be positive.')
    values: list[float] = []
    current = start_dec
    while current <= stop_dec:
        values.append(float(current))
        current += step_dec
    if values[-1] != float(stop_dec):
        values.append(float(stop_dec))
    return values



def _make_train_config(config: QSweepConfig, *, q_value: float, seed: int) -> TrainConfig:
    return TrainConfig(
        run_name=(
            f'{config.dataset.name}_{config.dataset.feature_mode}_wilson_seed{seed}'
            f'_{config.model.family}_mode{config.sampling.mode}_sched{config.sampling.online_schedule}'
            f'_budget{config.sampling.budget_value if config.sampling.budget_value is not None else config.sampling.batch_size}'
            f'_{config.sampling.budget_unit}_hops{config.sampling.num_hops}'
            f'_hd{config.model.hidden_dim}_L{config.model.num_layers}'
            f'_drop{config.model.dropout}_aggr{config.model.aggr}_jk{config.model.jk_mode}'
            f'_head{int(config.model.head_hidden_dim or 0)}_q{q_value:g}'
        ),
        runtime=replace(config.runtime, seed=seed),
        output=config.output,
        logging=config.logging,
        dataset=config.dataset,
        model=config.model,
        optimization=config.optimization,
        sampling=replace(config.sampling, wilson=replace(config.sampling.wilson, q=q_value)),
        evaluation=config.evaluation,
    )



def _run_one(job: dict[str, Any]) -> dict[str, Any]:
    from .train import run_training_experiment

    train_config = job['train_config']
    run_root = Path(job['run_root'])
    result, _ = run_training_experiment(train_config, run_root=run_root, configure_root_logging=True)
    return {'result': result.to_dict(), 'job': job['metadata']}



def _plot_scores(summary_rows: list[dict[str, Any]], out_path: Path, *, selection_metric: str) -> None:
    q_rows = [row for row in summary_rows if row.get('q') is not None]
    if not q_rows:
        return
    plt.figure(figsize=(10, 6))
    q_values = [float(row['q']) for row in q_rows]
    for mean_key, label in [
        ('mean_val_accuracy', 'val accuracy'),
        ('mean_test_accuracy', 'test accuracy'),
        ('mean_val_macro_f1', 'val macro_f1'),
        ('mean_test_macro_f1', 'test macro_f1'),
        ('mean_val_balanced_accuracy', 'val balanced_accuracy'),
        ('mean_test_balanced_accuracy', 'test balanced_accuracy'),
        ('mean_val_weighted_f1', 'val weighted_f1'),
        ('mean_test_weighted_f1', 'test weighted_f1'),
    ]:
        if mean_key in q_rows[0]:
            plt.plot(q_values, [float(row[mean_key]) for row in q_rows], marker='o', label=label)
    best_row = sort_rows_by_metric(q_rows, selection_metric)[0]
    plt.axvline(float(best_row['q']), linestyle='--', label=f"best q={best_row['q']}")
    plt.xlabel('Wilson q')
    plt.ylabel('score')
    plt.title('Wilson q sweep scores')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()



def _baseline_rows(config: QSweepConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_config = DatasetConfig(
        name=config.dataset.name,
        root=config.dataset.root,
        split_idx=config.dataset.split_idx,
        feature_mode=config.dataset.feature_mode,
        standardize_features=config.dataset.standardize_features,
        synthetic=config.dataset.synthetic,
    )
    rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    for run_index in range(config.q_sweep.runs):
        seed = config.q_sweep.base_seed + run_index
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



def run_q_sweep_study(config: QSweepConfig) -> Path:
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
        context='q_sweep study',
    )

    q_values = build_q_values(config.q_sweep.q_values, start=config.q_sweep.q_start, stop=config.q_sweep.q_stop, step=config.q_sweep.q_step)

    jobs: list[dict[str, Any]] = []
    for q_value in q_values:
        for run_index in range(config.q_sweep.runs):
            seed = config.q_sweep.base_seed + run_index
            train_config = _make_train_config(config, q_value=q_value, seed=seed)
            budget_value = float(config.sampling.budget_value) if config.sampling.budget_value is not None else float(config.sampling.batch_size)
            run_root = build_run_directory(
                root,
                dataset_name=config.dataset.name,
                feature_mode=config.dataset.feature_mode,
                sampler_name='wilson',
                model_family=config.model.family,
                seed=int(seed),
                batch_size=int(config.sampling.batch_size),
                num_hops=int(config.sampling.num_hops),
                hidden_dim=int(config.model.hidden_dim),
                num_layers=int(config.model.num_layers),
                dropout=float(config.model.dropout),
                aggr=config.model.aggr,
                jk_mode=config.model.jk_mode,
                head_hidden_dim=int(config.model.head_hidden_dim or 0),
                q=float(q_value),
                sampling_mode=config.sampling.mode,
                online_schedule=config.sampling.online_schedule if config.sampling.mode != 'static' else 'none',
                budget_unit=config.sampling.budget_unit,
                budget_value=budget_value,
            )
            metadata = {
                'model_family': config.model.family,
                'q': float(q_value),
                'batch_size': int(config.sampling.batch_size),
                'num_hops': int(config.sampling.num_hops),
                'hidden_dim': int(config.model.hidden_dim),
                'num_layers': int(config.model.num_layers),
                'dropout': float(config.model.dropout),
                'aggr': config.model.aggr,
                'jk_mode': config.model.jk_mode,
                'head_hidden_dim': int(config.model.head_hidden_dim or 0),
                'monitor_metric': config.optimization.monitor_metric,
                'sampling_mode': config.sampling.mode,
                'online_schedule': config.sampling.online_schedule if config.sampling.mode != 'static' else 'none',
                'budget_unit': config.sampling.budget_unit,
                'budget_value': budget_value,
            }
            jobs.append({'train_config': train_config, 'run_root': str(run_root), 'metadata': metadata})

    validate_unique_run_roots(jobs)
    detailed_results = maybe_parallel_map(_run_one, jobs, parallel_workers=config.q_sweep.parallel_workers)
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
    leaderboard_rows = sort_rows_by_metric(summary_rows, config.q_sweep.selection_metric)

    write_json(root / 'tables' / 'q_sweep_detailed_runs.json', {'detailed_results': detailed_results})
    write_csv(root / 'tables' / 'q_sweep_detailed_runs.csv', flat_rows)
    write_json(root / 'tables' / 'q_sweep_summary.json', {'summary_rows': summary_rows})
    write_csv(root / 'tables' / 'q_sweep_summary.csv', summary_rows)
    write_json(root / 'tables' / 'q_sweep_leaderboard.json', {'leaderboard': leaderboard_rows})
    write_csv(root / 'tables' / 'q_sweep_leaderboard.csv', leaderboard_rows)

    if summary_rows:
        _plot_scores(summary_rows, root / 'plots' / 'q_sweep_scores.png', selection_metric=config.q_sweep.selection_metric)

    LOGGER.info('Wilson q sweep complete. Outputs written to %s', root)
    return root



def main() -> None:
    args = parse_config_arg('Run a Wilson q sweep.')
    config = load_config(QSweepConfig, args.config)
    run_q_sweep_study(config)


if __name__ == '__main__':  # pragma: no cover
    main()
