from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
import matplotlib.pyplot as plt

from ..utils.io import ensure_dir, write_csv, write_json
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

SAMPLER_METRICS = [
    'mean_test_accuracy',
    'mean_test_macro_f1',
    'mean_test_balanced_accuracy',
    'mean_test_weighted_f1',
    'mean_average_sampled_nodes',
    'mean_average_sampled_edges',
    'mean_budget_match_ratio',
    'mean_mean_center_degree',
    'mean_mean_center_degree_ratio_vs_global',
    'mean_mean_center_class_coverage_ratio',
    'mean_mean_center_label_entropy_ratio',
    'mean_mean_redraws_used',
    'mean_mean_raw_mean_roots_before_cap',
    'mean_mean_fallback_uniform',
]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', newline='') as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: object, default: float = 0.0) -> float:
    if value in (None, '', 'None'):
        return default
    return float(value)


def _aggregate_by_dataset_sampler(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset_name = str(row.get('dataset_name', 'unknown'))
        sampler_name = str(row.get('sampler_name', 'unknown'))
        sampling_mode = str(row.get('sampling_mode', 'unknown'))
        online_schedule = str(row.get('online_schedule', 'unknown'))
        grouped[(dataset_name, sampler_name, sampling_mode, online_schedule)].append(row)

    summary: list[dict[str, object]] = []
    for (dataset_name, sampler_name, sampling_mode, online_schedule), records in sorted(grouped.items()):
        out: dict[str, object] = {
            'dataset_name': dataset_name,
            'sampler_name': sampler_name,
            'sampling_mode': sampling_mode,
            'online_schedule': online_schedule,
            'num_records': len(records),
        }
        for metric in SAMPLER_METRICS:
            numeric = [_safe_float(record.get(metric), default=float('nan')) for record in records if record.get(metric) not in (None, '', 'None')]
            if numeric:
                out[metric] = float(sum(numeric) / len(numeric))
        summary.append(out)
    return summary


def _plot_metric(rows: Sequence[dict[str, object]], *, out_path: Path, dataset_name: str, metric_key: str, title: str) -> None:
    subset = [row for row in rows if str(row.get('dataset_name')) == dataset_name and metric_key in row]
    if not subset:
        return
    subset = sorted(subset, key=lambda row: str(row.get('sampler_name')))
    x = [str(row['sampler_name']) for row in subset]
    y = [float(row[metric_key]) for row in subset]
    plt.figure(figsize=(7, 4.5))
    plt.bar(x, y)
    plt.title(title)
    plt.ylabel(metric_key.replace('mean_', '').replace('_', ' '))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def _collect_best_q(q_sweep_roots: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in q_sweep_roots:
        leaderboard = root / 'tables' / 'q_sweep_leaderboard.csv'
        if not leaderboard.exists():
            continue
        for row in _read_csv_rows(leaderboard):
            if str(row.get('sampler_name')) != 'wilson':
                continue
            rows.append(
                {
                    'dataset_name': str(row.get('dataset_name')),
                    'feature_mode': str(row.get('feature_mode')),
                    'best_q': _safe_float(row.get('q')),
                    'selection_metric': str(row.get('monitor_metric') or ''),
                    'score': _safe_float(row.get('mean_val_macro_f1') or row.get('mean_val_balanced_accuracy') or row.get('mean_best_monitor_value')),
                    'source_root': str(root),
                }
            )
            break
    return rows


def generate_sampler_report(
    *,
    benchmark_roots: Sequence[Path],
    q_sweep_roots: Sequence[Path],
    out_root: Path,
) -> Path:
    """Create compact tables/plots for sampler comparison and Wilson q choices."""

    ensure_dir(out_root)
    ensure_dir(out_root / 'plots')
    ensure_dir(out_root / 'tables')

    benchmark_rows: list[dict[str, str]] = []
    for root in benchmark_roots:
        summary_path = root / 'tables' / 'benchmark_summary.csv'
        if not summary_path.exists():
            LOGGER.warning('Skipping missing benchmark summary: %s', summary_path)
            continue
        benchmark_rows.extend(_read_csv_rows(summary_path))

    sampler_summary = _aggregate_by_dataset_sampler(benchmark_rows)
    best_q_rows = _collect_best_q(q_sweep_roots)

    write_csv(out_root / 'tables' / 'sampler_summary.csv', sampler_summary)
    write_json(out_root / 'tables' / 'sampler_summary.json', {'rows': sampler_summary})
    write_csv(out_root / 'tables' / 'best_wilson_q_by_dataset.csv', best_q_rows)
    write_json(out_root / 'tables' / 'best_wilson_q_by_dataset.json', {'rows': best_q_rows})

    datasets = sorted({str(row['dataset_name']) for row in sampler_summary})
    for dataset_name in datasets:
        _plot_metric(
            sampler_summary,
            out_path=out_root / 'plots' / f'{dataset_name}_macro_f1_by_sampler.png',
            dataset_name=dataset_name,
            metric_key='mean_test_macro_f1',
            title=f'{dataset_name}: test macro-F1 by sampler',
        )
        _plot_metric(
            sampler_summary,
            out_path=out_root / 'plots' / f'{dataset_name}_accuracy_by_sampler.png',
            dataset_name=dataset_name,
            metric_key='mean_test_accuracy',
            title=f'{dataset_name}: test accuracy by sampler',
        )
        _plot_metric(
            sampler_summary,
            out_path=out_root / 'plots' / f'{dataset_name}_sampled_nodes_by_sampler.png',
            dataset_name=dataset_name,
            metric_key='mean_average_sampled_nodes',
            title=f'{dataset_name}: average sampled nodes by sampler',
        )
        _plot_metric(
            sampler_summary,
            out_path=out_root / 'plots' / f'{dataset_name}_center_spread_by_sampler.png',
            dataset_name=dataset_name,
            metric_key='mean_mean_center_class_coverage_ratio',
            title=f'{dataset_name}: center class coverage ratio by sampler',
        )
        _plot_metric(
            sampler_summary,
            out_path=out_root / 'plots' / f'{dataset_name}_center_degree_ratio_by_sampler.png',
            dataset_name=dataset_name,
            metric_key='mean_mean_center_degree_ratio_vs_global',
            title=f'{dataset_name}: center degree ratio vs global mean by sampler',
        )

    LOGGER.info('Sampler report written to %s', out_root)
    return out_root
