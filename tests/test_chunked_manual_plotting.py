from __future__ import annotations

from pathlib import Path

from gnn_graph_sampling.experiments.benchmark import write_benchmark_plots_from_rows


def _row(seed: int, sampler: str, *, budget: float = 0.1) -> dict[str, object]:
    return {
        'weighting_regime': 'unweighted',
        'dataset_name': 'roman-empire',
        'dataset_kind': 'real',
        'feature_mode': 'enabled',
        'split_mode': 'generated',
        'training_objective': 'sampled_subgraph',
        'model_family': 'mixhop',
        'sampler_name': sampler,
        'adaptive_alpha': 0.1 if sampler == 'adaptive_wilson' else None,
        'sampling_mode': 'online',
        'online_schedule': 'per_batch',
        'budget_unit': 'ratio',
        'budget_value': budget,
        'batch_size': 64,
        'num_hops': 2,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.5,
        'aggr': 'mean',
        'jk_mode': 'last',
        'head_hidden_dim': 0,
        'q': 1.0,
        'seed': seed,
        'run_index': seed,
        'total_training_time': 1.0 + seed,
        'train_accuracy': 0.6,
        'val_accuracy': 0.55,
        'test_accuracy': 0.5,
        'train_macro_f1': 0.6,
        'val_macro_f1': 0.55,
        'test_macro_f1': 0.5,
        'train_balanced_accuracy': 0.6,
        'val_balanced_accuracy': 0.55,
        'test_balanced_accuracy': 0.5,
        'test_weighted_f1': 0.5,
        'average_center_nodes': 10.0,
        'mean_average_center_nodes': 10.0,
        'resolved_budget': 10.0,
        'mean_resolved_budget': 10.0,
    }


def test_write_benchmark_plots_from_rows_creates_plots_directory(tmp_path: Path) -> None:
    flat_rows = [_row(0, 'uniform'), _row(1, 'degree'), _row(2, 'adaptive_wilson')]
    summary_rows = []
    for row in flat_rows:
        summary = dict(row)
        for metric in (
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
            'total_training_time',
            'average_center_nodes',
            'resolved_budget',
        ):
            summary[f'mean_{metric}'] = row[metric]
            summary[f'std_{metric}'] = 0.0
        summary_rows.append(summary)

    root = tmp_path / 'combined' / 'unweighted' / 'roman_empire'
    write_benchmark_plots_from_rows(flat_rows=flat_rows, summary_rows=summary_rows, root=root)

    assert (root / 'plots' / 'train_accuracy_vs_runtime.png').exists()
    assert any((root / 'plots').rglob('test_accuracy_vs_budget.png'))
