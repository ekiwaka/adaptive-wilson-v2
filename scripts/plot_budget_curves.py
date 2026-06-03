from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

METRICS = [
    ('train_accuracy', 'mean_train_accuracy', 'std_train_accuracy', 'Train accuracy'),
    ('val_accuracy', 'mean_val_accuracy', 'std_val_accuracy', 'Validation accuracy'),
    ('test_accuracy', 'mean_test_accuracy', 'std_test_accuracy', 'Test accuracy'),

    ('train_macro_f1', 'mean_train_macro_f1', 'std_train_macro_f1', 'Train macro-F1'),
    ('val_macro_f1', 'mean_val_macro_f1', 'std_val_macro_f1', 'Validation macro-F1'),
    ('test_macro_f1', 'mean_test_macro_f1', 'std_test_macro_f1', 'Test macro-F1'),

    ('train_balanced_accuracy', 'mean_train_balanced_accuracy', 'std_train_balanced_accuracy', 'Train balanced accuracy'),
    ('val_balanced_accuracy', 'mean_val_balanced_accuracy', 'std_val_balanced_accuracy', 'Validation balanced accuracy'),
    ('test_balanced_accuracy', 'mean_test_balanced_accuracy', 'std_test_balanced_accuracy', 'Test balanced accuracy'),
]


def _mode_label(row: dict[str, str]) -> str:
    if row.get('sampling_mode') == 'static':
        return 'static'
    return f"online_{row.get('online_schedule', 'per_batch')}"


def _to_float(value: str | None) -> float | None:
    if value in (None, '', 'None'):
        return None
    return float(value)


def _budget_x(row: dict[str, str]) -> float:
    resolved = row.get('mean_resolved_budget') or row.get('resolved_budget') or row.get('budget_value') or row.get('batch_size') or '0'
    return float(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot metric-vs-budget curves from benchmark_summary.csv')
    parser.add_argument('--summary', type=Path, required=True, help='Path to benchmark_summary.csv')
    parser.add_argument('--outdir', type=Path, required=True, help='Output directory for plots/tables')
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    with args.summary.open('r', newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get('sampler_name', '')).startswith('baseline_'):
            continue
        grouped[(row['dataset_name'], row.get('split_mode', 'predefined'), _mode_label(row))].append(row)

    for (dataset_name, split_mode, mode_label), group in grouped.items():
        safe_dataset = dataset_name.replace('/', '_')
        group_dir = args.outdir / safe_dataset / split_mode / mode_label
        group_dir.mkdir(parents=True, exist_ok=True)

        table_cols = [
            'dataset_name',
            'split_mode',
            'training_objective',
            'sampling_mode',
            'online_schedule',
            'sampler_name',
            'budget_value',
            'mean_resolved_budget',
            'mean_train_node_count',
            'mean_val_node_count',
            'mean_test_node_count',
            'mean_val_accuracy',
            'mean_test_accuracy',
            'mean_val_macro_f1',
            'mean_test_macro_f1',
            'mean_val_balanced_accuracy',
            'mean_test_balanced_accuracy',
        ]
        with (group_dir / 'budget_curve_table.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=table_cols)
            writer.writeheader()
            for row in sorted(group, key=_budget_x):
                writer.writerow({key: row.get(key) for key in table_cols})

        for metric_key, mean_key, std_key, title in METRICS:
            plt.figure(figsize=(7.5, 4.8))
            for sampler_name in sorted({row['sampler_name'] for row in group}):
                subset = [row for row in group if row['sampler_name'] == sampler_name]
                subset.sort(key=_budget_x)

                xs = np.array([_budget_x(row) for row in subset], dtype=float)
                ys = np.array([_to_float(row.get(mean_key)) for row in subset], dtype=float)
                stds = np.array([
                    0.0 if _to_float(row.get(std_key)) is None else _to_float(row.get(std_key))
                    for row in subset
                ], dtype=float)

                plt.plot(xs, ys, marker='o', label=sampler_name)
                plt.fill_between(xs, ys - stds, ys + stds, alpha=0.18)

            plt.xlabel('Training-node budget')
            plt.ylabel(title)
            plt.title(f'{dataset_name} | {split_mode} | {mode_label} | {title} vs budget')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(group_dir / f'{metric_key}_vs_budget.png', dpi=180, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main()
