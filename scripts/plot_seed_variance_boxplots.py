from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot seed-level boxplots by budget from benchmark_detailed_runs.csv')
    parser.add_argument('--detailed', type=Path, required=True, help='Path to benchmark_detailed_runs.csv')
    parser.add_argument('--outdir', type=Path, required=True, help='Output directory')
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.detailed)

    metrics = [
        ('train_accuracy', 'Train accuracy'),
        ('test_accuracy', 'Test accuracy'),
        ('train_macro_f1', 'Train macro-F1'),
        ('test_macro_f1', 'Test macro-F1'),
        ('train_balanced_accuracy', 'Train balanced accuracy'),
        ('test_balanced_accuracy', 'Test balanced accuracy'),
    ]

    for (dataset_name, split_mode, sampling_mode), group in df.groupby(
        ['dataset_name', 'split_mode', 'sampling_mode']
    ):
        for metric_key, title in metrics:
            plt.figure(figsize=(10, 5))

            samplers = sorted(group['sampler_name'].dropna().unique())
            budgets = sorted(group['resolved_budget'].dropna().unique())

            positions = []
            box_data = []
            labels = []
            pos = 1

            for budget in budgets:
                budget_group = group[group['resolved_budget'] == budget]
                for sampler in samplers:
                    vals = budget_group.loc[
                        budget_group['sampler_name'] == sampler, metric_key
                    ].dropna().tolist()
                    if vals:
                        box_data.append(vals)
                        positions.append(pos)
                        labels.append(f'{sampler}\n{int(budget)}')
                        pos += 1
                pos += 1

            if not box_data:
                plt.close()
                continue

            plt.boxplot(box_data, positions=positions, widths=0.7)
            plt.xticks(positions, labels, rotation=45, ha='right')
            plt.ylabel(title)
            plt.xlabel('Sampler and training-node budget')
            plt.title(f'{dataset_name} | {split_mode} | {sampling_mode} | {title}')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()

            safe_dataset = dataset_name.replace('/', '_')
            out_path = args.outdir / f'{safe_dataset}_{split_mode}_{sampling_mode}_{metric_key}_boxplot.png'
            plt.savefig(out_path, dpi=180, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main()