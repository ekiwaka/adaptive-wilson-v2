from __future__ import annotations

import csv
import math
from collections import defaultdict
from copy import deepcopy
from statistics import fmean, stdev
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from ..analysis.wilson_checks import (
    cumulative_absolute_error,
    exact_wilson_root_kernel,
    estimate_wilson_root_marginals_by_forests,
    make_default_pairs,
    pairwise_root_inclusion,
    symmetry_invariance_error,
)
from ..config.loader import load_config
from ..config.schema import DatasetConfig, WilsonChecksConfig
from ..data.loading import load_dataset
from ..utils.io import ensure_dir, write_csv, write_json
from ..utils.logging import configure_logging, get_logger
from ..utils.random import seed_everything

LOGGER = get_logger(__name__)


def _plot_convergence(rows: list[dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    x = [max(float(row['num_forests']), 1.0) for row in rows]
    y = [max(float(row['absolute_error']), 1e-12) for row in rows]
    reference = [1.0 / (value**0.5) for value in x]
    if y:
        scale = y[0] / max(reference[0], 1e-12)
        reference = [scale * value for value in reference]
    plt.figure(figsize=(7, 5))
    plt.loglog(x, y, label='Monte Carlo absolute error')
    plt.loglog(x, reference, linestyle='--', label='1/sqrt(m) reference')
    plt.xlabel('Number of sampled forests')
    plt.ylabel('Absolute error in expected root count')
    plt.title('Wilson root-count Monte Carlo convergence')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def _run_one_dataset(config: WilsonChecksConfig, dataset_name: str, feature_mode: str) -> dict[str, Any]:
    q = float((config.wilson_checks.q if config.wilson_checks.q is not None else (config.sampling.wilson.q or 1.0)))
    dataset_config = DatasetConfig(
        name=dataset_name,
        root=config.data.root,
        split_idx=config.data.split_idx,
        planetoid_split=config.data.planetoid_split,
        feature_mode=feature_mode,
        standardize_features=config.data.standardize_features,
        split=config.data.split,
        synthetic=config.data.synthetic,
    )
    bundle = load_dataset(dataset_config, device=torch.device(config.runtime.device if config.runtime.device in {'cpu', 'cuda'} else 'cpu'), seed=config.runtime.seed)
    edge_index = bundle.data.edge_index.cpu()
    num_nodes = int(bundle.data.num_nodes)
    if num_nodes > int(config.wilson_checks.max_exact_nodes):
        raise ValueError(
            f'Wilson exact checks for {dataset_name} require {num_nodes} nodes, exceeding '
            f'checks.max_exact_nodes={config.wilson_checks.max_exact_nodes}. Increase the limit explicitly.'
        )

    out_dir = ensure_dir(Path(config.output.root) / config.output.experiment_name / dataset_name.replace('/', '_') / feature_mode)
    LOGGER.info('Running Wilson checks for dataset=%s feature_mode=%s q=%s nodes=%d', dataset_name, feature_mode, q, num_nodes)

    exact = exact_wilson_root_kernel(edge_index, num_nodes, q)
    mc = estimate_wilson_root_marginals_by_forests(
        edge_index,
        num_nodes,
        q,
        num_forests=config.wilson_checks.num_forests,
        seed=config.runtime.seed,
        backend=config.sampling.wilson.backend,
    )
    root_counts = [float(v) for v in mc['root_counts']]  # type: ignore[index]
    mc_marginals = mc['root_marginals']  # type: ignore[assignment]
    assert isinstance(mc_marginals, torch.Tensor)
    convergence = cumulative_absolute_error(root_counts, exact.expected_root_count)
    marginal_rows = [
        {
            'node': int(idx),
            'exact_root_marginal': float(exact.root_marginals[idx].item()),
            'mc_root_marginal': float(mc_marginals[idx].item()),
            'absolute_error': float(abs(exact.root_marginals[idx].item() - mc_marginals[idx].item())),
        }
        for idx in range(num_nodes)
    ]
    pair_rows = pairwise_root_inclusion(exact.kernel, make_default_pairs(num_nodes, max_pairs=config.wilson_checks.max_pairs))
    symmetry_error = symmetry_invariance_error(edge_index, num_nodes, q, seed=config.runtime.seed + 17)
    max_marginal_error = max((row['absolute_error'] for row in marginal_rows), default=0.0)
    min_negative_gap = min((row['negative_correlation_gap'] for row in pair_rows), default=0.0)

    write_csv(out_dir / 'root_count_convergence.csv', convergence)
    write_csv(out_dir / 'root_marginals.csv', marginal_rows)
    write_csv(out_dir / 'pairwise_inclusion.csv', pair_rows)
    if config.wilson_checks.save_plots:
        _plot_convergence(convergence, out_dir / 'root_count_convergence_loglog.png')
    metrics = {
        'dataset': dataset_name,
        'feature_mode': feature_mode,
        'q': q,
        'num_nodes': num_nodes,
        'num_edges': int(edge_index.size(1)),
        'exact_expected_root_count': exact.expected_root_count,
        'mc_mean_root_count': float(sum(root_counts) / max(len(root_counts), 1)),
        'mc_num_forests': int(config.wilson_checks.num_forests),
        'max_root_marginal_abs_error': float(max_marginal_error),
        'min_pairwise_negative_correlation_gap': float(min_negative_gap),
        'negative_correlation_violations': int(sum(1 for row in pair_rows if row['negative_correlation_gap'] < -1e-8)),
        'symmetry_invariance_max_error': symmetry_error,
    }
    write_json(out_dir / 'metrics.json', metrics)
    return metrics


def _aggregate_wilson_seed_outputs(config, seeds: list[int], original_experiment_name: str) -> None:
    root = Path(config.output.root)
    aggregate_root = root / original_experiment_name / 'aggregate'
    aggregate_root.mkdir(parents=True, exist_ok=True)

    grouped_files: dict[Path, list[tuple[int, Path]]] = defaultdict(list)

    for seed in seeds:
        seed_dir = root / f'{original_experiment_name}_seed{seed}'
        if not seed_dir.exists():
            raise RuntimeError(
                f'Cannot aggregate Wilson checks because seed output directory '
                f'does not exist: {seed_dir}'
            )

        for csv_path in seed_dir.rglob('root_count_convergence.csv'):
            relative_parent = csv_path.parent.relative_to(seed_dir)
            grouped_files[relative_parent].append((seed, csv_path))

    if not grouped_files:
        raise RuntimeError(
            'No root_count_convergence.csv files were found for seed aggregation.'
        )

    for relative_parent, seed_files in grouped_files.items():
        output_dir = aggregate_root / relative_parent
        output_dir.mkdir(parents=True, exist_ok=True)

        _aggregate_one_root_count_convergence_group(
            seed_files=seed_files,
            output_dir=output_dir,
            save_plots=bool(config.wilson_checks.save_plots),
        )


def _aggregate_one_root_count_convergence_group(
    *,
    seed_files: list[tuple[int, Path]],
    output_dir: Path,
    save_plots: bool,
) -> None:
    combined_rows: list[dict[str, object]] = []
    grouped_errors: dict[float, list[float]] = defaultdict(list)

    x_name_for_output = 'sample_count'

    for seed, csv_path in seed_files:
        with csv_path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []

            x_col = _detect_forest_count_column(fieldnames)
            error_col = _detect_absolute_error_column(fieldnames)

            for row in reader:
                sample_count = float(row[x_col])
                absolute_error = float(row[error_col])

                grouped_errors[sample_count].append(absolute_error)

                copied = dict(row)
                copied['seed'] = seed
                copied['source_file'] = str(csv_path)
                copied['sample_count_for_aggregation'] = sample_count
                copied['absolute_error_for_aggregation'] = absolute_error
                combined_rows.append(copied)

    combined_path = output_dir / 'root_count_convergence_all_seeds.csv'
    _write_dict_rows(combined_path, combined_rows)

    summary_rows = []
    for sample_count in sorted(grouped_errors):
        values = grouped_errors[sample_count]
        mean_error = fmean(values)
        std_error = stdev(values) if len(values) > 1 else 0.0
        stderr_error = std_error / math.sqrt(len(values)) if values else 0.0

        summary_rows.append(
            {
                x_name_for_output: sample_count,
                'mean_absolute_error': mean_error,
                'std_absolute_error': std_error,
                'stderr_absolute_error': stderr_error,
                'num_seeds': len(values),
            }
        )

    summary_path = output_dir / 'root_count_convergence_mean_std.csv'
    _write_dict_rows(summary_path, summary_rows)

    if save_plots:
        _plot_aggregate_root_count_convergence(
            summary_rows=summary_rows,
            output_path=output_dir / 'root_count_convergence_mean_std_loglog.png',
        )


def _detect_forest_count_column(fieldnames: list[str]) -> str:
    preferred = [
        'sample_count',
        'num_forests',
        'forest_count',
        'm',
        'num_samples',
    ]

    for name in preferred:
        if name in fieldnames:
            return name

    for name in fieldnames:
        lowered = name.lower()
        if 'forest' in lowered and ('count' in lowered or 'num' in lowered):
            return name

    for name in fieldnames:
        lowered = name.lower()
        if 'sample' in lowered and ('count' in lowered or 'num' in lowered):
            return name

    raise RuntimeError(
        'Could not detect the forest-count column in root_count_convergence.csv. '
        f'Available columns: {fieldnames}'
    )


def _detect_absolute_error_column(fieldnames: list[str]) -> str:
    preferred = [
        'absolute_error',
        'abs_error',
        'absolute_root_count_error',
        'root_count_absolute_error',
    ]

    for name in preferred:
        if name in fieldnames:
            return name

    for name in fieldnames:
        lowered = name.lower()
        if 'absolute' in lowered and 'error' in lowered:
            return name

    for name in fieldnames:
        lowered = name.lower()
        if 'abs' in lowered and 'error' in lowered:
            return name

    raise RuntimeError(
        'Could not detect the absolute-error column in root_count_convergence.csv. '
        f'Available columns: {fieldnames}'
    )


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text('', encoding='utf-8')
        return

    fieldnames: list[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_aggregate_root_count_convergence(
    *,
    summary_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    if not summary_rows:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.asarray([float(row['sample_count']) for row in summary_rows], dtype=float)
    mean = np.asarray([float(row['mean_absolute_error']) for row in summary_rows], dtype=float)
    std = np.asarray([float(row['std_absolute_error']) for row in summary_rows], dtype=float)

    positive = mean > 0
    if not positive.any():
        return

    eps = 1e-12
    lower = np.maximum(mean - std, eps)
    upper = np.maximum(mean + std, eps)

    reference = mean[0] * np.sqrt(x[0] / x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x, mean, label='Mean Monte Carlo absolute error across seeds')
    ax.fill_between(x, lower, upper, alpha=0.2, label='±1 std across seeds')
    ax.loglog(x, reference, linestyle='--', label='1/sqrt(m) reference')

    ax.set_xlabel('Number of sampled forests')
    ax.set_ylabel('Absolute error in expected root count')
    ax.set_title('Wilson root-count Monte Carlo convergence across seeds')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)





def run_wilson_checks(config):
    checks = config.wilson_checks
    seeds = checks.seeds if checks.seeds is not None else [config.runtime.seed]
    seeds = [int(seed) for seed in seeds]

    original_experiment_name = config.output.experiment_name
    all_rows = []

    for seed in seeds:
        seed_config = deepcopy(config)
        seed_config.runtime.seed = seed

        if len(seeds) > 1:
            seed_config.study_name = f'{config.study_name}_seed{seed}'
            seed_config.output.experiment_name = f'{original_experiment_name}_seed{seed}'

        seed_everything(seed_config.runtime.seed)

        rows = []
        for dataset_name in seed_config.data.datasets:
            for feature_mode in seed_config.data.feature_modes:
                row = _run_one_dataset(seed_config, dataset_name, feature_mode)
                row['seed'] = seed
                rows.append(row)

        output_dir = Path(seed_config.output.root) / seed_config.output.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(output_dir / 'summary.csv', rows)
        write_json(
            output_dir / 'summary.json',
            {
                'study_name': seed_config.study_name,
                'seed': seed,
                'rows': rows,
            },
        )

        all_rows.extend(rows)

    if len(seeds) > 1:
        multi_seed_dir = Path(config.output.root) / original_experiment_name
        multi_seed_dir.mkdir(parents=True, exist_ok=True)

        write_csv(multi_seed_dir / 'summary_all_seeds.csv', all_rows)
        write_json(
            multi_seed_dir / 'summary_all_seeds.json',
            {
                'study_name': config.study_name,
                'seeds': seeds,
                'aggregate_seeds': bool(checks.aggregate_seeds),
                'rows': all_rows,
            },
        )

        if checks.aggregate_seeds:
            _aggregate_wilson_seed_outputs(
                config=config,
                seeds=seeds,
                original_experiment_name=original_experiment_name,
            )

    return all_rows
    


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Run Wilson root-kernel and Monte Carlo checks.')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(WilsonChecksConfig, args.config)
    run_wilson_checks(config)


if __name__ == '__main__':
    main()
