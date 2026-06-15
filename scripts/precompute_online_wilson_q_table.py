from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import BenchmarkConfig, DatasetConfig
from gnn_graph_sampling.data.loading import load_dataset
from gnn_graph_sampling.sampling.wilson_sampler import WilsonCenterSampler
from gnn_graph_sampling.utils.random import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Precompute q values for online expected-budget Wilson mini-batches. '
            'For every requested budget B, the script searches q such that the expected '
            'number of eligible Wilson roots among the training nodes is approximately B.'
        )
    )
    parser.add_argument('--config', type=Path, required=True, help='Benchmark YAML to calibrate.')
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Output JSON path. Defaults to sampling.wilson.q_table_path from the config.',
    )
    parser.add_argument('--trials', type=int, default=None, help='Search trials per q candidate.')
    parser.add_argument('--validation-trials', type=int, default=256, help='Final validation forests per budget.')
    parser.add_argument('--seed', type=int, default=None, help='Calibration seed. Defaults to runtime.seed.')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing output JSON.')
    return parser.parse_args()


def _budget_counts(config: BenchmarkConfig, train_count: int) -> list[int]:
    values = list(config.sampling.budget_values)
    if not values:
        values = [float(value) for value in config.sampling.batch_sizes]
    counts: list[int] = []
    for value in values:
        if config.sampling.budget_unit == 'ratio':
            count = int(round(float(value) * float(train_count)))
        else:
            count = int(round(float(value)))
        count = max(1, min(count, int(train_count)))
        if count not in counts:
            counts.append(count)
    return counts


def _validate_q(
    sampler: WilsonCenterSampler,
    candidate_nodes: torch.Tensor,
    q: float,
    *,
    seed: int,
    trials: int,
) -> dict[str, float]:
    counts: list[float] = []
    generator = torch.Generator(device='cpu')
    for trial_idx in range(max(1, int(trials))):
        generator.manual_seed(int(seed) + int(trial_idx))
        roots = sampler._draw_eligible_roots(candidate_nodes, float(q), generator=generator)
        counts.append(float(roots.numel()))
    return {
        'validation_trials': float(max(1, int(trials))),
        'mean_roots': float(mean(counts)) if counts else 0.0,
        'std_roots': float(pstdev(counts)) if len(counts) > 1 else 0.0,
        'min_roots': float(min(counts)) if counts else 0.0,
        'max_roots': float(max(counts)) if counts else 0.0,
    }


def _dataset_config(config: BenchmarkConfig, *, dataset_name: str, feature_mode: str, split_idx: int) -> DatasetConfig:
    return DatasetConfig(
        name=dataset_name,
        root=config.data.root,
        split_idx=int(split_idx),
        planetoid_split=config.data.planetoid_split,
        feature_mode=feature_mode,
        standardize_features=config.data.standardize_features,
        split=config.data.split,
        synthetic=config.data.synthetic,
    )


def main() -> None:
    args = parse_args()
    config = load_config(BenchmarkConfig, args.config)
    out_path = args.out or (Path(config.sampling.wilson.q_table_path) if config.sampling.wilson.q_table_path else None)
    if out_path is None:
        raise SystemExit('No output path supplied. Use --out or set sampling.wilson.q_table_path in the config.')
    if out_path.exists() and not args.force:
        raise SystemExit(f'Output already exists: {out_path}. Pass --force to overwrite.')

    calibration_seed = int(args.seed if args.seed is not None else config.runtime.seed)
    seed_everything(calibration_seed, deterministic=bool(config.runtime.deterministic))

    split_indices = list(config.data.split_indices) if config.data.split_indices else [int(config.data.split_idx)]
    payload: dict[str, Any] = {
        'kind': 'online_expected_budget_wilson_q_table',
        'source_config': str(args.config),
        'sampling_regime': str(config.experiment.sampling_regime),
        'budget_unit': str(config.sampling.budget_unit),
        'budget_values': [float(v) for v in config.sampling.budget_values],
        'calibration_seed': calibration_seed,
        'q_search_trials': int(args.trials or config.sampling.wilson.q_calibration_trials),
        'validation_trials': int(args.validation_trials),
        'q_min': float(config.sampling.wilson.q_min),
        'q_max': float(config.sampling.wilson.q_max),
        'target_budget_tolerance': float(config.sampling.wilson.target_budget_tolerance),
        'datasets': {},
    }

    device = torch.device('cpu')
    for dataset_name in config.data.datasets:
        for feature_mode in config.data.feature_modes:
            for split_idx in split_indices:
                dataset_cfg = _dataset_config(
                    config,
                    dataset_name=dataset_name,
                    feature_mode=feature_mode,
                    split_idx=int(split_idx),
                )
                dataset = load_dataset(dataset_cfg, device=device, seed=calibration_seed)
                candidate_nodes = dataset.train_nodes.to(device=device, dtype=torch.long)
                budgets = _budget_counts(config, int(candidate_nodes.numel()))

                sampler = WilsonCenterSampler(
                    edge_index=dataset.data.edge_index,
                    num_nodes=int(dataset.data.num_nodes),
                    device=device,
                    backend=config.sampling.wilson.backend,
                    q=None,
                    redraw_on_shortfall=False,
                    max_sampling_attempts=1,
                    allow_uniform_fill=False,
                    make_undirected=config.sampling.wilson.make_undirected,
                    remove_self_loops=config.sampling.wilson.remove_self_loops,
                    auto_q_method='mean_degree',
                    q_calibration_trials=int(args.trials or config.sampling.wilson.q_calibration_trials),
                    q_calibration_max_iters=config.sampling.wilson.q_calibration_max_iters,
                    q_calibration_tolerance=config.sampling.wilson.q_calibration_tolerance,
                    q_min=config.sampling.wilson.q_min,
                    q_max=config.sampling.wilson.q_max,
                    q_calibration_seed=config.sampling.wilson.q_calibration_seed,
                    budget_targeting='expected_budget',
                    target_budget_tolerance=config.sampling.wilson.target_budget_tolerance,
                    max_budget_search_steps=config.sampling.wilson.max_budget_search_steps,
                    budget_target_failure_policy=config.sampling.wilson.budget_target_failure_policy,
                    weight_estimation=config.sampling.wilson.weight_estimation,
                    num_weight_forests=config.sampling.wilson.num_weight_forests,
                    weight_estimation_seed=config.sampling.wilson.weight_estimation_seed,
                    dataset_name=dataset.name,
                )

                rows: dict[str, Any] = {}
                for budget in budgets:
                    q, search_metadata = sampler._search_q_for_budget(
                        candidate_nodes,
                        int(budget),
                        seed=int(config.sampling.wilson.q_calibration_seed) + int(split_idx) * 100_000 + int(budget),
                    )
                    validation = _validate_q(
                        sampler,
                        candidate_nodes,
                        float(q),
                        seed=calibration_seed + int(split_idx) * 100_000 + int(budget) * 1000,
                        trials=int(args.validation_trials),
                    )
                    rel_error = abs(float(validation['mean_roots']) - float(budget)) / max(float(budget), 1.0)
                    rows[str(int(budget))] = {
                        'q': float(q),
                        'target_budget': float(budget),
                        'train_node_count': float(candidate_nodes.numel()),
                        'num_nodes': float(dataset.data.num_nodes),
                        'validation_mean_roots': float(validation['mean_roots']),
                        'validation_std_roots': float(validation['std_roots']),
                        'validation_min_roots': float(validation['min_roots']),
                        'validation_max_roots': float(validation['max_roots']),
                        'validation_relative_error': float(rel_error),
                        **{str(k): float(v) for k, v in search_metadata.items() if isinstance(v, (int, float))},
                    }
                    print(
                        f'{dataset.name} split={split_idx} budget={budget}: '
                        f'q={float(q):.8g}, mean_roots={validation["mean_roots"]:.3f}, '
                        f'std={validation["std_roots"]:.3f}, rel_err={rel_error:.4f}'
                    )

                key = dataset.name
                if len(config.data.feature_modes) > 1 or len(split_indices) > 1:
                    key = f'{dataset.name}__{feature_mode}__split{split_idx}'
                payload['datasets'][key] = {
                    'dataset_name': dataset.name,
                    'feature_mode': feature_mode,
                    'split_idx': int(split_idx),
                    'split_mode': config.data.split.mode,
                    'train_node_count': int(candidate_nodes.numel()),
                    'q_by_budget_count': rows,
                }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
