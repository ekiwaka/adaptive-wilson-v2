from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import BenchmarkConfig, DatasetConfig, SamplingConfig
from gnn_graph_sampling.data.loading import load_dataset
from gnn_graph_sampling.sampling.wilson_sampler import WilsonCenterSampler
from gnn_graph_sampling.training.preprocessing import resolve_budget_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Precompute Wilson q values from a benchmark YAML.')
    parser.add_argument('--config', type=Path, required=True, help='Benchmark YAML config.')
    parser.add_argument('--output', type=Path, default=None, help='Override q-table output path.')
    return parser.parse_args()


def _dataset_config(config: BenchmarkConfig, dataset_name: str, feature_mode: str) -> DatasetConfig:
    return DatasetConfig(
        name=dataset_name,
        root=config.data.root,
        split_idx=config.data.split_idx,
        planetoid_split=config.data.planetoid_split,
        feature_mode=feature_mode,
        standardize_features=config.data.standardize_features,
        split=config.data.split,
        synthetic=config.data.synthetic,
    )


def _sampling_config(config: BenchmarkConfig, budget_value: float) -> SamplingConfig:
    return SamplingConfig(
        name='wilson',
        mode='static',
        online_schedule='per_batch',
        budget_unit=config.sampling.budget_unit,
        budget_value=float(budget_value),
        batch_size=config.sampling.batch_sizes[0] if config.sampling.batch_sizes else 128,
        steps_per_epoch=config.sampling.steps_per_epoch,
        eval_batch_size=config.sampling.eval_batch_size,
        num_hops=config.sampling.num_hops[0] if config.sampling.num_hops else 1,
        sample_inference=config.sampling.sample_inference,
        inference_max_batches=config.sampling.inference_max_batches,
        cache_preprocessed_graph=config.sampling.cache_preprocessed_graph,
        reuse_cached_preprocessed_graph=config.sampling.reuse_cached_preprocessed_graph,
        cache_dirname=config.sampling.cache_dirname,
        wilson=config.sampling.wilson,
    )


def _resolve_output_path(args: argparse.Namespace, config: BenchmarkConfig) -> Path:
    if args.output is not None:
        return args.output
    if config.sampling.wilson.q_table_path:
        return Path(config.sampling.wilson.q_table_path)
    raise ValueError('Pass --output or set sampling.wilson.q_table_path in the YAML.')


def main() -> None:
    args = parse_args()
    config = load_config(BenchmarkConfig, args.config).validate()
    wilson = config.sampling.wilson

    output_path = _resolve_output_path(args, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        'version': 1,
        'source_config': str(args.config),
        'sampler_name': 'wilson',
        'budget_unit': config.sampling.budget_unit,
        'make_undirected': bool(wilson.make_undirected),
        'remove_self_loops': bool(wilson.remove_self_loops),
        'backend': wilson.backend,
        'q_calibration_trials': int(wilson.q_calibration_trials),
        'q_calibration_max_iters': int(wilson.q_calibration_max_iters),
        'q_calibration_tolerance': float(wilson.q_calibration_tolerance),
        'q_min': float(wilson.q_min),
        'q_max': float(wilson.q_max),
        'datasets': {},
    }

    feature_mode = (config.data.feature_modes or ['enabled'])[0]

    for dataset_name in config.data.datasets:
        data_config = _dataset_config(config, dataset_name, feature_mode)
        bundle = load_dataset(data_config, device=torch.device('cpu'), seed=int(config.runtime.seed))
        num_nodes = int(bundle.data.num_nodes)
        candidate_nodes = (
            torch.arange(num_nodes, dtype=torch.long)
            if data_config.split.mode == 'sampler'
            else bundle.train_nodes.detach().cpu().to(torch.long)
        )

        sampler = WilsonCenterSampler(
            edge_index=bundle.data.edge_index,
            num_nodes=num_nodes,
            device=torch.device('cpu'),
            backend=wilson.backend,
            q=None,
            redraw_on_shortfall=False,
            max_sampling_attempts=1,
            allow_uniform_fill=False,
            make_undirected=wilson.make_undirected,
            remove_self_loops=wilson.remove_self_loops,
            auto_q_method='mean_degree',
            q_calibration_trials=wilson.q_calibration_trials,
            q_calibration_max_iters=wilson.q_calibration_max_iters,
            q_calibration_tolerance=wilson.q_calibration_tolerance,
            q_min=wilson.q_min,
            q_max=wilson.q_max,
            q_calibration_seed=wilson.q_calibration_seed,
            budget_targeting='q_target_search',
            target_budget_tolerance=wilson.q_calibration_tolerance,
            max_budget_search_steps=wilson.q_calibration_max_iters,
            budget_target_failure_policy='warn_and_continue',
            dataset_name=bundle.name,
        )

        dataset_table: dict[str, Any] = {
            'dataset_name': bundle.name,
            'num_nodes': num_nodes,
            'num_candidate_nodes': int(candidate_nodes.numel()),
            'split_mode': data_config.split.mode,
            'q_by_budget_count': {},
        }

        for budget_value in config.sampling.budget_values:
            budget_count = resolve_budget_count(
                _sampling_config(config, float(budget_value)),
                int(candidate_nodes.numel()),
            )
            q, metadata = sampler._search_q_for_budget(  # internal calibration helper
                candidate_nodes.to(torch.long),
                budget=int(budget_count),
                seed=int(wilson.q_calibration_seed) + int(budget_count),
            )
            row = {
                'q': float(q),
                'budget_value': float(budget_value),
                'requested_budget': int(budget_count),
                'mean_raw_roots': float(metadata.get('q_search_estimated_budget', -1.0)),
                'relative_error': float(metadata.get('q_search_rel_error', -1.0)),
                'trials': int(wilson.q_calibration_trials),
                'max_iters': int(wilson.q_calibration_max_iters),
                'tolerance': float(wilson.q_calibration_tolerance),
            }
            dataset_table['q_by_budget_count'][str(int(budget_count))] = row
            print(
                f"[{bundle.name}] budget_value={budget_value} budget_count={budget_count} "
                f"q={q:.8g} mean_roots={row['mean_raw_roots']:.2f} rel_err={row['relative_error']:.4f}"
            )

        payload['datasets'][bundle.name] = dataset_table

    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f'Wrote Wilson q table to: {output_path}')


if __name__ == '__main__':
    main()
