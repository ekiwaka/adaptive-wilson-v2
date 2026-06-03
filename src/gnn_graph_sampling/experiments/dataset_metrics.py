from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from ..analysis.graph_diagnostics import compute_graph_diagnostics
from ..config.loader import load_config
from ..config.schema import DatasetConfig, DatasetMetricsConfig
from ..data.loading import load_dataset
from ..utils.device import resolve_device
from ..utils.io import slugify, write_csv, write_json
from ..utils.logging import get_logger
from .common import configure_study_logging, parse_config_arg, prepare_experiment_root, save_resolved_config

LOGGER = get_logger(__name__)


def _plot_degree_histogram(diagnostics: dict[str, Any], out_path: Path) -> None:
    degree = diagnostics.get('degree_distribution', {})
    counts = degree.get('histogram_counts', [])
    edges = degree.get('histogram_bin_edges', [])
    if not counts or not edges:
        return
    centers = [(edges[index] + edges[index + 1]) / 2.0 for index in range(len(edges) - 1)]
    width = max(1e-6, centers[1] - centers[0] if len(centers) > 1 else 1.0)
    plt.figure(figsize=(8, 5))
    plt.bar(centers, counts, width=width)
    plt.xlabel('Degree')
    plt.ylabel('Node count')
    plt.title(f"Degree distribution: {diagnostics['dataset_name']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def _plot_class_distribution(diagnostics: dict[str, Any], out_path: Path) -> None:
    counts = diagnostics.get('data_distribution', {}).get('class_counts', [])
    if not counts:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Class')
    plt.ylabel('Node count')
    plt.title(f"Class distribution: {diagnostics['dataset_name']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def _plot_component_sizes(diagnostics: dict[str, Any], out_path: Path) -> None:
    sizes = diagnostics.get('component_size_histogram', [])
    if not sizes:
        return
    top = sizes[: min(50, len(sizes))]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(top)), top)
    plt.xlabel('Component rank by size')
    plt.ylabel('Nodes')
    plt.title(f"Connected component sizes: {diagnostics['dataset_name']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def _plot_homophily_summary(diagnostics: dict[str, Any], out_path: Path) -> None:
    community = diagnostics.get('community_tendency', {})
    edge_homophily = float(community.get('edge_homophily', 0.0))
    heterophily = 1.0 - edge_homophily
    plt.figure(figsize=(6, 5))
    plt.bar(['homophily', 'heterophily'], [edge_homophily, heterophily])
    plt.ylim(0.0, 1.0)
    plt.ylabel('Fraction of edges')
    plt.title(f"Edge homophily/heterophily: {diagnostics['dataset_name']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def _summary_row(diagnostics: dict[str, Any]) -> dict[str, Any]:
    data_dist = diagnostics['data_distribution']
    community = diagnostics['community_tendency']
    degree = diagnostics['degree_distribution']
    return {
        'dataset_name': diagnostics['dataset_name'],
        'dataset_kind': diagnostics['dataset_kind'],
        'num_nodes': diagnostics['num_nodes'],
        'num_edges': diagnostics['num_edges'],
        'num_features': diagnostics['num_features'],
        'num_classes': diagnostics['num_classes'],
        'density': diagnostics['density'],
        'sparsity': diagnostics['sparsity'],
        'degree_min': degree['min'],
        'degree_max': degree['max'],
        'degree_mean': degree['mean'],
        'degree_std': degree['std'],
        'edge_homophily': community['edge_homophily'],
        'edge_heterophily': 1.0 - float(community['edge_homophily']),
        'connected_components_count': diagnostics['connected_components_count'],
        'largest_component_size': diagnostics['largest_component_size'],
        'train_size': data_dist['train_size'],
        'val_size': data_dist['val_size'],
        'test_size': data_dist['test_size'],
        'imbalance_ratio': data_dist['imbalance_ratio'],
    }


def run_dataset_metrics(config: DatasetMetricsConfig) -> Path:
    config.validate()
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

    resolution = resolve_device(config.runtime.device, config.runtime.device_policy)
    device = torch.device(resolution.resolved)
    rows: list[dict[str, Any]] = []

    for dataset_name in config.data.datasets:
        for feature_mode in config.data.feature_modes:
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
            LOGGER.info('Loading dataset=%s feature_mode=%s for metrics.', dataset_name, feature_mode)
            bundle = load_dataset(dataset_config, device=device, seed=config.runtime.seed)
            diagnostics = compute_graph_diagnostics(
                bundle,
                seed=config.runtime.seed,
                max_exact_path_nodes=config.metrics.max_exact_path_nodes,
                spectral_num_communities=config.metrics.spectral_num_communities,
            )
            diagnostics['resolved_device'] = resolution.resolved
            diagnostics['device_warnings'] = resolution.warnings

            dataset_dir = root / slugify(dataset_name) / slugify(feature_mode)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            write_json(dataset_dir / 'dataset_metrics.json', diagnostics)

            class_rows = [
                {'class_id': index, 'node_count': int(count)}
                for index, count in enumerate(diagnostics['data_distribution']['class_counts'])
            ]
            write_csv(dataset_dir / 'class_distribution.csv', class_rows)

            degree = diagnostics['degree_distribution']
            degree_rows = [
                {'bin_left': degree['histogram_bin_edges'][i], 'bin_right': degree['histogram_bin_edges'][i + 1], 'count': degree['histogram_counts'][i]}
                for i in range(len(degree['histogram_counts']))
            ]
            write_csv(dataset_dir / 'degree_histogram.csv', degree_rows)

            if config.metrics.save_plots:
                _plot_degree_histogram(diagnostics, dataset_dir / 'degree_distribution.png')
                _plot_class_distribution(diagnostics, dataset_dir / 'class_distribution.png')
                _plot_component_sizes(diagnostics, dataset_dir / 'component_sizes.png')
                _plot_homophily_summary(diagnostics, dataset_dir / 'homophily_heterophily.png')

            rows.append(_summary_row(diagnostics))

    write_csv(root / 'tables' / 'dataset_metrics_summary.csv', rows)
    write_json(root / 'tables' / 'dataset_metrics_summary.json', {'rows': rows})
    LOGGER.info('Dataset metrics workflow complete. Outputs written to %s', root)
    return root


def main() -> None:
    args = parse_config_arg('Download/load datasets and generate dataset metrics.')
    config = load_config(DatasetMetricsConfig, args.config)
    run_dataset_metrics(config)


if __name__ == '__main__':  # pragma: no cover
    main()
