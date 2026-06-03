from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from ..analysis.graph_diagnostics import compute_graph_diagnostics
from ..analysis.gradient_variance import estimate_gradient_variance
from ..config.loader import load_config
from ..config.schema import DatasetConfig, DiagnosticsConfig, SamplingConfig
from ..data.loading import load_dataset
from ..utils.device import resolve_device
from ..utils.io import slugify, write_csv, write_json
from ..utils.logging import get_logger
from .common import configure_study_logging, parse_config_arg, prepare_experiment_root, save_resolved_config

LOGGER = get_logger(__name__)



def _plot_degree_histogram(diagnostics: dict[str, object], out_path: Path) -> None:
    degree_distribution = diagnostics['degree_distribution']
    counts = degree_distribution['histogram_counts']
    edges = degree_distribution['histogram_bin_edges']
    if not counts or not edges:
        return
    centers = [(edges[index] + edges[index + 1]) / 2.0 for index in range(len(edges) - 1)]
    plt.figure(figsize=(8, 5))
    width = max(1e-6, centers[1] - centers[0] if len(centers) > 1 else 1.0)
    plt.bar(centers, counts, width=width)
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title(f"Degree distribution: {diagnostics['dataset_name']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()



def _plot_class_distribution(diagnostics: dict[str, object], out_path: Path) -> None:
    counts = diagnostics['data_distribution']['class_counts']
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Class')
    plt.ylabel('Node count')
    plt.title(f"Class distribution: {diagnostics['dataset_name']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()



def _plot_gradient_variance(rows: list[dict[str, object]], out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for sampler_name in sorted({str(row['sampler_name']) for row in rows}):
        subset = [row for row in rows if str(row['sampler_name']) == sampler_name]
        plt.plot(
            [int(row['batch_size']) for row in subset],
            [float(row['gradient_variance_trace']) for row in subset],
            marker='o',
            label=sampler_name,
        )
    plt.xlabel('Batch size')
    plt.ylabel('Gradient variance trace')
    plt.title('Gradient variance vs batch size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()



def run_diagnostics_study(config: DiagnosticsConfig) -> Path:
    """Run dataset diagnostics and gradient-variance analysis."""

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
    all_rows: list[dict[str, object]] = []

    for dataset_name in config.data.datasets:
        for feature_mode in config.data.feature_modes:
            LOGGER.info('Computing diagnostics for dataset=%s feature_mode=%s', dataset_name, feature_mode)
            dataset_config = DatasetConfig(
                name=dataset_name,
                root=config.data.root,
                split_idx=config.data.split_idx,
                feature_mode=feature_mode,
                standardize_features=config.data.standardize_features,
                synthetic=config.data.synthetic,
            )
            bundle = load_dataset(dataset_config, device=device, seed=config.runtime.seed)
            diagnostics = compute_graph_diagnostics(bundle, seed=config.runtime.seed)
            diagnostics['resolved_device'] = resolution.resolved
            diagnostics['device_warnings'] = resolution.warnings

            dataset_dir = root / slugify(dataset_name) / feature_mode
            dataset_dir.mkdir(parents=True, exist_ok=True)
            write_json(dataset_dir / 'diagnostics.json', diagnostics)
            _plot_degree_histogram(diagnostics, dataset_dir / 'degree_distribution.png')
            _plot_class_distribution(diagnostics, dataset_dir / 'class_distribution.png')

            base_sampling = SamplingConfig(
                name='uniform',
                batch_size=config.sampling.batch_sizes[0],
                steps_per_epoch=config.diagnostics.gradient_variance_batches,
                eval_batch_size=config.sampling.eval_batch_size,
                num_hops=config.sampling.num_hops[0],
                sample_inference=config.sampling.sample_inference,
                inference_max_batches=config.sampling.inference_max_batches,
                wilson=config.sampling.wilson,
            )
            gradient_rows = estimate_gradient_variance(
                dataset=bundle,
                dataset_config=dataset_config,
                model_config=config.model,
                optimization_config=config.optimization,
                sampling_config=base_sampling,
                sampler_names=config.sampling.samplers,
                batch_sizes=config.sampling.batch_sizes,
                seed=config.runtime.seed,
            )
            write_csv(dataset_dir / 'gradient_variance.csv', gradient_rows)
            _plot_gradient_variance(gradient_rows, dataset_dir / 'gradient_variance_vs_batch_size.png')

            all_rows.append(
                {
                    'dataset_name': diagnostics['dataset_name'],
                    'feature_mode': feature_mode,
                    'model_family': config.model.family,
                    'num_nodes': diagnostics['num_nodes'],
                    'num_edges': diagnostics['num_edges'],
                    'num_features': diagnostics['num_features'],
                    'num_classes': diagnostics['num_classes'],
                    'density': diagnostics['density'],
                    'sparsity': diagnostics['sparsity'],
                    'average_path_length_lcc': diagnostics['average_path_length_lcc'],
                    'average_clustering_coefficient': diagnostics['average_clustering_coefficient'],
                    'connected_components_count': diagnostics['connected_components_count'],
                    'largest_component_size': diagnostics['largest_component_size'],
                    'edge_homophily': diagnostics['community_tendency']['edge_homophily'],
                    'label_modularity': diagnostics['community_tendency']['label_modularity'],
                    'spectral_ari': diagnostics['spectral_community_detection']['ari_vs_labels'],
                    'spectral_nmi': diagnostics['spectral_community_detection']['nmi_vs_labels'],
                    'imbalance_ratio': diagnostics['data_distribution']['imbalance_ratio'],
                }
            )

    write_csv(root / 'tables' / 'dataset_diagnostics_summary.csv', all_rows)
    LOGGER.info('Diagnostics study complete. Outputs written to %s', root)
    return root



def main() -> None:
    """CLI entrypoint for diagnostics."""

    args = parse_config_arg('Run graph diagnostics and gradient-variance analysis.')
    config = load_config(DiagnosticsConfig, args.config)
    run_diagnostics_study(config)


if __name__ == '__main__':  # pragma: no cover
    main()
