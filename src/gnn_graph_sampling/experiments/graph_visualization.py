from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import torch

from ..config.loader import load_config
from ..config.schema import DatasetConfig, GraphVisualizationConfig
from ..data.loading import DatasetBundle, load_dataset
from ..utils.device import resolve_device
from ..utils.io import slugify, write_json
from ..utils.logging import get_logger
from .common import configure_study_logging, parse_config_arg, prepare_experiment_root, save_resolved_config

LOGGER = get_logger(__name__)


def _to_networkx(bundle: DatasetBundle) -> nx.Graph:
    graph = nx.Graph()
    num_nodes = int(bundle.data.num_nodes)
    graph.add_nodes_from(range(num_nodes))
    edge_index = bundle.data.edge_index.detach().cpu().to(torch.long)
    graph.add_edges_from((int(src), int(dst)) for src, dst in edge_index.t().tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _select_nodes(
    graph: nx.Graph,
    *,
    max_nodes: int | None,
    method: str,
    seed: int,
) -> list[int]:
    """Select nodes for visualization.

    Supported methods:
      - full_graph: visualize the entire graph. Ignores max_nodes.
      - largest_component: visualize nodes from the largest connected component,
        capped by max_nodes if provided.
      - random: random node sample, capped by max_nodes.
    """

    method = str(method).lower()

    if method == 'full_graph':
        return list(graph.nodes())

    if max_nodes is None:
        return list(graph.nodes())

    if max_nodes <= 0:
        raise ValueError(f'visualization.max_nodes must be positive or null. Got {max_nodes}.')

    if graph.number_of_nodes() <= max_nodes:
        return list(graph.nodes())

    if method == 'largest_component':
        component = max(nx.connected_components(graph), key=len)
        nodes = list(component)
        if len(nodes) <= max_nodes:
            return nodes

        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed))
        perm = torch.randperm(len(nodes), generator=generator).tolist()
        return [nodes[index] for index in perm[:max_nodes]]

    if method == 'random':
        nodes = list(graph.nodes())
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed))
        perm = torch.randperm(len(nodes), generator=generator).tolist()
        return [nodes[index] for index in perm[:max_nodes]]

    raise ValueError(
        f"Unsupported visualization.subgraph_method={method!r}. "
        "Expected one of: full_graph, largest_component, random."
    )


def _layout(graph: nx.Graph, *, name: str, seed: int) -> dict[int, tuple[float, float]]:
    if name == 'spring':
        return nx.spring_layout(graph, seed=int(seed))
    if name == 'kamada_kawai':
        return nx.kamada_kawai_layout(graph)
    if name == 'spectral':
        return nx.spectral_layout(graph)
    if name == 'shell':
        return nx.shell_layout(graph)
    if name == 'circular':
        return nx.circular_layout(graph)
    if name == 'random':
        return nx.random_layout(graph, seed=int(seed))
    raise ValueError(f'Unsupported layout: {name}')


def _node_colors(bundle: DatasetBundle, nodes: list[int], graph: nx.Graph, color_by: str) -> list[float]:
    if color_by == 'label' and hasattr(bundle.data, 'y'):
        labels = bundle.data.y.detach().cpu().to(torch.float32)
        return [float(labels[node].item()) for node in nodes]
    if color_by == 'split':
        train = set(bundle.train_nodes.detach().cpu().tolist())
        val = set(bundle.val_nodes.detach().cpu().tolist())
        values = []
        for node in nodes:
            if node in train:
                values.append(0.0)
            elif node in val:
                values.append(1.0)
            else:
                values.append(2.0)
        return values
    if color_by == 'degree':
        return [float(graph.degree(node)) for node in nodes]
    return [0.0 for _ in nodes]


def _node_sizes(graph: nx.Graph, nodes: list[int], *, node_size_by: str, base_size: float) -> list[float]:
    if node_size_by == 'degree':
        degrees = [float(graph.degree(node)) for node in nodes]
        max_degree = max(max(degrees), 1.0)
        return [base_size * (0.5 + 2.0 * degree / max_degree) for degree in degrees]
    return [float(base_size) for _ in nodes]


def visualize_dataset(bundle: DatasetBundle, config: GraphVisualizationConfig, out_path: Path) -> dict[str, Any]:
    graph = _to_networkx(bundle)
    nodes = _select_nodes(
        graph,
        max_nodes=config.visualization.max_nodes,
        method=config.visualization.subgraph_method,
        seed=config.runtime.seed,
    )
    subgraph = graph.subgraph(nodes).copy()
    pos = _layout(subgraph, name=config.visualization.layout, seed=config.runtime.seed)

    colors = _node_colors(bundle, list(subgraph.nodes()), subgraph, config.visualization.color_by)
    sizes = _node_sizes(
        subgraph,
        list(subgraph.nodes()),
        node_size_by=config.visualization.node_size_by,
        base_size=config.visualization.node_size,
    )

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(subgraph, pos, alpha=float(config.visualization.edge_alpha), width=0.5)
    nodes_collection = nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_color=colors,
        node_size=sizes,
        alpha=float(config.visualization.node_alpha),
        cmap='viridis',
        linewidths=0.0,
    )
    if config.visualization.color_by != 'constant':
        plt.colorbar(nodes_collection, shrink=0.75, label=config.visualization.color_by)
    if config.visualization.label_nodes:
        if config.visualization.label_with == 'label' and hasattr(bundle.data, 'y'):
            labels_tensor = bundle.data.y.detach().cpu().to(torch.long)
            labels = {node: str(int(labels_tensor[node].item())) for node in subgraph.nodes()}
        else:
            labels = {node: str(int(node)) for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=6)
    plt.title(f"{bundle.name}: {subgraph.number_of_nodes()} nodes / {subgraph.number_of_edges()} edges")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(config.visualization.dpi), bbox_inches='tight')
    plt.close()

    return {
        'dataset_name': bundle.name,
        'full_num_nodes': int(graph.number_of_nodes()),
        'full_num_edges': int(graph.number_of_edges()),
        'visualized_num_nodes': int(subgraph.number_of_nodes()),
        'visualized_num_edges': int(subgraph.number_of_edges()),
        'layout': config.visualization.layout,
        'color_by': config.visualization.color_by,
        'node_size_by': config.visualization.node_size_by,
        'label_nodes': bool(config.visualization.label_nodes),
        'label_with': config.visualization.label_with,
        'output_path': str(out_path),
    }


def run_graph_visualization(config: GraphVisualizationConfig) -> Path:
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
            bundle = load_dataset(dataset_config, device=device, seed=config.runtime.seed)
            dataset_dir = root / slugify(dataset_name) / slugify(feature_mode)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            figure_path = dataset_dir / f"graph_structure.{config.visualization.output_format}"
            row = visualize_dataset(bundle, config, figure_path)
            write_json(dataset_dir / 'visualization_metadata.json', row)
            rows.append(row)

    write_json(root / 'tables' / 'graph_visualization_summary.json', {'rows': rows})
    LOGGER.info('Graph visualization workflow complete. Outputs written to %s', root)
    return root


def main() -> None:
    args = parse_config_arg('Generate graph structure visualizations.')
    config = load_config(GraphVisualizationConfig, args.config)
    run_graph_visualization(config)


if __name__ == '__main__':  # pragma: no cover
    main()
