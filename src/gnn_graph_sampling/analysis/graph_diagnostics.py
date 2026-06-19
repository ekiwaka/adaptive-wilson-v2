from __future__ import annotations
import random
from typing import Any, Optional
import networkx as nx
import numpy as np
import torch
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..data.loading import DatasetBundle



def _to_networkx(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_nodes)))
    edges = list(zip(edge_index[0].cpu().tolist(), edge_index[1].cpu().tolist()))
    graph.add_edges_from(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph



def _approx_average_shortest_path(graph: nx.Graph, *, seed: int, max_sources: int = 64) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    nodes = list(graph.nodes())
    rng = random.Random(int(seed))
    if len(nodes) > max_sources:
        nodes = rng.sample(nodes, max_sources)
    distances: list[int] = []
    for source in nodes:
        lengths = nx.single_source_shortest_path_length(graph, source)
        distances.extend(int(distance) for target, distance in lengths.items() if target != source)
    return float(sum(distances) / max(len(distances), 1))



def _label_modularity(graph: nx.Graph, labels: list[int]) -> float:
    communities: dict[int, set[int]] = {}
    for node, label in enumerate(labels):
        communities.setdefault(int(label), set()).add(int(node))
    parts = [community for community in communities.values() if community]
    if not parts or graph.number_of_edges() == 0:
        return 0.0
    return float(nx.algorithms.community.quality.modularity(graph, parts))



def compute_graph_diagnostics(
    bundle: DatasetBundle,
    *,
    seed: int,
    max_exact_path_nodes: int = 2000,
    spectral_num_communities: Optional[int] = None,
) -> dict[str, Any]:
    """Compute graph-level diagnostics used in benchmark reporting.

    The function reports degree statistics, sparsity, connected components,
    approximate small-world behaviour, homophily, modularity, and a basic
    spectral community-detection proxy against the known node labels.
    """

    graph = _to_networkx(bundle.data.edge_index, int(bundle.data.num_nodes))
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    density = nx.density(graph) if num_nodes > 1 else 0.0
    sparsity = 1.0 - float(density)

    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=np.float64)
    histogram_counts, histogram_edges = np.histogram(
        degrees,
        bins=min(32, max(2, int(np.sqrt(max(len(degrees), 1))))),
    ) if degrees.size else (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))

    components = [list(component) for component in nx.connected_components(graph)]
    components.sort(key=len, reverse=True)
    component_sizes = [len(component) for component in components]
    largest_component = graph.subgraph(components[0]).copy() if components else nx.Graph()

    if largest_component.number_of_nodes() <= max_exact_path_nodes and largest_component.number_of_nodes() > 1:
        average_path_length = float(nx.average_shortest_path_length(largest_component))
    else:
        average_path_length = _approx_average_shortest_path(largest_component, seed=seed)

    clustering = float(nx.average_clustering(graph)) if num_nodes > 0 else 0.0
    er_graph = nx.gnm_random_graph(num_nodes, num_edges, seed=int(seed))
    er_components = list(nx.connected_components(er_graph))
    er_largest = er_graph.subgraph(max(er_components, key=len)).copy() if er_components else nx.Graph()
    if er_largest.number_of_nodes() <= max_exact_path_nodes and er_largest.number_of_nodes() > 1:
        er_path_length = float(nx.average_shortest_path_length(er_largest))
    else:
        er_path_length = _approx_average_shortest_path(er_largest, seed=seed)
    er_clustering = float(nx.average_clustering(er_graph)) if er_graph.number_of_nodes() > 0 else 0.0
    small_world_sigma = 0.0
    if er_clustering > 0.0 and er_path_length > 0.0 and average_path_length > 0.0:
        small_world_sigma = float((clustering / er_clustering) / (average_path_length / er_path_length))

    labels = bundle.data.y.detach().cpu().to(torch.long)
    class_counts = torch.bincount(labels, minlength=bundle.num_classes).tolist()
    imbalance_ratio = float(max(class_counts) / max(min(class_counts), 1)) if class_counts else 1.0
    edge_index_cpu = bundle.data.edge_index.detach().cpu().to(torch.long)
    edge_homophily = float((labels[edge_index_cpu[0]] == labels[edge_index_cpu[1]]).float().mean().item()) if edge_index_cpu.numel() else 0.0
    label_modularity = _label_modularity(graph, labels.tolist())

    spectral_k = int(spectral_num_communities or bundle.num_classes)
    ari = 0.0
    nmi = 0.0
    spectral_modularity = 0.0
    if largest_component.number_of_nodes() >= max(2, spectral_k):
        node_order = list(largest_component.nodes())
        index_of = {node: idx for idx, node in enumerate(node_order)}
        rows: list[int] = []
        cols: list[int] = []
        for src, dst in largest_component.edges():
            rows.extend([index_of[src], index_of[dst]])
            cols.extend([index_of[dst], index_of[src]])
        adjacency = sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(len(node_order), len(node_order)))
        laplacian = csgraph.laplacian(adjacency, normed=True)
        eig_vectors = None
        try:
            _, eig_vectors = eigsh(laplacian, k=min(spectral_k, laplacian.shape[0] - 1), which='SM')
        except Exception:
            eig_vectors = None
        if eig_vectors is not None and eig_vectors.shape[0] >= spectral_k:
            clusters = KMeans(n_clusters=spectral_k, n_init=10, random_state=int(seed)).fit_predict(eig_vectors)
            true_labels = labels[node_order].numpy()
            ari = float(adjusted_rand_score(true_labels, clusters))
            nmi = float(normalized_mutual_info_score(true_labels, clusters))
            communities: list[set[int]] = []
            assigned_nodes: set[int] = set()
            for cluster_id in range(spectral_k):
                members = {int(node_order[idx]) for idx in np.where(clusters == cluster_id)[0].tolist()}
                if members:
                    communities.append(members)
                    assigned_nodes.update(members)

            remaining_nodes = set(graph.nodes()) - assigned_nodes
            if remaining_nodes:
                communities.append(remaining_nodes)

            if communities and graph.number_of_edges() > 0 and nx.algorithms.community.is_partition(graph, communities):
                spectral_modularity = float(nx.algorithms.community.quality.modularity(graph, communities))

    features = bundle.data.x.detach().cpu().to(torch.float32)
    feature_density = float((features != 0).float().mean().item()) if features.numel() else 0.0
    label_entropy = 0.0
    if class_counts:
        probabilities = np.asarray(class_counts, dtype=np.float64) / max(sum(class_counts), 1)
        probabilities = probabilities[probabilities > 0]
        label_entropy = float(-(probabilities * np.log2(probabilities)).sum())

    return {
        'dataset_name': bundle.name,
        'dataset_kind': bundle.kind,
        'num_nodes': int(num_nodes),
        'num_edges': int(num_edges),
        'num_features': int(bundle.num_features),
        'num_classes': int(bundle.num_classes),
        'density': float(density),
        'sparsity': float(sparsity),
        'degree_distribution': {
            'min': float(degrees.min()) if degrees.size else 0.0,
            'max': float(degrees.max()) if degrees.size else 0.0,
            'mean': float(degrees.mean()) if degrees.size else 0.0,
            'std': float(degrees.std()) if degrees.size else 0.0,
            'p50': float(np.percentile(degrees, 50)) if degrees.size else 0.0,
            'p90': float(np.percentile(degrees, 90)) if degrees.size else 0.0,
            'p99': float(np.percentile(degrees, 99)) if degrees.size else 0.0,
            'histogram_counts': histogram_counts.tolist(),
            'histogram_bin_edges': histogram_edges.tolist(),
        },
        'average_path_length_lcc': float(average_path_length),
        'average_clustering_coefficient': float(clustering),
        'connected_components_count': len(components),
        'largest_component_size': int(component_sizes[0]) if component_sizes else 0,
        'component_size_histogram': component_sizes,
        'small_world': {
            'sigma': float(small_world_sigma),
            'graph_clustering': float(clustering),
            'graph_path_length': float(average_path_length),
            'er_clustering': float(er_clustering),
            'er_path_length': float(er_path_length),
        },
        'community_tendency': {
            'edge_homophily': float(edge_homophily),
            'label_modularity': float(label_modularity),
        },
        'spectral_community_detection': {
            'num_communities': spectral_k,
            'ari_vs_labels': float(ari),
            'nmi_vs_labels': float(nmi),
            'modularity': float(spectral_modularity),
        },
        'data_distribution': {
            'class_counts': class_counts,
            'imbalance_ratio': float(imbalance_ratio),
            'label_entropy_bits': float(label_entropy),
            'feature_density': float(feature_density),
            'feature_mean': float(features.mean().item()) if features.numel() else 0.0,
            'feature_std': float(features.std().item()) if features.numel() else 0.0,
            'train_size': int(bundle.train_nodes.numel()),
            'val_size': int(bundle.val_nodes.numel()),
            'test_size': int(bundle.test_nodes.numel()),
        },
    }
