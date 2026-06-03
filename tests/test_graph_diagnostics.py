from __future__ import annotations
import math
import pytest
import torch
from torch_geometric.data import Data

pytest.importorskip('torch_geometric')

from gnn_graph_sampling.analysis.graph_diagnostics import compute_graph_diagnostics
from gnn_graph_sampling.data.loading import DatasetBundle



def _make_bundle(edge_pairs: list[tuple[int, int]], labels: list[int], x_dim: int = 3) -> DatasetBundle:
    num_nodes = len(labels)
    undirected_edges: list[tuple[int, int]] = []
    for src, dst in edge_pairs:
        undirected_edges.append((src, dst))
        undirected_edges.append((dst, src))
    edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
    x = torch.arange(num_nodes * x_dim, dtype=torch.float32).view(num_nodes, x_dim)
    y = torch.tensor(labels, dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[: max(1, num_nodes // 2)] = True
    val_mask[max(1, num_nodes // 2): max(2, (3 * num_nodes) // 4)] = True
    test_mask[~(train_mask | val_mask)] = True
    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=num_nodes)
    return DatasetBundle(
        name='Toy',
        kind='synthetic',
        data=data,
        num_features=x_dim,
        num_classes=len(set(labels)),
        train_nodes=torch.nonzero(train_mask, as_tuple=False).view(-1),
        val_nodes=torch.nonzero(val_mask, as_tuple=False).view(-1),
        test_nodes=torch.nonzero(test_mask, as_tuple=False).view(-1),
        metadata={},
    )



def test_path_graph_diagnostics_have_expected_values() -> None:
    bundle = _make_bundle(edge_pairs=[(0, 1), (1, 2), (2, 3)], labels=[0, 0, 1, 1])
    diagnostics = compute_graph_diagnostics(bundle, seed=0)

    assert diagnostics['num_nodes'] == 4
    assert diagnostics['num_edges'] == 3
    assert math.isclose(diagnostics['density'], 0.5, rel_tol=1e-6)
    assert math.isclose(diagnostics['sparsity'], 0.5, rel_tol=1e-6)
    assert math.isclose(diagnostics['average_path_length_lcc'], 5.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(diagnostics['average_clustering_coefficient'], 0.0, abs_tol=1e-9)
    assert diagnostics['connected_components_count'] == 1
    assert diagnostics['largest_component_size'] == 4

    degree_distribution = diagnostics['degree_distribution']
    assert math.isclose(degree_distribution['min'], 1.0, rel_tol=1e-6)
    assert math.isclose(degree_distribution['max'], 2.0, rel_tol=1e-6)
    assert math.isclose(degree_distribution['mean'], 1.5, rel_tol=1e-6)
    assert degree_distribution['histogram_counts']

    community_tendency = diagnostics['community_tendency']
    assert math.isclose(community_tendency['edge_homophily'], 2.0 / 3.0, rel_tol=1e-6)

    data_distribution = diagnostics['data_distribution']
    assert data_distribution['class_counts'] == [2, 2]
    assert data_distribution['train_size'] >= 1
    assert data_distribution['val_size'] >= 1
    assert data_distribution['test_size'] >= 1



def test_disconnected_graph_reports_component_statistics() -> None:
    bundle = _make_bundle(
        edge_pairs=[
            (0, 1), (1, 2), (2, 0),
            (3, 4), (4, 5), (5, 3),
        ],
        labels=[0, 0, 0, 1, 1, 1],
    )
    diagnostics = compute_graph_diagnostics(bundle, seed=0)

    assert diagnostics['connected_components_count'] == 2
    assert diagnostics['largest_component_size'] == 3
    assert diagnostics['component_size_histogram'] == [3, 3]
    assert math.isclose(diagnostics['average_path_length_lcc'], 1.0, rel_tol=1e-6)



def test_community_and_spectral_sections_are_present_on_clustered_graph() -> None:
    bundle = _make_bundle(
        edge_pairs=[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
        labels=[0, 0, 0, 1, 1, 1],
    )
    diagnostics = compute_graph_diagnostics(bundle, seed=0)

    assert diagnostics['community_tendency']['label_modularity'] > 0.0
    spectral = diagnostics['spectral_community_detection']
    assert spectral['num_communities'] == 2
    assert -1.0 <= spectral['ari_vs_labels'] <= 1.0
    assert 0.0 <= spectral['nmi_vs_labels'] <= 1.0
    assert isinstance(spectral['modularity'], float)
    assert 'small_world' in diagnostics
    assert 'sigma' in diagnostics['small_world']
