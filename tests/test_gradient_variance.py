from __future__ import annotations
import pytest
import torch
from torch_geometric.data import Data

pytest.importorskip('torch_geometric')

from gnn_graph_sampling.analysis.gradient_variance import estimate_gradient_variance
from gnn_graph_sampling.config.schema import DatasetConfig, ModelConfig, OptimizationConfig, SamplingConfig, WilsonConfig
from gnn_graph_sampling.data.loading import DatasetBundle



def _make_bundle() -> DatasetBundle:
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    undirected_edges: list[tuple[int, int]] = []
    for src, dst in edge_pairs:
        undirected_edges.append((src, dst))
        undirected_edges.append((dst, src))
    edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
    x = torch.randn(6, 4)
    y = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    train_mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)
    val_mask = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.bool)
    test_mask = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)
    data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=6)
    return DatasetBundle(
        name='ToyGradient',
        kind='synthetic',
        data=data,
        num_features=4,
        num_classes=2,
        train_nodes=torch.nonzero(train_mask, as_tuple=False).view(-1),
        val_nodes=torch.nonzero(val_mask, as_tuple=False).view(-1),
        test_nodes=torch.nonzero(test_mask, as_tuple=False).view(-1),
        metadata={},
    )



def test_gradient_variance_returns_rows_for_requested_samplers_and_batch_sizes() -> None:
    bundle = _make_bundle()
    dataset_config = DatasetConfig(name='toy_gradient', feature_mode='enabled')
    model_config = ModelConfig(hidden_dim=8, num_layers=3, dropout=0.2, head_hidden_dim=8)
    optimization_config = OptimizationConfig(epochs=1, use_class_weights=False, label_smoothing=0.0)
    sampling_config = SamplingConfig(
        name='uniform',
        batch_size=2,
        steps_per_epoch=2,
        eval_batch_size=4,
        num_hops=1,
        sample_inference=False,
        inference_max_batches=4,
        wilson=WilsonConfig(backend='python', q=1.0, max_sampling_attempts=4),
    )

    rows = estimate_gradient_variance(
        dataset=bundle,
        dataset_config=dataset_config,
        model_config=model_config,
        optimization_config=optimization_config,
        sampling_config=sampling_config,
        sampler_names=['uniform', 'degree', 'wilson'],
        batch_sizes=[2, 4],
        seed=0,
    )

    assert len(rows) == 6
    for row in rows:
        assert row['dataset_name'] == 'ToyGradient'
        assert row['sampler_name'] in {'uniform', 'degree', 'wilson'}
        assert row['batch_size'] in {2, 4}
        assert row['gradient_variance_trace'] >= 0.0
        assert row['mean_gradient_norm'] >= 0.0
        assert row['std_gradient_norm'] >= 0.0
