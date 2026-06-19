from __future__ import annotations

import pytest
import torch

pytest.importorskip('torch_geometric.utils')

from gnn_graph_sampling.sampling.online_weights import compute_online_sample_weights
from gnn_graph_sampling.training.loop import compute_positional_weighted_cross_entropy


def test_uniform_weighted_loss_equals_unweighted_when_weights_are_equal() -> None:
    torch.manual_seed(0)
    logits = torch.randn(6, 3)
    target = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()
    sample_weight = torch.ones(6)
    weighted = compute_positional_weighted_cross_entropy(logits, target, loss_fn, sample_weight)
    unweighted = loss_fn(logits, target)
    assert torch.allclose(weighted, unweighted)


def test_degree_online_weights_are_finite_and_non_constant() -> None:
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 2, 3, 4, 5]],
        dtype=torch.long,
    )
    center_nodes = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    candidate_nodes = torch.arange(6, dtype=torch.long)
    weights = compute_online_sample_weights(
        sampler_name='degree',
        edge_index=edge_index,
        num_nodes=6,
        center_nodes=center_nodes,
        candidate_nodes=candidate_nodes,
    )
    assert torch.isfinite(weights.weights).all()
    assert weights.weights.numel() == center_nodes.numel()
    assert pytest.approx(float(weights.weights.mean()), rel=1e-6) == 1.0
    assert float(weights.weights.max()) > float(weights.weights.min())


def test_wilson_online_weights_are_finite() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    center_nodes = torch.tensor([0, 2], dtype=torch.long)
    candidate_nodes = torch.arange(3, dtype=torch.long)
    weights = compute_online_sample_weights(
        sampler_name='wilson',
        edge_index=edge_index,
        num_nodes=3,
        center_nodes=center_nodes,
        candidate_nodes=candidate_nodes,
        sample_metadata={'q': 1.0},
    )
    assert torch.isfinite(weights.weights).all()
    assert torch.allclose(weights.weights, torch.ones_like(weights.weights))
    assert weights.metadata['weight_estimation'] == 'online_uniform_proxy_for_wilson'


def test_non_finite_sample_weights_fail_clearly() -> None:
    logits = torch.randn(2, 2)
    target = torch.tensor([0, 1])
    with pytest.raises(ValueError, match='non-finite'):
        compute_positional_weighted_cross_entropy(
            logits,
            target,
            torch.nn.CrossEntropyLoss(),
            torch.tensor([1.0, float('nan')]),
        )
