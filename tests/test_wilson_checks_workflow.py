from __future__ import annotations

import torch

from gnn_graph_sampling.analysis.wilson_checks import (
    cumulative_absolute_error,
    exact_wilson_root_kernel,
    estimate_wilson_root_marginals_by_forests,
    make_default_pairs,
    pairwise_root_inclusion,
)


def test_wilson_checks_exact_and_monte_carlo_outputs_are_finite() -> None:
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long)
    exact = exact_wilson_root_kernel(edge_index, 4, q=1.0)
    assert exact.expected_root_count > 0.0
    assert torch.isfinite(exact.root_marginals).all()
    pairs = pairwise_root_inclusion(exact.kernel, make_default_pairs(4, max_pairs=3))
    assert pairs
    assert all(row['negative_correlation_gap'] >= -1e-8 for row in pairs)
    mc = estimate_wilson_root_marginals_by_forests(edge_index, 4, 1.0, num_forests=3, seed=0, backend='python')
    assert len(mc['root_counts']) == 3
    rows = cumulative_absolute_error([float(v) for v in mc['root_counts']], exact.expected_root_count)
    assert len(rows) == 3
