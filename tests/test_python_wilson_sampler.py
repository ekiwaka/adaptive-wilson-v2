from __future__ import annotations
import torch

from gnn_graph_sampling.sampling.wilson_backend import PythonWilsonForestBackend



def test_python_wilson_q_sampler_returns_valid_roots() -> None:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )
    backend = PythonWilsonForestBackend(edge_index=edge_index, num_nodes=4, device=torch.device('cpu'))
    generator = torch.Generator(device='cpu')
    generator.manual_seed(0)
    parent, roots_mask, stats, _ = backend.sample_forest_with_q_factor(q=1.0, generator=generator)

    assert parent.shape == (4,)
    assert roots_mask.shape == (4,)
    assert int(roots_mask.sum().item()) >= 1
    assert stats['num_roots'] >= 1
