from __future__ import annotations
import torch

from gnn_graph_sampling.sampling.wilson_backend import PythonWilsonForestBackend, WilsonBackendFactory



def test_python_backend_can_be_selected_explicitly() -> None:
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    backend = WilsonBackendFactory.create('python', edge_index=edge_index, num_nodes=3, device=torch.device('cpu'))
    assert isinstance(backend, PythonWilsonForestBackend)
