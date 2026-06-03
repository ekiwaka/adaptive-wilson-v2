from __future__ import annotations
import torch
from .base import CenterSampler, SamplerOutput


class UniformCenterSampler(CenterSampler):
    """Uniformly sample center nodes without replacement."""

    name = 'uniform'

    def sample(self, candidate_nodes: torch.Tensor, budget: int, *, generator: torch.Generator) -> SamplerOutput:
        if candidate_nodes.numel() == 0:
            raise ValueError('candidate_nodes must be non-empty.')
        budget = min(int(budget), int(candidate_nodes.numel()))
        permutation = torch.randperm(candidate_nodes.numel(), generator=generator, device=candidate_nodes.device)
        chosen = candidate_nodes[permutation[:budget]]
        return SamplerOutput(
            center_nodes=chosen,
            metadata={'requested_budget': float(budget), 'realized_budget': float(chosen.numel()), 'fallback_uniform': 0.0},
        )


class DegreeCenterSampler(CenterSampler):
    """Sample center nodes with probability proportional to node degree."""

    name = 'degree'

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> None:
        src = edge_index[0].to(dtype=torch.long)
        degree_values = torch.bincount(src, minlength=int(num_nodes)).to(dtype=torch.float32, device=device)
        self.degree = torch.clamp(degree_values, min=1.0)

    def sample(self, candidate_nodes: torch.Tensor, budget: int, *, generator: torch.Generator) -> SamplerOutput:
        if candidate_nodes.numel() == 0:
            raise ValueError('candidate_nodes must be non-empty.')
        budget = min(int(budget), int(candidate_nodes.numel()))
        weights = self.degree[candidate_nodes]
        sampled_local = torch.multinomial(weights, num_samples=budget, replacement=False, generator=generator)
        chosen = candidate_nodes[sampled_local]
        return SamplerOutput(
            center_nodes=chosen,
            metadata={
                'requested_budget': float(budget),
                'realized_budget': float(chosen.numel()),
                'mean_center_degree': float(weights[sampled_local].mean().item()),
                'fallback_uniform': 0.0,
            },
        )
