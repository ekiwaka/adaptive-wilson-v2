from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ..sampling.wilson_sampler import WilsonCenterSampler


@dataclass(slots=True)
class WilsonExactKernel:
    """Exact q-forest root kernel K = q (L + qI)^{-1}.

    The implementation uses linear solves rather than explicitly forming a
    matrix inverse. It is intended for diagnostics and small-to-medium graphs.
    """

    kernel: torch.Tensor
    expected_root_count: float
    root_marginals: torch.Tensor


def _dense_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = edge_index.to(dtype=torch.long, device='cpu')
    adjacency = torch.zeros((int(num_nodes), int(num_nodes)), dtype=torch.float64)
    if edge_index.numel() == 0:
        return adjacency
    src, dst = edge_index[0], edge_index[1]
    valid = (src >= 0) & (src < int(num_nodes)) & (dst >= 0) & (dst < int(num_nodes)) & (src != dst)
    src = src[valid]
    dst = dst[valid]
    adjacency[src, dst] = 1.0
    adjacency[dst, src] = 1.0
    return adjacency


def laplacian_dense(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adjacency = _dense_undirected_adjacency(edge_index, num_nodes)
    degree = torch.diag(adjacency.sum(dim=1))
    return degree - adjacency


def exact_wilson_root_kernel(edge_index: torch.Tensor, num_nodes: int, q: float) -> WilsonExactKernel:
    if q <= 0.0:
        raise ValueError('q must be positive for Wilson exact checks.')
    laplacian = laplacian_dense(edge_index, num_nodes)
    eye = torch.eye(int(num_nodes), dtype=torch.float64)
    # Solve (L + qI) X = I and multiply by q, instead of explicitly inverting.
    kernel = float(q) * torch.linalg.solve(laplacian + float(q) * eye, eye)
    marginals = kernel.diag().clamp(min=0.0, max=1.0)
    return WilsonExactKernel(
        kernel=kernel,
        expected_root_count=float(torch.trace(kernel).item()),
        root_marginals=marginals,
    )


def pairwise_root_inclusion(kernel: torch.Tensor, pairs: Iterable[tuple[int, int]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i, j in pairs:
        i = int(i)
        j = int(j)
        kii = float(kernel[i, i].item())
        kjj = float(kernel[j, j].item())
        kij = float(kernel[i, j].item())
        kji = float(kernel[j, i].item())
        det = kii * kjj - kij * kji
        independent = kii * kjj
        rows.append(
            {
                'i': float(i),
                'j': float(j),
                'p_i': kii,
                'p_j': kjj,
                'pair_inclusion_determinant': det,
                'independent_product': independent,
                'negative_correlation_gap': independent - det,
            }
        )
    return rows


def make_default_pairs(num_nodes: int, *, max_pairs: int = 64) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(int(num_nodes)):
        for j in range(i + 1, int(num_nodes)):
            pairs.append((i, j))
            if len(pairs) >= int(max_pairs):
                return pairs
    return pairs


def estimate_wilson_root_marginals_by_forests(
    edge_index: torch.Tensor,
    num_nodes: int,
    q: float,
    *,
    num_forests: int,
    seed: int,
    backend: str = 'python',
) -> dict[str, torch.Tensor | list[float]]:
    if num_forests <= 0:
        raise ValueError('num_forests must be positive.')
    sampler = WilsonCenterSampler(
        edge_index=edge_index.cpu(),
        num_nodes=int(num_nodes),
        device=torch.device('cpu'),
        backend=backend,
        q=float(q),
        allow_uniform_fill=False,
        budget_targeting='legacy',
    )
    counts = torch.zeros(int(num_nodes), dtype=torch.float64)
    root_counts: list[float] = []
    generator = torch.Generator(device='cpu')
    for forest_idx in range(int(num_forests)):
        generator.manual_seed(int(seed) + forest_idx)
        _, roots_mask, _, _ = sampler.backend.sample_forest_with_q_factor(q=float(q), generator=generator)
        roots_mask = roots_mask.to(device='cpu', dtype=torch.bool)
        counts += roots_mask.to(dtype=torch.float64)
        root_counts.append(float(roots_mask.sum().item()))
    return {
        'root_marginals': counts / float(num_forests),
        'root_counts': root_counts,
    }


def cumulative_absolute_error(root_counts: list[float], reference: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    total = 0.0
    for idx, value in enumerate(root_counts, start=1):
        total += float(value)
        estimate = total / float(idx)
        rows.append(
            {
                'num_forests': float(idx),
                'mean_root_count': estimate,
                'absolute_error': abs(estimate - float(reference)),
                'reference_expected_roots': float(reference),
            }
        )
    return rows


def symmetry_invariance_error(edge_index: torch.Tensor, num_nodes: int, q: float, *, seed: int) -> float:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    permutation = torch.randperm(int(num_nodes), generator=generator)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(int(num_nodes))
    permuted_edges = permutation[edge_index.cpu().long()]
    original = exact_wilson_root_kernel(edge_index, int(num_nodes), q).root_marginals
    permuted = exact_wilson_root_kernel(permuted_edges, int(num_nodes), q).root_marginals[inverse]
    return float((original - permuted).abs().max().item())
