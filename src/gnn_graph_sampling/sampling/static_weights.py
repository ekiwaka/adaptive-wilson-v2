from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch

from .base import CenterSampler

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:  # pragma: no cover
    sp = None
    spla = None


@dataclass(slots=True)
class StaticSampleWeights:
    """Position-aligned weights for sampler-selected train_nodes."""

    weights: torch.Tensor
    method: str
    metadata: dict[str, float | str]


def _normalise_to_mean_one(weights: torch.Tensor, eps: float) -> torch.Tensor:
    weights = weights.to(dtype=torch.float32)
    weights = torch.clamp(weights, min=eps)
    return weights / torch.clamp(weights.mean(), min=eps)


def compute_static_sample_weights(
    *,
    sampler_name: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    train_nodes: torch.Tensor,
    sample_metadata: Mapping[str, float | str],
    eps: float = 1e-12,
    strict_one_pass_wilson: bool = True,
    wilson_weight_estimation: str = 'exact',
    wilson_num_weight_forests: int = 256,
    wilson_weight_estimation_seed: int = 314159,
    sampler: Optional[CenterSampler] = None,
    budget_count: Optional[int] = None,
) -> StaticSampleWeights:
    """Compute normalized static sample weights aligned with train_nodes."""

    sampler_name = sampler_name.lower()

    if sampler_name in {'none', 'uniform'}:
        weights = torch.ones(train_nodes.numel(), dtype=torch.float32)
        return StaticSampleWeights(
            weights=weights,
            method='uniform_mean_one',
            metadata={'min_weight': 1.0, 'max_weight': 1.0},
        )

    if sampler_name == 'degree':
        weights = compute_degree_static_weights(
            edge_index=edge_index,
            num_nodes=num_nodes,
            train_nodes=train_nodes,
            eps=eps,
        )
        return StaticSampleWeights(
            weights=weights,
            method='degree_inverse_probability_mean_one',
            metadata={
                'min_weight': float(weights.min().item()),
                'max_weight': float(weights.max().item()),
            },
        )

    if sampler_name != 'wilson':
        raise ValueError(f'Unsupported sampler for static sample weights: {sampler_name!r}')

    q_for_weights = (
        sample_metadata.get('q_used_for_sample')
        or sample_metadata.get('q_final')
        or sample_metadata.get('q_initial')
    )

    if q_for_weights is None:
        raise ValueError(
            'Wilson static sample weighting requires one of '
            "sample_metadata['q_used_for_sample'], "
            "sample_metadata['q_final'], or "
            "sample_metadata['q_initial']."
        )

    q_redraws_used = float(sample_metadata.get('q_redraws_used', 0.0))
    fallback_uniform = float(sample_metadata.get('fallback_uniform', 0.0))

    if wilson_weight_estimation == 'exact':
        if strict_one_pass_wilson:
            budget_targeting = float(sample_metadata.get('budget_targeting', 0.0))
            one_pass_wilson = float(sample_metadata.get('one_pass_wilson', 0.0))

            if q_redraws_used > 0:
                raise ValueError(
                    'Strict Wilson static weighting requires one-pass Wilson sampling. '
                    f'Got q_redraws_used={q_redraws_used}.'
                )
            if fallback_uniform > 0:
                raise ValueError(
                    'Strict Wilson static weighting requires allow_uniform_fill=false. '
                    f'Got fallback_uniform={fallback_uniform}.'
                )

            if budget_targeting > 0.0:
                raise ValueError(
                    'Strict exact Wilson weights require fixed-q one-pass Wilson sampling, '
                    'but sample metadata indicates budget_targeting was enabled. '
                    'Use monte_carlo_forests for q_target_search.'
                )

            if one_pass_wilson <= 0.0:
                raise ValueError(
                    'Strict exact Wilson weights require one_pass_wilson=1. '
                    'Use monte_carlo_forests when the final sample is selected after '
                    'multiple Wilson draws or adaptive budget targeting.'
                )

        weights = compute_wilson_static_weights_exact(
            edge_index=edge_index,
            num_nodes=num_nodes,
            train_nodes=train_nodes,
            q=float(q_for_weights),
            eps=eps,
        )
        return StaticSampleWeights(
            weights=weights,
            method='wilson_inverse_root_probability_mean_one_exact',
            metadata={
                'q_used_for_weights': float(q_for_weights),
                'min_weight': float(weights.min().item()),
                'max_weight': float(weights.max().item()),
                'q_redraws_used': q_redraws_used,
                'fallback_uniform': fallback_uniform,
            },
        )

    if wilson_weight_estimation != 'monte_carlo_forests':
        raise ValueError('wilson_weight_estimation must be exact or monte_carlo_forests.')
    if sampler is None:
        raise ValueError('sampler must be provided for Wilson Monte Carlo static weights.')
    if budget_count is None:
        raise ValueError('budget_count must be provided for Wilson Monte Carlo static weights.')

    weights, mc_metadata = compute_wilson_static_weights_monte_carlo(
        sampler=sampler,
        num_nodes=num_nodes,
        train_nodes=train_nodes,
        budget_count=int(budget_count),
        num_forests=int(wilson_num_weight_forests),
        seed=int(wilson_weight_estimation_seed),
        eps=eps,
    )
    return StaticSampleWeights(
        weights=weights,
        method='wilson_inverse_inclusion_probability_mean_one_monte_carlo',
        metadata={
            'q_used_for_weights': float(q_for_weights),
            'min_weight': float(weights.min().item()),
            'max_weight': float(weights.max().item()),
            'num_weight_forests': float(wilson_num_weight_forests),
            'q_redraws_used': q_redraws_used,
            'fallback_uniform': fallback_uniform,
            **mc_metadata,
        },
    )


def compute_degree_static_weights(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    train_nodes: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    src = edge_index[0].detach().cpu().to(torch.long)
    degree = torch.bincount(src, minlength=int(num_nodes)).to(torch.float32)
    degree = torch.clamp(degree, min=1.0)

    train_nodes_cpu = train_nodes.detach().cpu().to(torch.long)
    probs = degree[train_nodes_cpu] / torch.clamp(degree.sum(), min=eps)
    raw_weights = 1.0 / torch.clamp(probs, min=eps)
    return _normalise_to_mean_one(raw_weights, eps=eps)


def compute_wilson_static_weights_exact(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    train_nodes: torch.Tensor,
    q: float,
    eps: float,
) -> torch.Tensor:
    """Exact q(L+qI)^(-1) diagonal evaluation via sparse linear solves."""

    if q <= 0.0:
        raise ValueError(f'Wilson q must be positive. Got q={q}.')
    if sp is None or spla is None:
        raise ImportError(
            'compute_wilson_static_weights_exact requires scipy. Install scipy or use monte_carlo_forests.'
        )

    edge_index_cpu = edge_index.detach().cpu().to(torch.long)
    row = edge_index_cpu[0].numpy()
    col = edge_index_cpu[1].numpy()
    values = np.ones(row.shape[0], dtype=np.float64)

    adjacency = sp.coo_matrix(
        (values, (row, col)),
        shape=(int(num_nodes), int(num_nodes)),
        dtype=np.float64,
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    laplacian = sp.diags(degree, format='csr') - adjacency
    matrix = (laplacian + float(q) * sp.eye(int(num_nodes), format='csr')).tocsc()
    solve = spla.factorized(matrix)

    train_nodes_np = train_nodes.detach().cpu().to(torch.long).numpy()
    root_probs = np.empty(train_nodes_np.shape[0], dtype=np.float64)
    for out_idx, node_id in enumerate(train_nodes_np):
        basis = np.zeros(int(num_nodes), dtype=np.float64)
        basis[int(node_id)] = 1.0
        solution = solve(basis)
        root_probs[out_idx] = float(q) * float(solution[int(node_id)])

    root_probs = np.clip(root_probs, eps, None)
    raw_weights = 1.0 / root_probs
    weights = torch.from_numpy(raw_weights).to(torch.float32)
    return _normalise_to_mean_one(weights, eps=eps)


def compute_wilson_static_weights_monte_carlo(
    *,
    sampler: CenterSampler,
    num_nodes: int,
    train_nodes: torch.Tensor,
    budget_count: int,
    num_forests: int,
    seed: int,
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Estimate final inclusion probabilities by repeated Wilson sampler draws."""

    if num_forests <= 0:
        raise ValueError('num_forests must be positive.')

    counts = torch.zeros(int(num_nodes), dtype=torch.float64)
    candidate_nodes = torch.arange(int(num_nodes), dtype=torch.long)
    generator = torch.Generator(device='cpu')
    train_nodes_cpu = train_nodes.detach().cpu().to(torch.long)

    realized_sizes: list[int] = []
    for forest_idx in range(int(num_forests)):
        generator.manual_seed(int(seed) + forest_idx)
        sample_out = sampler.sample(candidate_nodes, int(budget_count), generator=generator)
        chosen = torch.unique(sample_out.center_nodes.detach().cpu().to(torch.long), sorted=False)
        counts[chosen] += 1.0
        realized_sizes.append(int(chosen.numel()))

    inclusion_probs = counts[train_nodes_cpu] / float(num_forests)
    inclusion_probs = torch.clamp(inclusion_probs, min=eps)
    raw_weights = 1.0 / inclusion_probs
    weights = _normalise_to_mean_one(raw_weights.to(torch.float32), eps=eps)

    realized = torch.tensor(realized_sizes, dtype=torch.float32)
    metadata = {
        'mc_mean_realized_budget': float(realized.mean().item()),
        'mc_std_realized_budget': float(realized.std(unbiased=False).item()),
        'mc_min_inclusion_prob': float(inclusion_probs.min().item()),
        'mc_max_inclusion_prob': float(inclusion_probs.max().item()),
    }
    return weights, metadata
