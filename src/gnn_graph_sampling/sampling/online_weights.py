from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(slots=True)
class OnlineSampleWeights:
    """Position-aligned weights for one online minibatch.

    The weights implement a self-normalized inverse-inclusion-probability
    mini-batch loss estimator:

        sum_i w_i loss_i / sum_i w_i,

    where w_i is proportional to 1 / pi_i and pi_i is the probability that node i
    is included as a sampled center under the current online sampler.
    """

    weights: torch.Tensor
    method: str
    metadata: dict[str, float | str]


def _mean_one(weights: torch.Tensor, eps: float) -> torch.Tensor:
    weights = weights.detach().cpu().to(torch.float32)
    weights = torch.clamp(weights, min=eps)
    if not torch.isfinite(weights).all():
        raise ValueError('online sample weights contain non-finite values before normalization.')
    return weights / torch.clamp(weights.mean(), min=eps)


def _inverse_probability_weights(inclusion_probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    probabilities = inclusion_probabilities.detach().cpu().to(torch.float32)
    probabilities = torch.clamp(probabilities, min=eps, max=1.0)
    if not torch.isfinite(probabilities).all():
        raise ValueError('online inclusion probabilities contain non-finite values.')
    return _mean_one(1.0 / probabilities, eps=eps)


def _metadata_from_weights(
    *,
    weights: torch.Tensor,
    probabilities: torch.Tensor | None,
    method: str,
    extra: Mapping[str, float | str] | None = None,
) -> dict[str, float | str]:
    metadata: dict[str, float | str] = {
        'min_weight': float(weights.min().item()),
        'max_weight': float(weights.max().item()),
        'mean_weight': float(weights.mean().item()),
        'weight_sum': float(weights.sum().item()),
        'weight_method': method,
    }
    if probabilities is not None:
        probabilities = probabilities.detach().cpu().to(torch.float32)
        metadata.update(
            {
                'min_inclusion_probability': float(probabilities.min().item()),
                'max_inclusion_probability': float(probabilities.max().item()),
                'mean_inclusion_probability': float(probabilities.mean().item()),
            }
        )
    if extra:
        metadata.update(dict(extra))
    return metadata


def compute_uniform_online_weights(center_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.ones(center_nodes.numel(), dtype=torch.float32)
    weights = torch.ones(center_nodes.numel(), dtype=torch.float32)
    return weights, probabilities


def compute_degree_online_probabilities(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    center_nodes: torch.Tensor,
    candidate_nodes: torch.Tensor,
    budget: int,
    eps: float,
) -> torch.Tensor:
    """Approximate inclusion probabilities for degree PPS sampling.

    DegreeCenterSampler samples without replacement with probabilities
    proportional to degree. Exact PPS-without-replacement inclusion
    probabilities require a sequential calculation; for online loss weighting we
    use the standard with-replacement inclusion approximation

        pi_i ≈ 1 - (1 - p_i)^k,

    where p_i = degree_i / sum_candidate degree and k is the requested center
    budget. This is a materially better estimator than using only the per-draw
    p_i, and it has the right limiting behavior as k grows.
    """

    src = edge_index.detach().cpu()[0].to(torch.long)
    degree = torch.bincount(src, minlength=int(num_nodes)).to(torch.float32)
    degree = torch.clamp(degree, min=1.0)

    candidate_nodes_cpu = candidate_nodes.detach().cpu().to(torch.long)
    center_nodes_cpu = center_nodes.detach().cpu().to(torch.long)

    candidate_mass = torch.clamp(degree[candidate_nodes_cpu].sum(), min=eps)
    per_draw_probability = torch.clamp(degree[center_nodes_cpu] / candidate_mass, min=eps, max=1.0)
    effective_budget = max(1, min(int(budget), int(candidate_nodes_cpu.numel())))

    inclusion_probability = 1.0 - torch.pow(1.0 - per_draw_probability, effective_budget)
    return torch.clamp(inclusion_probability, min=eps, max=1.0)


def compute_degree_online_weights(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    center_nodes: torch.Tensor,
    candidate_nodes: torch.Tensor,
    budget: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = compute_degree_online_probabilities(
        edge_index=edge_index,
        num_nodes=num_nodes,
        center_nodes=center_nodes,
        candidate_nodes=candidate_nodes,
        budget=budget,
        eps=eps,
    )
    return _inverse_probability_weights(probabilities, eps=eps), probabilities


def _wilson_online_weights_from_sampler(
    *,
    sampler: Any,
    center_nodes: torch.Tensor,
    candidate_nodes: torch.Tensor,
    budget: int,
    sample_metadata: Mapping[str, float | str],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | str]] | None:
    estimator = getattr(sampler, 'center_inclusion_probabilities', None)
    if estimator is None:
        return None

    probabilities, probability_metadata = estimator(
        center_nodes=center_nodes,
        candidate_nodes=candidate_nodes,
        budget=int(budget),
        sample_metadata=sample_metadata,
        eps=eps,
    )
    probabilities = probabilities.detach().cpu().to(torch.float32)
    weights = _inverse_probability_weights(probabilities, eps=eps)
    return weights, probabilities, dict(probability_metadata)


def compute_online_sample_weights(
    *,
    sampler_name: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    center_nodes: torch.Tensor,
    candidate_nodes: torch.Tensor,
    sample_metadata: Mapping[str, float | str] | None = None,
    eps: float = 1e-12,
    budget: int | None = None,
    sampler: Any | None = None,
) -> OnlineSampleWeights:
    """Compute normalized online loss weights aligned with center_nodes.

    The returned weights are proportional to inverse inclusion probabilities and
    normalized to have mean one. The loss itself is still computed as

        sum_i w_i loss_i / sum_i w_i,

    so multiplying all weights by a constant would not change the loss. Mean-one
    normalization keeps diagnostics and gradient scales easier to compare.
    """

    sample_metadata = sample_metadata or {}
    sampler_name = sampler_name.lower()
    if center_nodes.numel() == 0:
        raise ValueError('Cannot compute online weights for an empty minibatch.')

    effective_budget = int(budget) if budget is not None else int(center_nodes.numel())
    effective_budget = max(1, effective_budget)

    if sampler_name in {'none', 'uniform'}:
        weights, probabilities = compute_uniform_online_weights(center_nodes)
        method = 'uniform_equal_probability_mean_one_online'
        return OnlineSampleWeights(
            weights=weights,
            method=method,
            metadata=_metadata_from_weights(weights=weights, probabilities=probabilities, method=method),
        )

    if sampler_name == 'degree':
        weights, probabilities = compute_degree_online_weights(
            edge_index=edge_index,
            num_nodes=num_nodes,
            center_nodes=center_nodes,
            candidate_nodes=candidate_nodes,
            budget=effective_budget,
            eps=eps,
        )
        method = 'degree_inverse_inclusion_probability_mean_one_online'
        return OnlineSampleWeights(
            weights=weights,
            method=method,
            metadata=_metadata_from_weights(weights=weights, probabilities=probabilities, method=method),
        )

    if sampler_name in {'wilson', 'adaptive_wilson'}:
        estimated = None
        if sampler is not None:
            estimated = _wilson_online_weights_from_sampler(
                sampler=sampler,
                center_nodes=center_nodes,
                candidate_nodes=candidate_nodes,
                budget=effective_budget,
                sample_metadata=sample_metadata,
                eps=eps,
            )

        if estimated is not None:
            weights, probabilities, probability_metadata = estimated
            method = (
                'adaptive_wilson_inverse_root_marginal_mean_one_online'
                if sampler_name == 'adaptive_wilson'
                else 'wilson_inverse_root_marginal_mean_one_online'
            )
            metadata = _metadata_from_weights(
                weights=weights,
                probabilities=probabilities,
                method=method,
                extra=probability_metadata,
            )
            metadata['wilson_online_exact_weights_available'] = float(
                str(probability_metadata.get('wilson_online_probability_estimation', '')).startswith('exact')
            )
            metadata['weight_estimation'] = str(probability_metadata.get('wilson_online_probability_estimation', 'unknown'))
            return OnlineSampleWeights(weights=weights, method=method, metadata=metadata)

        # Explicit fallback: this should be rare after the Wilson sampler support
        # below, but keeps old custom samplers from silently crashing.
        weights = torch.ones(center_nodes.numel(), dtype=torch.float32)
        probabilities = torch.ones(center_nodes.numel(), dtype=torch.float32)
        method = 'wilson_uniform_proxy_no_probability_estimator_online'
        metadata = _metadata_from_weights(weights=weights, probabilities=probabilities, method=method)
        metadata.update(
            {
                'wilson_online_exact_weights_available': 0.0,
                'wilson_online_probability_estimation': 'missing_sampler_estimator_fallback_uniform',
                'weight_estimation': 'online_uniform_proxy_for_wilson',
            }
        )
        return OnlineSampleWeights(weights=weights, method=method, metadata=metadata)

    raise ValueError(f'Unsupported sampler for online sample weights: {sampler_name!r}')
