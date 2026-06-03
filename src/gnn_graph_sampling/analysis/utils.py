from __future__ import annotations

import torch

from ..config.schema import SamplingConfig
from ..data.loading import DatasetBundle
from ..sampling.center_selectors import DegreeCenterSampler, UniformCenterSampler
from ..sampling.wilson_sampler import AdaptiveWilsonCenterSampler, WilsonCenterSampler


def build_center_sampler(sampling: SamplingConfig, dataset: DatasetBundle, device: torch.device):
    """Instantiate a center sampler from sampling configuration."""

    sampler_device = torch.device('cpu')

    if sampling.name == 'none':
        return None
    if sampling.name == 'uniform':
        return UniformCenterSampler()
    if sampling.name == 'degree':
        return DegreeCenterSampler(dataset.data.edge_index, int(dataset.data.num_nodes), sampler_device)
    if sampling.name in {'wilson', 'adaptive_wilson'}:
        sampler_cls = AdaptiveWilsonCenterSampler if sampling.name == 'adaptive_wilson' else WilsonCenterSampler
        adaptive_kwargs = {}
        if sampling.name == 'adaptive_wilson':
            adaptive_kwargs = {
                'alpha': sampling.adaptive_wilson.alpha,
                'q_i_mode': sampling.adaptive_wilson.q_i_mode,
            }
        return sampler_cls(
            edge_index=dataset.data.edge_index,
            num_nodes=int(dataset.data.num_nodes),
            device=sampler_device,
            backend=sampling.wilson.backend,
            q=sampling.wilson.q,
            redraw_on_shortfall=sampling.wilson.redraw_on_shortfall,
            max_sampling_attempts=sampling.wilson.max_sampling_attempts,
            allow_uniform_fill=sampling.wilson.allow_uniform_fill,
            make_undirected=sampling.wilson.make_undirected,
            remove_self_loops=sampling.wilson.remove_self_loops,
            auto_q_method=sampling.wilson.auto_q_method,
            q_calibration_trials=sampling.wilson.q_calibration_trials,
            q_calibration_max_iters=sampling.wilson.q_calibration_max_iters,
            q_calibration_tolerance=sampling.wilson.q_calibration_tolerance,
            q_min=sampling.wilson.q_min,
            q_max=sampling.wilson.q_max,
            q_calibration_seed=sampling.wilson.q_calibration_seed,
            budget_targeting=sampling.wilson.budget_targeting,
            target_budget_tolerance=sampling.wilson.target_budget_tolerance,
            max_budget_search_steps=sampling.wilson.max_budget_search_steps,
            budget_target_failure_policy=sampling.wilson.budget_target_failure_policy,
            weight_estimation=sampling.wilson.weight_estimation,
            num_weight_forests=sampling.wilson.num_weight_forests,
            weight_estimation_seed=sampling.wilson.weight_estimation_seed,
            dataset_name=dataset.name,
            q_table_path=sampling.wilson.q_table_path,
            q_table_missing_policy=sampling.wilson.q_table_missing_policy,
            **adaptive_kwargs,
        )
    raise ValueError(f'Unsupported sampler: {sampling.name}')
