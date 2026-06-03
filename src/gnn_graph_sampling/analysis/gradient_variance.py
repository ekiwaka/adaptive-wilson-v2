from __future__ import annotations
from dataclasses import replace
from typing import Any, Sequence
import torch

from ..config.schema import DatasetConfig, ModelConfig, OptimizationConfig, SamplingConfig
from ..data.loading import DatasetBundle
from ..sampling.base import CenterSampler
from ..training.loop import build_batch_subgraph, build_loss_fn, build_model
from .utils import build_center_sampler



def _flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            pieces.append(parameter.grad.detach().reshape(-1).cpu())
    if not pieces:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(pieces, dim=0)



def estimate_gradient_variance(
    dataset: DatasetBundle,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    optimization_config: OptimizationConfig,
    sampling_config: SamplingConfig,
    *,
    sampler_names: Sequence[str],
    batch_sizes: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    """Estimate gradient variance as a function of batch size.

    A fresh randomly initialized model is used for every sampled batch so that the
    variance reflects the sampling scheme rather than drift in trained weights.
    """

    results: list[dict[str, Any]] = []
    for sampler_name in sampler_names:
        for batch_size in batch_sizes:
            gradients: list[torch.Tensor] = []
            generator = torch.Generator(device=dataset.data.x.device)
            generator.manual_seed(int(seed) + int(batch_size) * 13 + len(sampler_name))
            sampling = replace(sampling_config, name=sampler_name, batch_size=int(batch_size))
            sampler = build_center_sampler(sampling, dataset, dataset.data.x.device)
            for _ in range(int(sampling_config.steps_per_epoch)):
                model = build_model(model_config, in_channels=dataset.num_features, out_channels=dataset.num_classes).to(dataset.data.x.device)
                loss_fn, _ = build_loss_fn(optimization_config, dataset, dataset.data.x.device)
                model.zero_grad(set_to_none=True)
                sample_out = sampler.sample(dataset.train_nodes, int(batch_size), generator=generator)
                subgraph = build_batch_subgraph(dataset.data, sample_out.center_nodes, sampling.num_hops, sample_out.metadata)
                logits = model(subgraph.x, subgraph.edge_index)
                center_logits = logits[subgraph.center_local_idx]
                center_target = subgraph.y[subgraph.center_local_idx]
                loss = loss_fn(center_logits, center_target)
                loss.backward()
                gradients.append(_flatten_gradients(model))

            gradient_matrix = torch.stack(gradients, dim=0)
            centered = gradient_matrix - gradient_matrix.mean(dim=0, keepdim=True)
            grad_norms = gradient_matrix.norm(dim=1)
            results.append(
                {
                    'dataset_name': dataset.name,
                    'feature_mode': dataset_config.feature_mode,
                    'sampler_name': sampler_name,
                    'batch_size': int(batch_size),
                    'num_batches': int(sampling_config.steps_per_epoch),
                    'gradient_variance_trace': float(centered.pow(2).mean(dim=0).sum().item()),
                    'mean_gradient_norm': float(grad_norms.mean().item()),
                    'std_gradient_norm': float(grad_norms.std().item()) if grad_norms.numel() > 1 else 0.0,
                }
            )
    return results
