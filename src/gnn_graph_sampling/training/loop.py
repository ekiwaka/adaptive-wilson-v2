from __future__ import annotations
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from ..config.schema import ModelConfig, OptimizationConfig, SamplingConfig
from ..data.loading import DatasetBundle
from ..metrics.classification import ClassificationMetrics, classify_from_logits
from ..models.gcn import GCNClassifier
from ..models.graphsage import GraphSageClassifier
from ..models.mixhop import MixHopClassifier
from ..sampling.base import CenterSampler
from ..sampling.online_weights import compute_online_sample_weights
from .preprocessing import resolve_budget_count


@dataclass(slots=True)
class BatchStats:
    """Structural statistics for one sampled training batch."""

    sampled_nodes: int
    sampled_edges: int
    center_nodes: int
    sampler_metadata: dict[str, float]


@dataclass(slots=True)
class BatchSubgraph:
    """Induced k-hop training batch."""

    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    center_local_idx: torch.Tensor
    node_ids: torch.Tensor
    stats: BatchStats


@dataclass(slots=True)
class RunResult:
    """Output of a single training run."""

    seed: int
    dataset_name: str
    dataset_kind: str
    feature_mode: str
    model_family: str
    sampler_name: str
    sampling_mode: str
    online_schedule: str
    budget_unit: str
    budget_value: Optional[float]
    split_mode: str
    training_objective: str
    resolved_budget: int
    total_nodes: int
    train_node_count: int
    val_node_count: int
    test_node_count: int
    split_val_ratio_of_remaining: Optional[float]
    preprocessed_cache_path: Optional[str]
    requested_device: str
    resolved_device: str
    device_warnings: list[str]
    evaluation_mode: str
    evaluation_device: str
    evaluation_batch_size: int
    best_epoch: int
    best_cumulative_minibatch_updates: int
    best_cumulative_optimizer_updates: int
    first_epoch_to_95pct_best_monitor: Optional[int]
    first_cumulative_minibatch_updates_to_95pct_best_monitor: Optional[int]
    total_minibatch_updates: int
    total_optimizer_updates: int
    stopped_early: bool
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    training_time_per_epoch: float
    total_training_time: float
    inference_latency: float
    average_sampled_nodes: float
    average_sampled_edges: float
    average_center_nodes: float
    budget_match_ratio: float
    sampler_diagnostics: dict[str, float]
    checkpoint_path: str
    best_val_loss: float
    best_monitor_value: float
    monitor_metric: str
    final_lr: float
    epoch_history: list[dict[str, Any]]
    class_weights: Optional[list[float]]
    split_metadata: Optional[dict[str, Any]] = None
    run_root: Optional[str] = None
    resolved_config_path: Optional[str] = None
    metrics_path: Optional[str] = None
    history_path: Optional[str] = None
    batch_history_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EarlyStopper:
    """Simple patience-based early stopping helper."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = float('-inf')
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = float(score)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience



def build_model(model_config: ModelConfig, *, in_channels: int, out_channels: int) -> nn.Module:
    """Construct the configured message-passing backbone."""

    common_kwargs = dict(
        in_channels=in_channels,
        hidden_channels=model_config.hidden_dim,
        out_channels=out_channels,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        input_dropout=model_config.input_dropout,
        use_batch_norm=model_config.use_batch_norm,
        norm_type=model_config.norm_type,
        residual=model_config.residual,
        jk_mode=model_config.jk_mode,
        head_type=model_config.head_type,
        head_hidden_dim=model_config.head_hidden_dim,
    )

    if model_config.family == 'graphsage':
        return GraphSageClassifier(aggr=model_config.aggr, **common_kwargs)

    if model_config.family == 'gcn':
        return GCNClassifier(**common_kwargs)

    if model_config.family == 'mixhop':
        return MixHopClassifier(**common_kwargs)

    raise ValueError(f'Unsupported model family: {model_config.family}')



def build_batch_subgraph(data: Any, center_nodes: torch.Tensor, num_hops: int, metadata: dict[str, float]) -> BatchSubgraph:
    """Extract a k-hop induced subgraph around the chosen center nodes."""

    if not hasattr(data, '_cached_out_degree'):
        setattr(
            data,
            '_cached_out_degree',
            torch.bincount(data.edge_index[0].to(torch.long), minlength=int(data.num_nodes)).to(torch.float32),
        )
    if not hasattr(data, '_cached_num_classes'):
        max_label = int(torch.max(data.y).item()) if data.y.numel() else -1
        setattr(data, '_cached_num_classes', max_label + 1)

    out_degree = getattr(data, '_cached_out_degree')
    num_classes = max(int(getattr(data, '_cached_num_classes')), 1)

    center_nodes_cpu = center_nodes.detach().cpu().to(torch.long)
    center_degrees = out_degree[center_nodes_cpu]
    global_mean_degree = float(out_degree.mean().item()) if out_degree.numel() else 0.0
    mean_center_degree = float(center_degrees.mean().item()) if center_degrees.numel() else 0.0
    center_degree_ratio_vs_global = mean_center_degree / max(global_mean_degree, 1e-12)

    center_labels = data.y.detach().cpu().to(torch.long)[center_nodes_cpu]
    class_hist = torch.bincount(center_labels, minlength=num_classes).to(torch.float32)
    represented_classes = int((class_hist > 0).sum().item())
    coverage_ratio = represented_classes / max(num_classes, 1)
    label_mass = class_hist.sum()
    if float(label_mass.item()) > 0.0:
        probs = class_hist[class_hist > 0] / label_mass
        center_label_entropy_bits = float((-(probs * torch.log2(probs))).sum().item())
    else:
        center_label_entropy_bits = 0.0
    max_entropy = 0.0 if num_classes <= 1 else float(torch.log2(torch.tensor(float(num_classes))).item())
    center_label_entropy_ratio = center_label_entropy_bits / max(max_entropy, 1e-12) if max_entropy > 0.0 else 0.0

    enriched_metadata = {str(key): float(value) for key, value in metadata.items()}
    enriched_metadata.update(
        {
            'mean_center_degree': float(mean_center_degree),
            'center_degree_ratio_vs_global': float(center_degree_ratio_vs_global),
            'center_class_coverage': float(represented_classes),
            'center_class_coverage_ratio': float(coverage_ratio),
            'center_label_entropy_bits': float(center_label_entropy_bits),
            'center_label_entropy_ratio': float(center_label_entropy_ratio),
        }
    )

    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=center_nodes,
        num_hops=int(num_hops),
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=int(data.num_nodes),
    )
    return BatchSubgraph(
        x=data.x[subset],
        edge_index=sub_edge_index,
        y=data.y[subset],
        center_local_idx=mapping,
        node_ids=subset,
        stats=BatchStats(
            sampled_nodes=int(subset.numel()),
            sampled_edges=int(sub_edge_index.size(1)),
            center_nodes=int(center_nodes.numel()),
            sampler_metadata=enriched_metadata,
        ),
    )



def move_batch_subgraph_to_device(subgraph: BatchSubgraph, device: torch.device, *, non_blocking: bool = True) -> BatchSubgraph:
    return BatchSubgraph(
        x=subgraph.x.to(device, non_blocking=non_blocking),
        edge_index=subgraph.edge_index.to(device, non_blocking=non_blocking),
        y=subgraph.y.to(device, non_blocking=non_blocking),
        center_local_idx=subgraph.center_local_idx.to(device, non_blocking=non_blocking),
        node_ids=subgraph.node_ids.to(device, non_blocking=non_blocking),
        stats=subgraph.stats,
    )



def compute_class_weights(train_labels: torch.Tensor, num_classes: int, device: torch.device, *, mode: str, beta: float) -> torch.Tensor:
    counts = torch.bincount(train_labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    if mode == 'sqrt_inv':
        weights = counts.sum() / (float(num_classes) * counts)
        weights = torch.sqrt(weights)
    elif mode == 'effective_num':
        beta_tensor = torch.tensor(beta, device=counts.device, dtype=counts.dtype)
        effective_num = 1.0 - torch.pow(beta_tensor, counts)
        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
    else:
        raise ValueError(f'Unsupported class-weight mode: {mode}')
    weights = weights / torch.clamp(weights.mean(), min=1e-12)
    return weights.to(device)



def should_disable_class_weights_for_dataset(optimization: OptimizationConfig, dataset: DatasetBundle) -> bool:
    if not optimization.use_class_weights:
        return True

    auto_disable = optimization.auto_disable_class_weights_for_small_balanced_splits
    if auto_disable is None:
        auto_disable = optimization.auto_disable_class_weights_for_tiny_balanced_citation_splits
    if not auto_disable:
        return False

    train_labels = dataset.data.y[dataset.train_nodes]
    if int(train_labels.numel()) > int(optimization.class_weight_auto_disable_max_train_nodes):
        return False
    counts = torch.bincount(train_labels, minlength=dataset.num_classes).float()
    positive = counts[counts > 0]
    if positive.numel() == 0:
        return False
    imbalance_ratio = float(positive.max().item() / max(positive.min().item(), 1.0))
    return imbalance_ratio <= float(optimization.class_weight_auto_disable_max_imbalance_ratio)



def build_loss_fn(optimization: OptimizationConfig, dataset: DatasetBundle, device: torch.device) -> tuple[torch.nn.Module, Optional[torch.Tensor]]:
    weight = None
    if not should_disable_class_weights_for_dataset(optimization, dataset):
        weight = compute_class_weights(
            dataset.data.y[dataset.train_nodes],
            num_classes=dataset.num_classes,
            device=device,
            mode=optimization.class_weight_mode,
            beta=optimization.class_weight_beta,
        )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=optimization.label_smoothing)
    return loss_fn, weight



def build_optimizer(model: torch.nn.Module, optimization: OptimizationConfig) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith('.bias') or 'norm' in name.lower():
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': optimization.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=optimization.lr,
    )



def build_scheduler(optimizer: torch.optim.Optimizer, optimization: OptimizationConfig) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=optimization.scheduler_factor,
        patience=optimization.scheduler_patience,
        min_lr=optimization.scheduler_min_lr,
    )


@torch.no_grad()
def evaluate_full_graph(model: torch.nn.Module, dataset: DatasetBundle, node_ids: torch.Tensor, *, loss_fn: torch.nn.Module) -> ClassificationMetrics:
    model.eval()
    start = time.perf_counter()
    logits = model(dataset.data.x, dataset.data.edge_index)
    latency = time.perf_counter() - start
    masked_logits = logits[node_ids]
    target = dataset.data.y[node_ids]
    return classify_from_logits(masked_logits, target, num_classes=dataset.num_classes, loss_fn=loss_fn, latency=latency)


@torch.no_grad()
def evaluate_exact_batched(
    model: torch.nn.Module,
    dataset: DatasetBundle,
    node_ids: torch.Tensor,
    *,
    loss_fn: torch.nn.Module,
    eval_device: torch.device,
    batch_size: int,
    non_blocking: bool = True,
) -> ClassificationMetrics:
    model.eval()
    start = time.perf_counter()
    num_hops = int(getattr(model, 'num_layers', 1))

    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []

    for center_nodes in node_ids.split(int(batch_size)):
        subgraph = build_batch_subgraph(dataset.data, center_nodes, num_hops=num_hops, metadata={})
        subgraph = move_batch_subgraph_to_device(subgraph, eval_device, non_blocking=non_blocking)
        logits = model(subgraph.x, subgraph.edge_index)
        center_logits = logits[subgraph.center_local_idx]

        logits_chunks.append(center_logits.cpu())
        target_chunks.append(subgraph.y[subgraph.center_local_idx].cpu())

    latency = time.perf_counter() - start
    all_logits = torch.cat(logits_chunks, dim=0)
    all_targets = torch.cat(target_chunks, dim=0)

    return classify_from_logits(all_logits, all_targets, num_classes=dataset.num_classes, loss_fn=loss_fn, latency=latency)


@torch.no_grad()
def evaluate_with_sampling(
    model: torch.nn.Module,
    dataset: DatasetBundle,
    node_ids: torch.Tensor,
    sampler: CenterSampler,
    sampling: SamplingConfig,
    generator: torch.Generator,
    *,
    loss_fn: torch.nn.Module,
) -> ClassificationMetrics:
    model.eval()
    start = time.perf_counter()
    predictions = torch.full((int(dataset.data.num_nodes),), -1, dtype=torch.long, device=dataset.data.x.device)
    remaining = node_ids.clone()
    batches = 0
    eval_budget = resolve_budget_count(sampling, int(node_ids.numel())) if sampling.budget_value is not None else int(sampling.eval_batch_size)
    while remaining.numel() > 0 and batches < int(sampling.inference_max_batches):
        batches += 1
        sample_out = sampler.sample(remaining, min(eval_budget, int(sampling.eval_batch_size)), generator=generator)
        subgraph = build_batch_subgraph(dataset.data, sample_out.center_nodes, sampling.num_hops, sample_out.metadata)
        logits = model(subgraph.x, subgraph.edge_index)
        predictions[sample_out.center_nodes] = logits[subgraph.center_local_idx].argmax(dim=-1)
        remaining = remaining[predictions[remaining] < 0]

    if remaining.numel() > 0:
        full_logits = model(dataset.data.x, dataset.data.edge_index)
        predictions[remaining] = full_logits[remaining].argmax(dim=-1)

    latency = time.perf_counter() - start
    full_logits_for_loss = model(dataset.data.x, dataset.data.edge_index)[node_ids]
    return classify_from_logits(full_logits_for_loss, dataset.data.y[node_ids], num_classes=dataset.num_classes, loss_fn=loss_fn, latency=latency)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: DatasetBundle,
    node_ids: torch.Tensor,
    *,
    loss_fn: torch.nn.Module,
    evaluation_mode: str,
    evaluation_device: torch.device,
    evaluation_batch_size: int,
    non_blocking: bool = True,
) -> ClassificationMetrics:
    if evaluation_mode == 'full_graph':
        return evaluate_full_graph(model, dataset, node_ids, loss_fn=loss_fn)
    if evaluation_mode == 'exact_batched':
        return evaluate_exact_batched(
            model,
            dataset,
            node_ids,
            loss_fn=loss_fn,
            eval_device=evaluation_device,
            batch_size=evaluation_batch_size,
            non_blocking=non_blocking,
        )
    raise ValueError(f'Unsupported evaluation mode: {evaluation_mode}')



def _step_optimizer(
    *,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
    scaler: Optional[torch.amp.GradScaler],
    amp_enabled: bool,
    accumulation_divisor: int = 1,
) -> None:
    if scaler is not None and amp_enabled:
        scaler.scale(loss / accumulation_divisor).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    else:
        (loss / accumulation_divisor).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad],
            max_norm=grad_clip_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)



def train_one_epoch(
    model: torch.nn.Module,
    dataset: DatasetBundle,
    sampler: CenterSampler,
    sampling: SamplingConfig,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    generator: torch.Generator,
    *,
    device: torch.device,
    grad_clip_norm: float,
    accumulation_steps: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_enabled: bool = False,
    use_normalized_weights: bool = False,
    weight_eps: float = 1e-12,
) -> tuple[float, list[BatchStats], dict[str, Any], list[dict[str, Any]]]:
    """Train the model for one epoch using sampled k-hop batches (online per-batch)."""

    model.train()
    accumulation_steps = max(1, int(accumulation_steps))
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    batch_stats: list[BatchStats] = []
    batch_history: list[dict[str, Any]] = []
    unique_center_nodes: set[int] = set()
    optimizer_updates = 0

    num_hops = int(getattr(model, 'num_layers', sampling.num_hops))
    budget = resolve_budget_count(sampling, int(dataset.train_nodes.numel()))

    for step_idx in range(int(sampling.steps_per_epoch)):
        sample_out = sampler.sample(dataset.train_nodes, budget, generator=generator)
        center_nodes_cpu_for_stats = sample_out.center_nodes.detach().cpu().to(torch.long)
        unique_center_nodes.update(int(value) for value in center_nodes_cpu_for_stats.tolist())
        subgraph = build_batch_subgraph(dataset.data, sample_out.center_nodes, num_hops, sample_out.metadata)
        online_sample_weight = None
        if use_normalized_weights:
            online_weights = compute_online_sample_weights(
                sampler_name=sampling.name,
                edge_index=dataset.data.edge_index,
                num_nodes=int(dataset.data.num_nodes),
                center_nodes=sample_out.center_nodes,
                candidate_nodes=dataset.train_nodes,
                sample_metadata=sample_out.metadata,
                eps=weight_eps,
                budget=budget,
                sampler=sampler,
            )
            online_sample_weight = online_weights.weights
            subgraph.stats.sampler_metadata.update(
                {
                    'online_normalized_weighting_enabled': 1.0,
                    'online_weight_min': float(online_weights.weights.min().item()),
                    'online_weight_max': float(online_weights.weights.max().item()),
                    'online_weight_mean': float(online_weights.weights.mean().item()),
                    'online_weight_sum': float(online_weights.weights.sum().item()),
                    'online_weight_method': online_weights.method,
                    **{f'online_{key}': value for key, value in online_weights.metadata.items()},
                }
            )
        else:
            subgraph.stats.sampler_metadata['online_normalized_weighting_enabled'] = 0.0
        batch_stats.append(subgraph.stats)
        subgraph = move_batch_subgraph_to_device(subgraph, device, non_blocking=True)
        if online_sample_weight is not None:
            online_sample_weight = online_sample_weight.to(device=device, dtype=torch.float32)

        if scaler is not None and amp_enabled:
            with torch.amp.autocast('cuda', enabled=True):
                logits = model(subgraph.x, subgraph.edge_index)
                center_logits = logits[subgraph.center_local_idx]
                center_target = subgraph.y[subgraph.center_local_idx]
                loss = compute_positional_weighted_cross_entropy(
                    center_logits,
                    center_target,
                    loss_fn,
                    online_sample_weight,
                    eps=weight_eps,
                )
            scaler.scale(loss / accumulation_steps).backward()
            should_step = ((step_idx + 1) % accumulation_steps == 0) or (step_idx + 1 == int(sampling.steps_per_epoch))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
        else:
            logits = model(subgraph.x, subgraph.edge_index)
            center_logits = logits[subgraph.center_local_idx]
            center_target = subgraph.y[subgraph.center_local_idx]
            loss = compute_positional_weighted_cross_entropy(
                center_logits,
                center_target,
                loss_fn,
                online_sample_weight,
                eps=weight_eps,
            )
            (loss / accumulation_steps).backward()
            should_step = ((step_idx + 1) % accumulation_steps == 0) or (step_idx + 1 == int(sampling.steps_per_epoch))
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1

        current_loss = float(loss.item())
        losses.append(current_loss)
        batch_stats[-1].sampler_metadata['batch_loss'] = current_loss
        batch_history.append(
            {
                'step_in_epoch': int(step_idx + 1),
                'optimizer_update_after_batch': int(should_step),
                'batch_loss': current_loss,
                'center_nodes': int(subgraph.stats.center_nodes),
                'sampled_nodes': int(subgraph.stats.sampled_nodes),
                'sampled_edges': int(subgraph.stats.sampled_edges),
                'requested_budget': float(sample_out.metadata.get('requested_budget', budget)),
                'realized_budget': float(sample_out.metadata.get('realized_budget', subgraph.stats.center_nodes)),
                'online_normalized_weighting_enabled': float(
                    batch_stats[-1].sampler_metadata.get('online_normalized_weighting_enabled', 0.0)
                ),
                'online_weight_min': batch_stats[-1].sampler_metadata.get('online_weight_min'),
                'online_weight_max': batch_stats[-1].sampler_metadata.get('online_weight_max'),
                'online_weight_mean': batch_stats[-1].sampler_metadata.get('online_weight_mean'),
                'online_weight_sum': batch_stats[-1].sampler_metadata.get('online_weight_sum'),
                'online_weight_method': batch_stats[-1].sampler_metadata.get('online_weight_method'),
                'online_min_inclusion_probability': batch_stats[-1].sampler_metadata.get('online_min_inclusion_probability'),
                'online_max_inclusion_probability': batch_stats[-1].sampler_metadata.get('online_max_inclusion_probability'),
                'online_mean_inclusion_probability': batch_stats[-1].sampler_metadata.get('online_mean_inclusion_probability'),
            }
        )

    mean_loss = float(sum(losses) / max(len(losses), 1))
    loss_variance = float(sum((value - mean_loss) ** 2 for value in losses) / max(len(losses), 1))
    loss_std = float(loss_variance ** 0.5)
    residuals = [float(value - mean_loss) for value in losses]
    residual_mean = float(sum(residuals) / max(len(residuals), 1))
    residual_squared_mean = float(sum(value * value for value in residuals) / max(len(residuals), 1))

    for stat, batch_row in zip(batch_stats, batch_history):
        residual = float(stat.sampler_metadata.get('batch_loss', mean_loss) - mean_loss)
        stat.sampler_metadata['expected_minibatch_loss'] = mean_loss
        stat.sampler_metadata['epoch_batch_loss_mean'] = mean_loss
        stat.sampler_metadata['epoch_batch_loss_variance'] = loss_variance
        stat.sampler_metadata['epoch_batch_loss_std'] = loss_std
        stat.sampler_metadata['minibatch_loss_minus_expectation'] = residual
        stat.sampler_metadata['minibatch_loss_minus_epoch_mean'] = residual
        stat.sampler_metadata['minibatch_loss_minus_expectation_squared'] = residual * residual
        batch_row['expected_minibatch_loss'] = mean_loss
        batch_row['epoch_batch_loss_mean'] = mean_loss
        batch_row['epoch_batch_loss_variance'] = loss_variance
        batch_row['epoch_batch_loss_std'] = loss_std
        batch_row['minibatch_loss_minus_expectation'] = residual
        batch_row['minibatch_loss_minus_expectation_squared'] = residual * residual

    train_count = max(int(dataset.train_nodes.numel()), 1)
    diagnostics = {
        'observed_minibatches': float(len(losses)),
        'optimizer_updates': float(optimizer_updates),
        'expected_minibatch_loss': mean_loss,
        'epoch_batch_loss_mean': mean_loss,
        'epoch_batch_loss_variance': loss_variance,
        'epoch_batch_loss_std': loss_std,
        'minibatch_loss_minus_expectation_mean': residual_mean,
        'minibatch_loss_minus_expectation_squared_mean': residual_squared_mean,
        'unique_center_nodes_epoch': float(len(unique_center_nodes)),
        'unique_center_node_coverage_epoch': float(len(unique_center_nodes) / train_count),
        'total_center_draws_epoch': float(sum(stat.center_nodes for stat in batch_stats)),
        'unique_center_node_ids': sorted(int(value) for value in unique_center_nodes),
    }

    return mean_loss, batch_stats, diagnostics, batch_history



def train_one_epoch_full_graph_semisupervised(
    model: torch.nn.Module,
    dataset: DatasetBundle,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    *,
    device: torch.device,
    grad_clip_norm: float,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_enabled: bool = False,
) -> tuple[float, list[BatchStats], dict[str, Any], list[dict[str, Any]]]:
    """Train with a full-graph forward pass and a train-mask-only loss.

    If dataset.data.train_sample_weight exists, the loss becomes:

        sum_i w_i CE_i / sum_i w_i

    where the weights are position-aligned with dataset.train_nodes.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    x = dataset.data.x.to(device)
    edge_index = dataset.data.edge_index.to(device)
    y = dataset.data.y.to(device)
    train_nodes = dataset.train_nodes.to(device)

    train_sample_weight = getattr(dataset.data, "train_sample_weight", None)

    if train_sample_weight is not None:
        if train_sample_weight.numel() != dataset.train_nodes.numel():
            raise ValueError(
                "dataset.data.train_sample_weight must be position-aligned with "
                "dataset.train_nodes. "
                f"Got {train_sample_weight.numel()} weights for "
                f"{dataset.train_nodes.numel()} train nodes."
            )

        train_sample_weight = train_sample_weight.to(
            device=device,
            dtype=torch.float32,
        )

    if scaler is not None and amp_enabled:
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(x, edge_index)
            loss = compute_positional_weighted_cross_entropy(
                logits=logits[train_nodes],
                target=y[train_nodes],
                loss_fn=loss_fn,
                sample_weight=train_sample_weight,
                eps=1e-12,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    else:
        logits = model(x, edge_index)
        loss = compute_positional_weighted_cross_entropy(
            logits=logits[train_nodes],
            target=y[train_nodes],
            loss_fn=loss_fn,
            sample_weight=train_sample_weight,
            eps=1e-12,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    stats_metadata = {
        'full_graph_training': 1.0,
        'split_train_fraction': float(dataset.train_nodes.numel())
        / max(int(dataset.data.num_nodes), 1),
        'split_val_fraction': float(dataset.val_nodes.numel())
        / max(int(dataset.data.num_nodes), 1),
        'split_test_fraction': float(dataset.test_nodes.numel())
        / max(int(dataset.data.num_nodes), 1),
        'epoch_batch_loss_variance': 0.0,
        'minibatch_loss_minus_epoch_mean': 0.0,
        **{
            str(k): float(v)
            for k, v in dataset.metadata.items()
            if isinstance(v, (int, float, bool))
        },
    }

    if train_sample_weight is not None:
        stats_metadata.update(
            {
                'used_train_sample_weight': 1.0,
                'train_sample_weight_min': float(train_sample_weight.min().item()),
                'train_sample_weight_max': float(train_sample_weight.max().item()),
                'train_sample_weight_mean': float(train_sample_weight.mean().item()),
            }
        )
    else:
        stats_metadata['used_train_sample_weight'] = 0.0

    stats = BatchStats(
        sampled_nodes=int(dataset.data.num_nodes),
        sampled_edges=int(dataset.data.edge_index.size(1)),
        center_nodes=int(dataset.train_nodes.numel()),
        sampler_metadata=stats_metadata,
    )

    loss_value = float(loss.item())
    diagnostics = {
        'observed_minibatches': 1.0,
        'optimizer_updates': 1.0,
        'expected_minibatch_loss': loss_value,
        'epoch_batch_loss_mean': loss_value,
        'epoch_batch_loss_variance': 0.0,
        'epoch_batch_loss_std': 0.0,
        'minibatch_loss_minus_expectation_mean': 0.0,
        'minibatch_loss_minus_expectation_squared_mean': 0.0,
        'unique_center_nodes_epoch': float(dataset.train_nodes.numel()),
        'unique_center_node_coverage_epoch': 1.0,
        'total_center_draws_epoch': float(dataset.train_nodes.numel()),
        'unique_center_node_ids': [int(value) for value in dataset.train_nodes.detach().cpu().to(torch.long).tolist()],
    }
    batch_history = [
        {
            'step_in_epoch': 1,
            'optimizer_update_after_batch': 1,
            'batch_loss': loss_value,
            'expected_minibatch_loss': loss_value,
            'epoch_batch_loss_mean': loss_value,
            'epoch_batch_loss_variance': 0.0,
            'epoch_batch_loss_std': 0.0,
            'minibatch_loss_minus_expectation': 0.0,
            'minibatch_loss_minus_expectation_squared': 0.0,
            'center_nodes': int(dataset.train_nodes.numel()),
            'sampled_nodes': int(dataset.data.num_nodes),
            'sampled_edges': int(dataset.data.edge_index.size(1)),
            'online_normalized_weighting_enabled': 0.0,
        }
    ]

    return loss_value, [stats], diagnostics, batch_history


def train_one_epoch_on_reduced_graph(
    model: torch.nn.Module,
    reduced_dataset: DatasetBundle,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    *,
    device: torch.device,
    grad_clip_norm: float,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_enabled: bool = False,
    use_normalized_weights: bool = False,
    weight_eps: float = 1e-12,
) -> tuple[float, list[BatchStats], dict[str, Any], list[dict[str, Any]]]:
    """Train with one full-batch update on a preprocessed/reduced graph."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    x = reduced_dataset.data.x.to(device)
    edge_index = reduced_dataset.data.edge_index.to(device)
    y = reduced_dataset.data.y.to(device)
    train_nodes = reduced_dataset.train_nodes.to(device)
    train_sample_weight = getattr(reduced_dataset.data, 'train_sample_weight', None) if use_normalized_weights else None
    if train_sample_weight is not None:
        train_sample_weight = train_sample_weight.to(device=device, dtype=torch.float32)

    if scaler is not None and amp_enabled:
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(x, edge_index)
            loss = compute_positional_weighted_cross_entropy(
                logits[train_nodes],
                y[train_nodes],
                loss_fn,
                train_sample_weight,
                eps=weight_eps,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    else:
        logits = model(x, edge_index)
        loss = compute_positional_weighted_cross_entropy(
            logits[train_nodes],
            y[train_nodes],
            loss_fn,
            train_sample_weight,
            eps=weight_eps,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


    stats = BatchStats(
        sampled_nodes=int(reduced_dataset.data.num_nodes),
        sampled_edges=int(reduced_dataset.data.edge_index.size(1)),
        center_nodes=int(reduced_dataset.train_nodes.numel()),
        sampler_metadata={
            'preprocessed_graph': 1.0,
            'used_train_sample_weight': float(train_sample_weight is not None),
            'epoch_batch_loss_variance': 0.0,
            'minibatch_loss_minus_epoch_mean': 0.0,
            'reduced_train_fraction': float(reduced_dataset.train_nodes.numel()) / max(int(reduced_dataset.data.num_nodes), 1),
            **{str(k): float(v) for k, v in reduced_dataset.metadata.items() if isinstance(v, (int, float, bool))},
        },
    )
    loss_value = float(loss.item())
    diagnostics = {
        'observed_minibatches': 1.0,
        'optimizer_updates': 1.0,
        'expected_minibatch_loss': loss_value,
        'epoch_batch_loss_mean': loss_value,
        'epoch_batch_loss_variance': 0.0,
        'epoch_batch_loss_std': 0.0,
        'minibatch_loss_minus_expectation_mean': 0.0,
        'minibatch_loss_minus_expectation_squared_mean': 0.0,
        'unique_center_nodes_epoch': float(reduced_dataset.train_nodes.numel()),
        'unique_center_node_coverage_epoch': 1.0,
        'total_center_draws_epoch': float(reduced_dataset.train_nodes.numel()),
        'unique_center_node_ids': [int(value) for value in reduced_dataset.train_nodes.detach().cpu().to(torch.long).tolist()],
    }
    batch_history = [
        {
            'step_in_epoch': 1,
            'optimizer_update_after_batch': 1,
            'batch_loss': loss_value,
            'expected_minibatch_loss': loss_value,
            'epoch_batch_loss_mean': loss_value,
            'epoch_batch_loss_variance': 0.0,
            'epoch_batch_loss_std': 0.0,
            'minibatch_loss_minus_expectation': 0.0,
            'minibatch_loss_minus_expectation_squared': 0.0,
            'center_nodes': int(reduced_dataset.train_nodes.numel()),
            'sampled_nodes': int(reduced_dataset.data.num_nodes),
            'sampled_edges': int(reduced_dataset.data.edge_index.size(1)),
            'online_normalized_weighting_enabled': float(train_sample_weight is not None),
        }
    ]

    return loss_value, [stats], diagnostics, batch_history



def aggregate_sampler_metadata(batch_stats: list[BatchStats]) -> dict[str, float]:
    collector: dict[str, list[float]] = {}
    for stat in batch_stats:
        for key, value in stat.sampler_metadata.items():
            if isinstance(value, bool):
                numeric_value = float(value)
            elif isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                continue
            collector.setdefault(key, []).append(numeric_value)
    summary: dict[str, float] = {}
    for key, values in collector.items():
        summary[f'mean_{key}'] = float(sum(values) / max(len(values), 1))
        if len(values) > 1:
            mean_value = summary[f'mean_{key}']
            summary[f'std_{key}'] = float((sum((value - mean_value) ** 2 for value in values) / len(values)) ** 0.5)
        else:
            summary[f'std_{key}'] = 0.0
    return summary


def compute_positional_weighted_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_fn: torch.nn.Module,
    sample_weight: torch.Tensor | None,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    if sample_weight is None:
        return loss_fn(logits, target)

    sample_weight = sample_weight.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    if sample_weight.ndim != 1:
        raise ValueError(
            f"sample_weight must be 1D, got shape {tuple(sample_weight.shape)}."
        )

    if sample_weight.numel() != target.numel():
        raise ValueError(
            f"sample_weight length {sample_weight.numel()} does not match "
            f"target length {target.numel()}."
        )

    if not torch.isfinite(sample_weight).all():
        raise ValueError("sample_weight contains non-finite values.")

    if torch.allclose(sample_weight, torch.ones_like(sample_weight)):
        return loss_fn(logits, target)

    class_weight = getattr(loss_fn, "weight", None)
    label_smoothing = float(getattr(loss_fn, "label_smoothing", 0.0))

    per_node_loss = F.cross_entropy(
        logits,
        target,
        weight=class_weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )

    denominator = sample_weight.sum().clamp_min(eps)
    return (sample_weight * per_node_loss).sum() / denominator
