from __future__ import annotations
import time
from typing import Any
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ..config.schema import DatasetConfig
from ..data.loading import load_dataset
from ..metrics.classification import classify_from_probabilities


BASELINE_NAMES = (
    'baseline_majority_class',
    'baseline_random_class',
    'baseline_feature_only_linear',
)


def _metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        raise KeyError(f'Metric {metric_name} was not present in baseline metrics.')
    return float(value)


def _empirical_class_probabilities(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float32)
    return counts / torch.clamp(counts.sum(), min=1.0)


def _evaluate_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    latency: float,
) -> dict[str, Any]:
    return classify_from_probabilities(
        probabilities,
        labels,
        num_classes=num_classes,
        latency=latency,
    ).to_dict()


def _expand_probabilities(probabilities: np.ndarray, classes_: np.ndarray, num_classes: int) -> torch.Tensor:
    full = np.full((probabilities.shape[0], num_classes), 1e-12, dtype=np.float32)
    full[:, classes_.astype(np.int64)] = probabilities.astype(np.float32)
    return torch.from_numpy(full)


def _make_result(
    *,
    sampler_name: str,
    dataset,
    dataset_config: DatasetConfig,
    seed: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    total_training_time: float,
    monitor_metric: str,
) -> dict[str, Any]:
    return {
        'dataset_name': dataset.name,
        'dataset_kind': dataset.kind,
        'feature_mode': dataset_config.feature_mode,
        'sampler_name': sampler_name,
        'seed': int(seed),
        'requested_device': 'cpu',
        'resolved_device': 'cpu',
        'device_warnings': [],
        'evaluation_mode': 'feature_only_baseline',
        'evaluation_device': 'cpu',
        'evaluation_batch_size': int(dataset.test_nodes.numel()),
        'best_epoch': 0,
        'stopped_early': False,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_time_per_epoch': float(total_training_time),
        'total_training_time': float(total_training_time),
        'inference_latency': float(test_metrics.get('latency', 0.0)),
        'average_sampled_nodes': 0.0,
        'average_sampled_edges': 0.0,
        'average_center_nodes': 0.0,
        'budget_match_ratio': 0.0,
        'sampler_diagnostics': {},
        'checkpoint_path': '',
        'best_val_loss': float(val_metrics.get('loss', 0.0)),
        'best_monitor_value': _metric_value(val_metrics, monitor_metric),
        'monitor_metric': monitor_metric,
        'final_lr': 0.0,
        'epoch_history': [],
        'class_weights': None,
        'run_root': None,
        'resolved_config_path': None,
        'metrics_path': None,
        'history_path': None,
    }


def compute_trivial_baselines(
    dataset_config: DatasetConfig,
    *,
    seed: int,
    monitor_metric: str,
) -> list[dict[str, Any]]:
    """Compute majority, random, and feature-only linear baselines for one dataset split."""

    dataset = load_dataset(dataset_config, device=torch.device('cpu'), seed=seed)
    x = dataset.data.x.detach().cpu().numpy().astype(np.float32)
    y = dataset.data.y.detach().cpu().numpy().astype(np.int64)
    train_idx = dataset.train_nodes.detach().cpu().numpy().astype(np.int64)
    val_idx = dataset.val_nodes.detach().cpu().numpy().astype(np.int64)
    test_idx = dataset.test_nodes.detach().cpu().numpy().astype(np.int64)
    train_labels_t = dataset.data.y[dataset.train_nodes].detach().cpu()
    num_classes = int(dataset.num_classes)

    empirical_probs = _empirical_class_probabilities(train_labels_t, num_classes)
    majority_class = int(torch.argmax(empirical_probs).item())
    rng = np.random.default_rng(int(seed))

    results: list[dict[str, Any]] = []

    start = time.perf_counter()
    majority_template = torch.zeros((1, num_classes), dtype=torch.float32)
    majority_template[0, majority_class] = 1.0
    majority_train = majority_template.repeat(len(train_idx), 1)
    majority_val = majority_template.repeat(len(val_idx), 1)
    majority_test = majority_template.repeat(len(test_idx), 1)
    majority_total = time.perf_counter() - start
    results.append(
        _make_result(
            sampler_name='baseline_majority_class',
            dataset=dataset,
            dataset_config=dataset_config,
            seed=seed,
            train_metrics=_evaluate_probabilities(majority_train, dataset.data.y[dataset.train_nodes].cpu(), num_classes=num_classes, latency=0.0),
            val_metrics=_evaluate_probabilities(majority_val, dataset.data.y[dataset.val_nodes].cpu(), num_classes=num_classes, latency=0.0),
            test_metrics=_evaluate_probabilities(majority_test, dataset.data.y[dataset.test_nodes].cpu(), num_classes=num_classes, latency=0.0),
            total_training_time=majority_total,
            monitor_metric=monitor_metric,
        )
    )

    start = time.perf_counter()
    random_probs = empirical_probs.numpy()
    random_train_idx = rng.choice(num_classes, size=len(train_idx), p=random_probs)
    random_val_idx = rng.choice(num_classes, size=len(val_idx), p=random_probs)
    random_test_idx = rng.choice(num_classes, size=len(test_idx), p=random_probs)
    random_train = torch.from_numpy(np.eye(num_classes, dtype=np.float32)[random_train_idx])
    random_val = torch.from_numpy(np.eye(num_classes, dtype=np.float32)[random_val_idx])
    random_test = torch.from_numpy(np.eye(num_classes, dtype=np.float32)[random_test_idx])
    random_total = time.perf_counter() - start
    results.append(
        _make_result(
            sampler_name='baseline_random_class',
            dataset=dataset,
            dataset_config=dataset_config,
            seed=seed,
            train_metrics=_evaluate_probabilities(random_train, dataset.data.y[dataset.train_nodes].cpu(), num_classes=num_classes, latency=0.0),
            val_metrics=_evaluate_probabilities(random_val, dataset.data.y[dataset.val_nodes].cpu(), num_classes=num_classes, latency=0.0),
            test_metrics=_evaluate_probabilities(random_test, dataset.data.y[dataset.test_nodes].cpu(), num_classes=num_classes, latency=0.0),
            total_training_time=random_total,
            monitor_metric=monitor_metric,
        )
    )

    start = time.perf_counter()
    linear = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        random_state=int(seed),
    )
    linear.fit(x[train_idx], y[train_idx])
    fit_time = time.perf_counter() - start

    train_start = time.perf_counter()
    train_probs = _expand_probabilities(linear.predict_proba(x[train_idx]), linear.classes_, num_classes)
    train_latency = time.perf_counter() - train_start

    val_start = time.perf_counter()
    val_probs = _expand_probabilities(linear.predict_proba(x[val_idx]), linear.classes_, num_classes)
    val_latency = time.perf_counter() - val_start

    test_start = time.perf_counter()
    test_probs = _expand_probabilities(linear.predict_proba(x[test_idx]), linear.classes_, num_classes)
    test_latency = time.perf_counter() - test_start

    results.append(
        _make_result(
            sampler_name='baseline_feature_only_linear',
            dataset=dataset,
            dataset_config=dataset_config,
            seed=seed,
            train_metrics=_evaluate_probabilities(train_probs, dataset.data.y[dataset.train_nodes].cpu(), num_classes=num_classes, latency=train_latency),
            val_metrics=_evaluate_probabilities(val_probs, dataset.data.y[dataset.val_nodes].cpu(), num_classes=num_classes, latency=val_latency),
            test_metrics=_evaluate_probabilities(test_probs, dataset.data.y[dataset.test_nodes].cpu(), num_classes=num_classes, latency=test_latency),
            total_training_time=fit_time,
            monitor_metric=monitor_metric,
        )
    )

    return results
