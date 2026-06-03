from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass(slots=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    weighted_precision: float
    weighted_recall: float
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]
    latency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix,
            "latency": self.latency,
        }


def classify_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    loss_fn: torch.nn.Module,
    latency: float,
) -> ClassificationMetrics:
    """Compute classification metrics from logits and targets.

    This version is robust to mixed-device situations where logits / target
    are on CPU but loss_fn holds class weights on CUDA.
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2, got shape={tuple(logits.shape)}")
    if target.ndim != 1:
        raise ValueError(f"target must be rank-1, got shape={tuple(target.shape)}")
    if logits.size(0) != target.size(0):
        raise ValueError(
            f"logits and target must agree on batch dimension, got "
            f"{logits.size(0)} and {target.size(0)}"
        )

    prediction = logits.argmax(dim=-1)

    weight = getattr(loss_fn, "weight", None)
    if isinstance(weight, torch.Tensor):
        loss_device = weight.device
        loss_logits = logits.to(loss_device)
        loss_target = target.to(loss_device)
    else:
        loss_logits = logits
        loss_target = target

    loss = float(loss_fn(loss_logits, loss_target).item())

    target_np = target.detach().cpu().numpy()
    prediction_np = prediction.detach().cpu().numpy()

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        target_np,
        prediction_np,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        target_np,
        prediction_np,
        average="weighted",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        target_np,
        prediction_np,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    conf = confusion_matrix(
        target_np,
        prediction_np,
        labels=list(range(num_classes)),
    )

    return ClassificationMetrics(
        loss=loss,
        accuracy=float(accuracy_score(target_np, prediction_np)),
        macro_f1=float(macro_f1),
        balanced_accuracy=float(balanced_accuracy_score(target_np, prediction_np)),
        weighted_f1=float(weighted_f1),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        weighted_precision=float(weighted_precision),
        weighted_recall=float(weighted_recall),
        per_class_precision=[float(x) for x in np.asarray(per_class_precision)],
        per_class_recall=[float(x) for x in np.asarray(per_class_recall)],
        per_class_f1=[float(x) for x in np.asarray(per_class_f1)],
        confusion_matrix=conf.astype(int).tolist(),
        latency=float(latency),
    )


def classify_from_probabilities(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    latency: float,
) -> ClassificationMetrics:
    """Compute classification metrics from class probabilities."""

    probs = probabilities.to(torch.float32)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be rank-2, got shape={tuple(probs.shape)}")
    probs = torch.clamp(probs, min=1e-12)
    probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-12)
    logits = torch.log(probs)
    loss_fn = torch.nn.CrossEntropyLoss()
    return classify_from_logits(
        logits,
        target,
        num_classes=num_classes,
        loss_fn=loss_fn,
        latency=latency,
    )


def monitor_value(metrics: ClassificationMetrics, metric_name: str) -> float:
    """Return the metric used for learning-rate scheduling and early stopping."""

    if metric_name == 'accuracy':
        return metrics.accuracy
    if metric_name == 'macro_f1':
        return metrics.macro_f1
    if metric_name == 'balanced_accuracy':
        return metrics.balanced_accuracy
    if metric_name == 'weighted_f1':
        return metrics.weighted_f1
    raise ValueError(f'Unsupported monitor metric: {metric_name}')
