from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class SamplerOutput:
    """Output of a center-selection sampler.

    Attributes:
        center_nodes: Sampled center nodes.
        metadata: Scalar sampler diagnostics recorded for logging and analysis.
    """

    center_nodes: torch.Tensor
    metadata: dict[str, float]


class CenterSampler(ABC):
    """Interface for selecting training center nodes from a candidate pool."""

    name: str

    @abstractmethod
    def sample(self, candidate_nodes: torch.Tensor, budget: int, *, generator: torch.Generator) -> SamplerOutput:
        """Sample center nodes from a candidate pool."""
