from __future__ import annotations
from typing import Optional
import torch



def sample_next_node(
    current_node: int,
    neighbours: torch.Tensor,
    weights: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
) -> int:
    """Sample the next node in a random walk.

    Args:
        current_node: Current node id. Used only for error messages.
        neighbours: One-dimensional tensor of outgoing neighbour ids.
        weights: Optional non-negative edge weights aligned with neighbours.
        generator: Optional torch generator.

    Returns:
        The sampled next-node id.
    """

    if neighbours.numel() == 0:
        raise RuntimeError(f'Node {current_node} has no outgoing neighbours.')

    if weights is None:
        index = int(torch.randint(0, int(neighbours.numel()), (1,), generator=generator, device=neighbours.device).item())
        return int(neighbours[index].item())

    probabilities = weights.to(torch.float32)
    if torch.any(probabilities < 0):
        raise ValueError('Edge weights must be non-negative.')
    if float(probabilities.sum().item()) <= 0.0:
        index = int(torch.randint(0, int(neighbours.numel()), (1,), generator=generator, device=neighbours.device).item())
        return int(neighbours[index].item())

    index = int(torch.multinomial(probabilities, 1, replacement=True, generator=generator).item())
    return int(neighbours[index].item())
