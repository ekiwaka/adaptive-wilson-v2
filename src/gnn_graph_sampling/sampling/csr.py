from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(slots=True)
class CSRGraph:
    """Compressed sparse row adjacency structure."""

    num_nodes: int
    rowptr: torch.Tensor
    col: torch.Tensor
    weights: Optional[torch.Tensor]
    outgoing_mass: torch.Tensor

    @classmethod
    def from_edge_index(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        *,
        edge_weight: Optional[torch.Tensor] = None,
        make_undirected: bool = False,
        drop_self_loops: bool = False,
        device: Optional[torch.device] = None,
    ) -> 'CSRGraph':
        """Build a CSR graph from a COO edge list."""

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError('edge_index must have shape [2, num_edges].')
        device = device or edge_index.device
        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)
        weights = edge_weight.to(device=device) if edge_weight is not None else None

        if drop_self_loops:
            mask = src != dst
            src, dst = src[mask], dst[mask]
            if weights is not None:
                weights = weights[mask]

        if make_undirected:
            src, dst = torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)
            if weights is not None:
                weights = torch.cat([weights, weights], dim=0)

        if src.numel() == 0:
            rowptr = torch.zeros(int(num_nodes) + 1, dtype=torch.long, device=device)
            col = torch.empty(0, dtype=torch.long, device=device)
            outgoing_mass = torch.zeros(int(num_nodes), dtype=torch.float32, device=device)
            return cls(int(num_nodes), rowptr, col, weights, outgoing_mass)

        order = torch.argsort(src * int(num_nodes) + dst, stable=True)
        src, dst = src[order], dst[order]
        if weights is not None:
            weights = weights[order]

        degree_values = torch.bincount(src, minlength=int(num_nodes))
        rowptr = torch.empty(int(num_nodes) + 1, dtype=torch.long, device=device)
        rowptr[0] = 0
        rowptr[1:] = torch.cumsum(degree_values, dim=0)
        if weights is None:
            outgoing_mass = degree_values.to(torch.float32)
        else:
            outgoing_mass = torch.zeros(int(num_nodes), dtype=torch.float32, device=device)
            outgoing_mass.scatter_add_(0, src, weights.to(torch.float32))

        return cls(int(num_nodes), rowptr, dst, weights, outgoing_mass)
