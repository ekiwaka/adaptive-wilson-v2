from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MixHopConv


class MixHopClassifier(nn.Module):
    """MixHop classifier for heterophilous graph node classification.

    MixHop concatenates multiple adjacency powers in each message-passing layer.
    This lets the model combine immediate-neighbor and higher-order information,
    which is often useful on heterophilous graphs where one-hop neighbors may
    have different labels.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        input_dropout: float = 0.0,
        powers: tuple[int, ...] = (0, 1, 2),
        use_batch_norm: bool = True,
        norm_type: Optional[Literal['batch', 'layer', 'none']] = None,
        residual: bool = True,
        jk_mode: Literal['last', 'cat'] = 'last',
        head_type: Literal['mlp', 'linear'] = 'mlp',
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError('MixHopClassifier requires num_layers >= 2.')
        if jk_mode not in {'last', 'cat'}:
            raise ValueError("jk_mode must be 'last' or 'cat'.")
        if head_type not in {'mlp', 'linear'}:
            raise ValueError("head_type must be 'mlp' or 'linear'.")
        if not powers:
            raise ValueError('MixHop powers must be non-empty.')

        self.dropout = float(dropout)
        self.input_dropout = float(input_dropout)
        self.use_residual = bool(residual)
        self.jk_mode = jk_mode
        self.num_layers = int(num_layers)
        self.powers = tuple(int(power) for power in powers)

        effective_norm_type: Literal['batch', 'layer', 'none'] = (
            norm_type if norm_type is not None else ('batch' if use_batch_norm else 'none')
        )

        self.input_projections = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.compressors = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        layer_input_dim = int(in_channels)
        mixed_dim = hidden_channels * len(self.powers)
        for _ in range(num_layers):
            self.input_projections.append(nn.Linear(layer_input_dim, hidden_channels))
            self.convs.append(MixHopConv(hidden_channels, hidden_channels, powers=list(self.powers)))
            self.compressors.append(nn.Linear(mixed_dim, hidden_channels))
            self.norms.append(self._make_norm_layer(hidden_channels, effective_norm_type))
            self.residual_projections.append(
                nn.Identity()
                if layer_input_dim == hidden_channels
                else nn.Linear(layer_input_dim, hidden_channels, bias=False)
            )
            layer_input_dim = hidden_channels

        head_input_dim = hidden_channels if jk_mode == 'last' else hidden_channels * num_layers
        mlp_hidden = int(head_hidden_dim or hidden_channels)
        if head_type == 'linear':
            self.head = nn.Linear(head_input_dim, out_channels)
        else:
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(mlp_hidden, out_channels),
            )

    @staticmethod
    def _make_norm_layer(hidden_channels: int, norm_type: Literal['batch', 'layer', 'none']) -> nn.Module:
        if norm_type == 'batch':
            return nn.BatchNorm1d(hidden_channels)
        if norm_type == 'layer':
            return nn.LayerNorm(hidden_channels)
        return nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden_states: list[torch.Tensor] = []
        hidden = F.dropout(x, p=self.input_dropout, training=self.training)

        for projection, conv, compressor, norm, residual_projection in zip(
            self.input_projections,
            self.convs,
            self.compressors,
            self.norms,
            self.residual_projections,
        ):
            residual_value = residual_projection(hidden)
            hidden = projection(hidden)
            hidden = conv(hidden, edge_index)
            hidden = compressor(hidden)
            hidden = norm(hidden)
            hidden = F.relu(hidden)
            if self.use_residual:
                hidden = hidden + residual_value
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden_states.append(hidden)

        representation = torch.cat(hidden_states, dim=-1) if self.jk_mode == 'cat' else hidden_states[-1]
        return self.head(representation)
