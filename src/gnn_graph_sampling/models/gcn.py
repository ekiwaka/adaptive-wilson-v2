from __future__ import annotations
from typing import Literal, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNClassifier(nn.Module):
    """A stable GCN baseline for sampled node classification experiments.

    This keeps the same residual / normalization / JK / head interface used by
    the GraphSAGE model so experiments can swap backbones through YAML only.
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
        use_batch_norm: bool = True,
        norm_type: Optional[Literal['batch', 'layer', 'none']] = None,
        residual: bool = True,
        jk_mode: Literal['last', 'cat'] = 'last',
        head_type: Literal['mlp', 'linear'] = 'mlp',
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError('GCNClassifier requires num_layers >= 2.')
        if jk_mode not in {'last', 'cat'}:
            raise ValueError("jk_mode must be 'last' or 'cat'.")
        if head_type not in {'mlp', 'linear'}:
            raise ValueError("head_type must be 'mlp' or 'linear'.")

        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.input_dropout = float(input_dropout)
        self.use_residual = bool(residual)
        self.jk_mode = jk_mode
        effective_norm_type: Literal['batch', 'layer', 'none'] = (
            norm_type if norm_type is not None else ('batch' if use_batch_norm else 'none')
        )

        layer_input_dims = [in_channels] + [hidden_channels] * (num_layers - 1)
        self.convs = nn.ModuleList(
            [
                GCNConv(
                    in_channels=layer_input_dims[index],
                    out_channels=hidden_channels,
                    add_self_loops=True,
                    normalize=True,
                    cached=False,
                    bias=True,
                )
                for index in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [self._make_norm_layer(hidden_channels, effective_norm_type) for _ in range(num_layers)]
        )
        self.residual_projections = nn.ModuleList(
            [
                nn.Identity() if layer_input_dims[index] == hidden_channels else nn.Linear(layer_input_dims[index], hidden_channels, bias=False)
                for index in range(num_layers)
            ]
        )

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
        for conv, norm, residual_projection in zip(self.convs, self.norms, self.residual_projections):
            residual_value = residual_projection(hidden)
            hidden = conv(hidden, edge_index)
            hidden = norm(hidden)
            hidden = F.relu(hidden)
            if self.use_residual:
                hidden = hidden + residual_value
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden_states.append(hidden)

        representation = torch.cat(hidden_states, dim=-1) if self.jk_mode == 'cat' else hidden_states[-1]
        return self.head(representation)
