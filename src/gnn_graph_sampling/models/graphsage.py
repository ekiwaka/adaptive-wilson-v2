from __future__ import annotations
from typing import Literal, Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSageClassifier(nn.Module):
    """GraphSAGE classifier with configurable normalization and prediction head.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden dimension used in every SAGE layer.
        out_channels: Number of output classes.
        num_layers: Number of SAGE layers. Must be at least 2.
        dropout: Dropout probability applied after each hidden layer.
        input_dropout: Dropout probability applied to input features before the
            first GraphSAGE layer.
        aggr: SAGE aggregation function.
        use_batch_norm: Legacy compatibility flag. When norm_type is not
            provided, True maps to batch normalization and False maps to
            no normalization.
        norm_type: One of batch, layer, or none.
        residual: Whether to add a residual connection after each hidden layer.
        jk_mode: last to use the final hidden layer only or cat to
            concatenate all hidden states before the prediction head.
        head_type: mlp for a 2-layer prediction head or linear for a
            single linear classifier.
        head_hidden_dim: Hidden dimension of the MLP prediction head.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        *,
        input_dropout: float = 0.0,
        aggr: str = 'mean',
        use_batch_norm: bool = True,
        norm_type: Optional[Literal['batch', 'layer', 'none']] = None,
        residual: bool = True,
        jk_mode: str = 'last',
        head_type: Literal['mlp', 'linear'] = 'mlp',
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError('num_layers must be at least 2.')
        if jk_mode not in {'last', 'cat'}:
            raise ValueError("jk_mode must be either 'last' or 'cat'.")
        if norm_type is None:
            norm_type = 'batch' if use_batch_norm else 'none'
        if norm_type not in {'batch', 'layer', 'none'}:
            raise ValueError("norm_type must be one of 'batch', 'layer', or 'none'.")
        if head_type not in {'mlp', 'linear'}:
            raise ValueError("head_type must be either 'mlp' or 'linear'.")

        self.dropout = float(dropout)
        self.input_dropout = float(input_dropout)
        self.use_residual = bool(residual)
        self.jk_mode = jk_mode
        self.num_layers = int(num_layers)
        self.norm_type = norm_type
        self.head_type = head_type

        layer_input_dims = [in_channels] + [hidden_channels] * num_layers
        self.convs = nn.ModuleList(
            [SAGEConv(layer_input_dims[index], hidden_channels, aggr=aggr) for index in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [self._make_norm_layer(hidden_channels, norm_type) for _ in range(num_layers)]
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
        """Run GraphSAGE over a graph or induced subgraph."""

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
