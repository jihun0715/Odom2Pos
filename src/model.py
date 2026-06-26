"""RNN models for odom-to-GT velocity mapping."""

from __future__ import annotations

import torch
from torch import nn


class RNNRegressor(nn.Module):
    """Sequence-to-sequence velocity regressor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        rnn_type: str = "rnn",
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.lower()
        rnn_cls: type[nn.RNNBase]
        if rnn_type == "rnn":
            rnn_cls = nn.RNN
        elif rnn_type == "gru":
            rnn_cls = nn.GRU
        else:
            raise ValueError(f"Unsupported RNN_TYPE: {rnn_type!r}")

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.output(y)
