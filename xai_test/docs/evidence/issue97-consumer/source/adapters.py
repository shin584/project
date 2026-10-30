from abc import ABC, abstractmethod

import torch
from torch import nn


class IInputAdapter(nn.Module, ABC):
    """Abstract input adapter interface."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OneHotAdapter(IInputAdapter):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Expected input shape [Batch, Seq, Input_Dim]"
        assert x.size(-1) == self.input_dim, "Unexpected input feature dimension"
        return self.proj(x)
    