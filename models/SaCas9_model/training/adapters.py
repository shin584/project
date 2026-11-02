"""
PyTorch adapters for loss functions and optimizers.
"""
from typing import Any

import torch

from training.interfaces import ILossFunction, IOptimizer


class PyTorchLossAdapter(ILossFunction):
    """
    Adapter for torch.nn loss modules.
    """

    def __init__(self, criterion: torch.nn.Module) -> None:
        self.criterion = criterion

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, labels)


class PyTorchOptimizerAdapter(IOptimizer):
    """
    Adapter for torch.optim optimizers.
    """

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()
