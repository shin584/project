"""
Trainer with dependency injection for training components.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.interfaces import IPredictor, ILossFunction, IOptimizer, ITrainerCallback


class Trainer:
    """
    Trainer orchestrator using dependency injection.
    """

    def __init__(
        self,
        model: IPredictor,
        dataloader: DataLoader,
        criterion: ILossFunction,
        optimizer: IOptimizer,
        callbacks: List[ITrainerCallback],
        val_loader: DataLoader | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.stop_training = False

    def train_epoch(self) -> float:
        if hasattr(self.model, "train"):
            self.model.train()

        total_loss = 0.0
        total_mae = 0.0
        batch_count = 0

        for features, labels in self.dataloader:
            self.optimizer.zero_grad()
            predictions = self.model.forward(features)
            loss = self._calculate_loss(predictions, labels)
            mae = F.l1_loss(predictions, labels, reduction="mean")
            loss.backward()
            self.optimizer.step()

            total_loss += self._loss_value(loss)
            total_mae += self._loss_value(mae)
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        avg_mae = total_mae / batch_count if batch_count > 0 else 0.0
        val_loss, val_mae = self.validate()
        try:
            current_lr = float(self.optimizer.param_groups[0]["lr"])
        except AttributeError:
            current_lr = float(self.optimizer.optimizer.param_groups[0]["lr"])

        metrics: Dict[str, float] = {
            "loss": avg_loss,
            "mae": avg_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "learning_rate": current_lr,
        }
        for callback in self.callbacks:
            callback.on_epoch_end(metrics)
            if getattr(callback, "stop_training", False):
                self.stop_training = True

        return avg_loss

    def validate(self) -> Tuple[float, float]:
        if hasattr(self.model, "eval"):
            self.model.eval()

        total_loss = 0.0
        total_mae = 0.0
        batch_count = 0

        loader = self.val_loader or self.dataloader
        with torch.no_grad():
            for features, labels in loader:
                predictions = self.model.forward(features)
                loss = self._calculate_loss(predictions, labels)
                mae = F.l1_loss(predictions, labels, reduction="mean")
                total_loss += self._loss_value(loss)
                total_mae += self._loss_value(mae)
                batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        avg_mae = total_mae / batch_count if batch_count > 0 else 0.0
        return avg_loss, avg_mae

    def _calculate_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.criterion(predictions, labels)

    @staticmethod
    def _loss_value(loss: torch.Tensor) -> float:
        if hasattr(loss, "item"):
            return float(loss.item())
        return float(loss)
