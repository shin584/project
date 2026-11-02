"""
Trainer callback implementations.
"""
from typing import Dict, Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from training.interfaces import ITrainerCallback


class ProgressBarLogger(ITrainerCallback):
    """
    Keras-style progress bar logger for epoch-level metrics.
    """

    def __init__(self, total_epochs: Optional[int] = None) -> None:
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_end(self, metrics: Dict[str, float]) -> None:
        self.current_epoch += 1
        parts = []
        for key, value in metrics.items():
            parts.append(f"{key}: {value:.4f}")
            
        metrics_str = " - ".join(parts)
        epoch_str = f"Epoch {self.current_epoch}/{self.total_epochs}" if self.total_epochs else f"Epoch {self.current_epoch}"

        print(f"{epoch_str} ━━━━━━━━━━━━━━━━━━━━ {metrics_str}")

    def on_train_end(self) -> None:
        pass


class EarlyStoppingCallback(ITrainerCallback):
    """
    Early stopping based on validation loss.
    """

    def __init__(self, model: torch.nn.Module, patience: int, save_path: str) -> None:
        self.model = model
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float("inf")
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, metrics: Dict[str, float]) -> None:
        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
            torch.save(self.model.state_dict(), self.save_path)
            return

        self.wait += 1
        if self.wait >= self.patience:
            self.stop_training = True


class LRSchedulerCallback(ITrainerCallback):
    """
    ReduceLROnPlateau scheduler hook.
    """

    def __init__(self, scheduler: ReduceLROnPlateau) -> None:
        self.scheduler = scheduler

    def on_epoch_end(self, metrics: Dict[str, float]) -> None:
        val_loss = metrics.get("val_loss")
        if val_loss is None:
            return

        prev_lr = self._current_lr()
        self.scheduler.step(val_loss)
        new_lr = self._current_lr()
        if new_lr < prev_lr:
            tqdm.write(
                f"Learning rate reduced: {prev_lr:.6f} -> {new_lr:.6f}"
            )

    def _current_lr(self) -> float:
        return float(self.scheduler.optimizer.param_groups[0]["lr"])
