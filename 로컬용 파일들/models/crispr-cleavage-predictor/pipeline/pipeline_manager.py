"""
데이터 검증, 특징 추출, 추론을 순차적으로 연결하는 파이프라인 매니저입니다.
CRISPR 워크플로우를 위한 파이프-필터 오케스트레이션 계층을 구현합니다.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from data.data_processor import DatasetSplitter, SequenceRecord, SequenceValidator
from features.feature_extractor import SequenceFeatureExtractor, TensorLike
from training.interfaces import IPredictor, ILossFunction, IOptimizer, ITrainerCallback
from training.trainer import Trainer


class PipelineManager:
    def __init__(
        self,
        model: Optional[IPredictor] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[ILossFunction] = None,
        optimizer: Optional[IOptimizer] = None,
        callbacks: Optional[List[ITrainerCallback]] = None,
        model_factory: Optional[Callable[[], IPredictor]] = None,
        loss_factory: Optional[Callable[[], ILossFunction]] = None,
        optimizer_factory: Optional[
            Callable[[IPredictor], Tuple[torch.optim.Optimizer, IOptimizer]]
        ] = None,
        scheduler_factory: Optional[Callable[[torch.optim.Optimizer], object]] = None,
        callbacks_factory: Optional[
            Callable[[IPredictor, Optional[object], int], List[ITrainerCallback]]
        ] = None,
        validator: Optional[SequenceValidator] = None,
        splitter: Optional[DatasetSplitter] = None,
        extractor: Optional[SequenceFeatureExtractor] = None,
    ) -> None:
        # 의존성 주입으로 오케스트레이터가 구체 구현과 느슨하게 결합되도록 유지합니다.
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.model_factory = model_factory
        self.loss_factory = loss_factory
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.callbacks_factory = callbacks_factory
        self.validator = validator or SequenceValidator()
        self.splitter = splitter or DatasetSplitter()
        self.extractor = extractor

    def run_inference_pipeline(self, dna: str) -> float:
        if self.extractor is None:
            self.extractor = SequenceFeatureExtractor(validator=self.validator)
            
        # 검증 -> 특징 추출 -> 추론 순서로 실행합니다.
        features = self.extractor.get_features(dna)
        tensor = self._ensure_tensor(features)
        output = self.model.predict(tensor)
        return float(output.squeeze().item())

    def run_train_pipeline(
        self,
        epochs: int,
        fold_loaders: Optional[Sequence[Tuple[DataLoader, DataLoader]]] = None,
    ) -> List[float]:
        if fold_loaders:
            return self._run_kfold_pipeline(epochs, fold_loaders)

        if self.train_loader is None:
            raise ValueError("train_loader must be provided for training.")
        if self.criterion is None:
            raise ValueError("criterion must be provided for training.")
        if self.optimizer is None:
            raise ValueError("optimizer must be provided for training.")

        trainer = Trainer(
            model=self.model,
            dataloader=self.train_loader,
            val_loader=self.val_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            callbacks=self.callbacks,
        )

        losses: List[float] = []
        for _ in range(epochs):
            losses.append(trainer.train_epoch())
            if trainer.stop_training:
                break

        self._finalize_callbacks(self.callbacks)
        return losses

    def _run_kfold_pipeline(
        self,
        epochs: int,
        fold_loaders: Sequence[Tuple[DataLoader, DataLoader]],
    ) -> List[float]:
        if self.model_factory is None:
            raise ValueError("model_factory must be provided for K-Fold training.")
        if self.loss_factory is None:
            raise ValueError("loss_factory must be provided for K-Fold training.")
        if self.optimizer_factory is None:
            raise ValueError("optimizer_factory must be provided for K-Fold training.")
        if self.callbacks_factory is None:
            raise ValueError("callbacks_factory must be provided for K-Fold training.")

        fold_val_losses: List[float] = []
        fold_val_mae: List[float] = []

        for fold_index, (train_loader, val_loader) in enumerate(fold_loaders, start=1):
            model = self.model_factory()
            raw_optimizer, optimizer_adapter = self.optimizer_factory(model)
            criterion = self.loss_factory()

            scheduler = None
            if self.scheduler_factory is not None:
                scheduler = self.scheduler_factory(raw_optimizer)

            callbacks = self.callbacks_factory(model, scheduler, fold_index)

            trainer = Trainer(
                model=model,
                dataloader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer_adapter,
                callbacks=callbacks,
            )

            for _ in range(epochs):
                trainer.train_epoch()
                if trainer.stop_training:
                    break

            val_loss, val_mae = trainer.validate()
            fold_val_losses.append(val_loss)
            fold_val_mae.append(val_mae)

            self._finalize_callbacks(callbacks)

        if fold_val_losses:
            avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)
            avg_val_mae = sum(fold_val_mae) / len(fold_val_mae)
            print(
                "K-Fold average validation metrics "
                f"(loss={avg_val_loss:.4f}, mae={avg_val_mae:.4f})"
            )

        return fold_val_losses

    def _score_fold(self, records: Sequence[SequenceRecord]) -> List[float]:
        # 각 서열을 독립적으로 점수화하여 상태 의존성을 줄입니다.
        scores: List[float] = []
        for sequence, _ in records:
            score = self.run_inference_pipeline(sequence)
            scores.append(score)
        return scores

    @staticmethod
    def _ensure_tensor(features: TensorLike) -> torch.Tensor:
        # torch 기반이 아닌 추출기가 반환한 numpy 배열도 수용합니다.
        if isinstance(features, torch.Tensor):
            return features
        return torch.tensor(features)

    @staticmethod
    def _finalize_callbacks(callbacks: List[ITrainerCallback]) -> None:
        for callback in callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end()
