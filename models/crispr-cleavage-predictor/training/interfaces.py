"""
Training interfaces for optimizer, loss, and callbacks.
"""
from abc import ABC, abstractmethod
from typing import Dict
import torch
from torch import nn


class IPredictor(ABC):
    """
    [DIP] 예측기(Predictor) 추상화 인터페이스.
    파이프라인이나 상위 모듈이 PyTorch의 구체적인 `DNNPredictor`에 묶이지 않고,
    이 추상 클래스에 의존하게 하여 향후 머신러닝/다른 딥러닝 프레임워크 기반 
    예측기로 쉽게 교체할 수 있도록 합니다. (의존성 역전 원칙)
    """

    @abstractmethod
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        벡터 피처를 기반으로 예측 결과(Cleavage Efficiency Score)를 반환합니다.
        """
        raise NotImplementedError




class IOptimizer(ABC):
    """
    Optimizer interface for dependency inversion.
    """

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError


class ILossFunction(ABC):
    """
    Loss function interface for dependency inversion.
    """

    @abstractmethod
    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ITrainerCallback(ABC):
    """
    Callback interface for observer pattern.
    """

    @abstractmethod
    def on_epoch_end(self, metrics: Dict[str, float]) -> None:
        raise NotImplementedError
