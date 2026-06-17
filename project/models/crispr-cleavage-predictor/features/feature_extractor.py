"""
CRISPR 파이프라인의 Feature Layer 모듈입니다.
DNA 서열을 머신러닝/딥러닝 모델이 이해할 수 있는 벡터(Tensor)로 변환하는 역할을 담당합니다.
"""
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np

from data.data_processor import SequenceValidator


TensorLike = Union["torch.Tensor", np.ndarray]

# 1. 인터페이스 정의
class IFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, sequence: str) -> TensorLike:
        pass

class OneHotExtractor(IFeatureExtractor):
    def __init__(self, sequence_length: int = 30) -> None:
        self.sequence_length = sequence_length
        self._mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    def extract(self, sequence: str) -> TensorLike:
        if len(sequence) != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {len(sequence)}"
            )

        matrix = np.zeros((self.sequence_length, 4), dtype=np.float32)
        for idx, base in enumerate(sequence.upper()):
            if base not in self._mapping:
                raise ValueError(f"Invalid base '{base}' in sequence: {sequence}")
            matrix[idx, self._mapping[base]] = 1.0
        return matrix

# 3. 파사드 패턴
class SequenceFeatureExtractor:
    def __init__(
        self,
        extractor: Optional[IFeatureExtractor] = None,
        validator: Optional[SequenceValidator] = None,
    ) -> None:
        self.extractor = extractor or OneHotExtractor()
        self.validator = validator

    def get_features(self, dna: str) -> TensorLike:
        sequence = dna
        if self.validator is not None:
            sequence = self.validator.validate_sequence(dna)
        return self.extractor.extract(sequence)