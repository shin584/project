"""
CRISPR 예측 파이프라인의 데이터 레이어 전처리 모듈입니다.
단일 책임 원칙(SRP)과 개방-폐쇄 원칙(OCP)을 준수하도록 설계되었습니다.
"""
from typing import AbstractSet, List, Optional, Sequence, Tuple


SequenceRecord = Tuple[str, float]


class SequenceValidator:
    """
    [SRP] 데이터 품질 검증(Validation)을 전담하는 클래스.
    [OCP] allowed_chars를 주입받아, 향후 RNA 등 새로운 염기서열 규칙에 유연하게 대응합니다.
    """
    def __init__(self, allowed_chars: Optional[AbstractSet[str]] = None) -> None:
        # 허용된 문자가 없을 경우 기본값으로 DNA 염기서열 규격 강제
        self.allowed_chars = allowed_chars or frozenset({"A", "T", "G", "C"})

    def validate_sequence(self, sequence: str) -> str:
        """
        단일 염기서열의 유효성을 검사하고 정규화(대문자 변환)합니다.
        
        Args:
            sequence: 원시 염기서열 문자열
        Returns:
            대문자로 정규화된 염기서열
        Raises:
            ValueError: 빈 문자열이거나 허용되지 않은 문자가 포함된 경우
        """
        if not sequence:
            raise ValueError("Sequence is empty.")

        cleaned = sequence.upper()
        # 차집합 연산을 통해 허용되지 않은 문자 검출 (O(N) 성능 확보)
        invalid = set(cleaned) - self.allowed_chars
        if invalid:
            invalid_chars = "".join(sorted(invalid))
            raise ValueError(f"Invalid DNA characters: {invalid_chars}")

        return cleaned

    def validate_records(self, records: Sequence[SequenceRecord]) -> List[SequenceRecord]:
        """
        다수의 (Sequence, CleavageEfficiency) 튜플 리스트를 일괄 검증합니다.
        """
        validated: List[SequenceRecord] = []
        for sequence, efficiency in records:  # gene -> efficiency 로 변경
            cleaned = self.validate_sequence(sequence)
            validated.append((cleaned, efficiency)) # 여기도 맞춰서 변경
        return validated


class DatasetSplitter:
    """
    [SRP] 데이터셋 분할(Test Set 분리 및 Group K-Fold)을 전담하는 클래스.
    특정 레이블(Test Label)을 분리하거나, 레이블 단위로 K-Fold 그룹을 묶는 역할을 합니다.
    """
    def __init__(self, test_label: float = 0.0, k_folds: int = 3) -> None:
        self.test_label = test_label
        self.k_folds = k_folds

    def split_test_set(
        self, records: Sequence[SequenceRecord]
    ) -> Tuple[List[SequenceRecord], List[SequenceRecord]]:
        """
        전체 레코드에서 지정된 test_label을 기준으로 Train / Test 셋을 분할합니다.
        """
        train: List[SequenceRecord] = []
        test: List[SequenceRecord] = []
        for record in records:
            if record[1] == self.test_label:
                test.append(record)
            else:
                train.append(record)
        return train, test

    def group_k_fold(
        self, records: Sequence[SequenceRecord]
    ) -> List[Tuple[List[SequenceRecord], List[SequenceRecord]]]:
        """
        레이블(Label) 단위 기준으로 Group K-Fold 분할을 수행합니다.
        
        Raises:
            ValueError: 지정된 K 폴드 수가 유전자 그룹 수와 일치하지 않을 때 발생
        """
        # 고유한 레이블(Label) 그룹 추출 후 정렬
        groups = sorted({label for _, label in records})
        if self.k_folds != len(groups):
            raise ValueError("k_folds must equal the number of groups.")

        folds: List[Tuple[List[SequenceRecord], List[SequenceRecord]]] = []
        for val_label in groups:
            val = [record for record in records if record[1] == val_label]
            train = [record for record in records if record[1] != val_label]
            folds.append((train, val))

        return folds


class DataProcessor:
    """
    [Facade Pattern] 기존 클라이언트 코드를 보호하기 위한 파사드 클래스.
    내부적으로 Validator와 Splitter에 작업을 위임(Delegation)하여 낮은 결합도를 유지합니다.
    """
    def __init__(
        self,
        validator: Optional[SequenceValidator] = None,
        splitter: Optional[DatasetSplitter] = None,
    ) -> None:
        # 외부에서 의존성 주입(DI)이 없으면 기본 인스턴스 생성
        self.validator = validator or SequenceValidator()
        self.splitter = splitter or DatasetSplitter()

    def validate_sequence(self, sequence: str) -> str:
        return self.validator.validate_sequence(sequence)

    def validate_records(self, records: Sequence[SequenceRecord]) -> List[SequenceRecord]:
        return self.validator.validate_records(records)

    def split_test_set(
        self, records: Sequence[SequenceRecord]
    ) -> Tuple[List[SequenceRecord], List[SequenceRecord]]:
        return self.splitter.split_test_set(records)

    def group_k_fold(
        self, records: Sequence[SequenceRecord]
    ) -> List[Tuple[List[SequenceRecord], List[SequenceRecord]]]:
        return self.splitter.group_k_fold(records)
