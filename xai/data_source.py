"""Model B(SaCas9) 설명 대시보드의 데이터 경계. 2학기 코드를 그대로 옮겨 왔다.

UI는 `XAIDataSource`("샘플 하나의 XAI 묶음을 달라")에만 의존한다.
`PrecomputedDataSource`는 Colab에서 사전 계산한 Testset 결과
(`data/final_analysis_result/` + `data/test_metadata.csv`)를 읽는다.

[2학기 병합] 이 파일이 다루는 Model B(SaCas9) 결과는 **사전계산값**이다.
실제 연동 모델(SpCas9 · Cas12a)의 XAI는 `xai/explain.py`가 모델에서 직접 계산한다.
두 경로는 서로 독립이며, 화면에서도 구분해 보여준다.

경로만 2학기 원본과 다르다: 데이터가 프로젝트 루트의 `data/` 아래에 있다.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

SEQ_LEN = 36
BASES = ("A", "C", "G", "T")
N_COMPLEX_SCENARIOS = 15

# Region boundaries (0-indexed, inclusive) on the 36bp sequence.
PAM_REGION = (25, 30)
SEED_REGION = (17, 24)
DISTAL_REGION = (0, 7)

# Model B's embedding is NT's last hidden state (7 tokens x 1280) flattened
# token-major, followed by the 4 physical features.
N_TOKENS = 7
HIDDEN_DIM = 1280
TOKEN_NT_SPAN = 6
PHYSICAL_FEATURE_KEYS = ("mfe", "dg", "tm", "gc")
PHYSICAL_FEATURE_LABELS = {"mfe": "MFE", "dg": "ΔG", "tm": "Tm", "gc": "GC"}
SHAP_GROUP_COUNT = N_TOKENS + len(PHYSICAL_FEATURE_KEYS)
CLS_GROUP_NAME = "[CLS] (global sequence context)"

# 프로젝트 루트의 data/ (xai/ 의 한 단계 위)
_XAI_TEST_ROOT = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RESULT_DIR = _XAI_TEST_ROOT / "final_analysis_result"
DEFAULT_METADATA_PATH = _XAI_TEST_ROOT / "test_metadata.csv"


def data_available() -> bool:
    """사전계산 데이터가 모두 있는지. 없으면 화면에서 Model B 섹션을 숨긴다."""
    return (
        (DEFAULT_RESULT_DIR / "model_analysis_arrays.npz").exists()
        and (DEFAULT_RESULT_DIR / "model_analysis_summary.json").exists()
        and (DEFAULT_RESULT_DIR / "model_b_testset_predictions.csv").exists()
        and DEFAULT_METADATA_PATH.exists()
    )


@dataclass(frozen=True)
class ShapGroup:
    """One Token-grouped SHAP bar: a signed contribution plus the input it summarizes."""

    name: str
    shap_value: float
    feature_value: str  # the 6-mer / physical value; "" for [CLS]

    @property
    def label(self) -> str:
        return f"{self.name}: {self.feature_value}" if self.feature_value else self.name


@dataclass(frozen=True)
class ComplexScenario:
    scenario_index: int
    mutation_id: str
    region: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class SampleBundle:
    """Everything the dashboard shows for one sample.

    `true_score`/`error` exist only for Testset samples and `ig_attribution`
    only for the Case Study samples; `None` is their explicit absence.
    """

    sample_id: int
    sequence: str
    prediction: float
    true_score: float | None
    error: float | None
    physical_features: dict[str, float]
    ism_delta: np.ndarray  # (36, 3), relative delta per alternative base
    ism_alt_bases: tuple[tuple[str, str, str], ...]  # lexicographic, per position
    complex_mismatch_delta: np.ndarray  # (15,)
    attention_rollout: np.ndarray  # (36, 36) Information Dependency
    shap_base_value: float
    shap_groups: tuple[ShapGroup, ...]  # 11 Token-grouped SHAP groups
    ig_attribution: np.ndarray | None  # (36,) projected, signed, L1-normalized


class XAIDataSource(Protocol):
    def sample_ids(self) -> list[int]: ...

    def ig_sample_ids(self) -> list[int]: ...

    def get_sample(self, sample_id: int) -> SampleBundle: ...

    def ranking(self) -> pd.DataFrame: ...

    def complex_scenarios(self) -> list[ComplexScenario]: ...


def alt_bases(wt_base: str) -> tuple[str, str, str]:
    """ISM's 3 alternative bases at a position, in the export's lexicographic order."""
    return tuple(sorted(set(BASES) - {wt_base}))


def token_grouped_shap(
    shap_row: np.ndarray, sequence: str, physical_features: dict[str, float]
) -> tuple[ShapGroup, ...]:
    """Collapse one 8,964-wide SHAP row into the 11 Token-grouped SHAP groups.

    Each token's 1,280 embedding dimensions are summed with sign, so the group
    keeps its push-up/push-down direction; physical features stay individual.
    """
    n_embedding = N_TOKENS * HIDDEN_DIM
    per_token = shap_row[:n_embedding].reshape(N_TOKENS, HIDDEN_DIM).sum(axis=1)
    groups = [ShapGroup(CLS_GROUP_NAME, float(per_token[0]), "")]
    for k in range(1, N_TOKENS):
        start = (k - 1) * TOKEN_NT_SPAN
        kmer = sequence[start : start + TOKEN_NT_SPAN]
        groups.append(
            ShapGroup(
                f"Token {k} (pos {start}-{start + TOKEN_NT_SPAN - 1})",
                float(per_token[k]),
                kmer,
            )
        )
    for j, key in enumerate(PHYSICAL_FEATURE_KEYS):
        groups.append(
            ShapGroup(
                PHYSICAL_FEATURE_LABELS[key],
                float(shap_row[n_embedding + j]),
                f"{physical_features[key]:.2f}",
            )
        )
    return tuple(groups)


class PrecomputedDataSource:
    """Stage-1 data source over the Colab-precomputed Testset export."""

    def __init__(
        self,
        result_dir: Path | str = DEFAULT_RESULT_DIR,
        metadata_path: Path | str = DEFAULT_METADATA_PATH,
    ):
        result_dir = Path(result_dir)
        with np.load(result_dir / "model_analysis_arrays.npz") as arrays:
            self._ism = arrays["ism_delta_model_b"]
            self._complex = arrays["mismatch_complex_model_b"]
            self._rollout = arrays["attention_rollout"]
            self._shap_values = arrays["shap_values"]
        with open(result_dir / "model_analysis_summary.json", encoding="utf-8") as f:
            summary = json.load(f)

        meta = pd.read_csv(metadata_path)
        preds = pd.read_csv(result_dir / "model_b_testset_predictions.csv")
        self._table = meta[["sample_id", "sequence", "true_score"]].merge(
            preds, on="sample_id", validate="one_to_one"
        )
        self._table = self._table.set_index("sample_id", drop=False).sort_index()

        self._ig = {
            int(case["sample_id"]): np.asarray(
                case["integrated_gradients"]["model_b_phase4_projected_norm_attr_36bp"],
                dtype=np.float64,
            )
            for case in summary["case_studies"]
        }
        self._scenarios = [
            ComplexScenario(
                scenario_index=int(m["scenario_index"]),
                mutation_id=m["mutation_id"],
                region=m["region"],
                positions=tuple(m["positions"]),
            )
            for m in sorted(
                summary["complex_mismatch_metadata"],
                key=lambda m: m["scenario_index"],
            )
        ]
        # TreeSHAP is additive around E[f(X)], which the export doesn't store;
        # recover it as the Testset-mean gap between prediction and SHAP sum
        # so every waterfall starts from the same reference.
        self._shap_base_value = float(
            np.mean(self._table["pred_raw"].to_numpy() - self._shap_values.sum(axis=1))
        )

    def sample_ids(self) -> list[int]:
        return self._table.index.tolist()

    def ig_sample_ids(self) -> list[int]:
        return sorted(self._ig)

    def complex_scenarios(self) -> list[ComplexScenario]:
        return list(self._scenarios)

    def ranking(self) -> pd.DataFrame:
        return (
            self._table[["sample_id", "sequence", "pred_raw"]]
            .rename(columns={"pred_raw": "prediction"})
            .sort_values("prediction", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

    def get_sample(self, sample_id: int) -> SampleBundle:
        if sample_id not in self._table.index:
            raise KeyError(f"Unknown sample_id {sample_id}")
        row = self._table.loc[sample_id]
        sequence = row["sequence"]
        prediction = float(row["pred_raw"])
        true_score = float(row["true_score"])
        physical = {k: float(row[k]) for k in PHYSICAL_FEATURE_KEYS}
        return SampleBundle(
            sample_id=int(sample_id),
            sequence=sequence,
            prediction=prediction,
            true_score=true_score,
            error=abs(true_score - prediction),
            physical_features=physical,
            ism_delta=self._ism[sample_id],
            ism_alt_bases=tuple(alt_bases(b) for b in sequence),
            complex_mismatch_delta=self._complex[sample_id],
            attention_rollout=self._rollout[sample_id],
            shap_base_value=self._shap_base_value,
            shap_groups=token_grouped_shap(
                self._shap_values[sample_id].astype(np.float64), sequence, physical
            ),
            ig_attribution=self._ig.get(int(sample_id)),
        )
