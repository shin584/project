"""Data source boundary for the Model B explanation dashboard (issue #20).

The UI depends only on `XAIDataSource`: "give me one sample's XAI bundle".
Stage 1 (`PrecomputedDataSource`) reads the Colab-precomputed Testset export
in `final_analysis_result/` plus `test_metadata.csv`; stage 2 will swap in a
live-inference implementation without touching the UI (`plan_draft.md`).

Per ADR 0002 nothing here exposes Model A output or Concordant/Discordant
labels - the 15 Case Studies are only used as "samples with Integrated
Gradients available".
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

_XAI_TEST_ROOT = Path(__file__).resolve().parent.parent
# Token-grouped SHAP lives in the shared core (`xai_test/shap_grouping.py`)
# so the dashboard and the CLI compute identical groups from the same code.
# Unused names are re-exported: charts.py and the tests import them from here.
sys.path.append(str(_XAI_TEST_ROOT))
from shap_grouping import (  # noqa: F401
    CLS_GROUP_NAME,
    HIDDEN_DIM,
    N_TOKENS,
    PHYSICAL_FEATURE_KEYS,
    PHYSICAL_FEATURE_LABELS,
    SHAP_GROUP_COUNT,
    TOKEN_NT_SPAN,
    ShapGroup,
    token_grouped_shap,
)

SEQ_LEN = 36
BASES = ("A", "C", "G", "T")
N_COMPLEX_SCENARIOS = 15

# Region boundaries (0-indexed, inclusive) on the 36bp sequence. Mirrors the
# half-open constants in mismatch_profiling.py (Seed/Distal) and
# integrated_gradients.py (PAM); test_data_source.py guards against drift.
PAM_REGION = (25, 30)
SEED_REGION = (17, 24)
DISTAL_REGION = (0, 7)

DEFAULT_RESULT_DIR = _XAI_TEST_ROOT / "final_analysis_result"
DEFAULT_METADATA_PATH = _XAI_TEST_ROOT / "test_metadata.csv"


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
