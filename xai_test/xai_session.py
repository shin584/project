"""XAI demo session: the testable core behind the demo CLI (issues #21, #23).

`XAISession` is the single seam between the export contract and the terminal:
it is constructed from the standardized export directory
(`model_analysis_summary.json`, `model_analysis_arrays.npz`,
`model_b_testset_predictions.csv`) plus `test_metadata.csv`, and its public
operations return rendered text. With no predictors it runs in cached-only
mode and never touches a model. So far only the summary JSON (Case Studies)
and the metadata are read; the arrays and predictions arrive with `report()`
and raw-sequence `explain` in later slices of #21.

Unlike the dashboard (ADR 0002), the session shows Model A next to Model B
and surfaces Concordant/Discordant labels, because the presentation's claim
is comparative (ADR 0003).
"""

import json
from pathlib import Path

import pandas as pd
from case_study_selection import (
    CONCORDANT,
    PRIMARY_DISCORDANT,
    REVERSE_PRIMARY_DISCORDANT,
    REVERSE_SECONDARY_DISCORDANT,
    SECONDARY_DISCORDANT,
)

_XAI_TEST_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = _XAI_TEST_ROOT / "final_analysis_result"
DEFAULT_METADATA_PATH = _XAI_TEST_ROOT / "test_metadata.csv"

# One-line case-type explanations; Korean prose, English technical terms.
CASE_TYPE_EXPLANATIONS = {
    PRIMARY_DISCORDANT: (
        "Model A는 크게 틀리고(오차 상위 25%) Model B는 정확한(오차 하위 25%) 샘플 "
        "- Model B 개선의 핵심 신호"
    ),
    SECONDARY_DISCORDANT: (
        "Model A 오차가 중앙값 이상이고 Model B가 그 오차를 절반 이상 줄인 샘플 "
        "- Primary Discordant 부족분을 채운 완화 기준"
    ),
    REVERSE_PRIMARY_DISCORDANT: (
        "Model A는 정확하고(오차 하위 25%) Model B는 크게 틀린(오차 상위 25%) 샘플 "
        "- Model B가 Model A보다 나빠진 반례"
    ),
    REVERSE_SECONDARY_DISCORDANT: (
        "Model B 오차가 중앙값 이상이고 Model A가 그 오차의 절반 이하인 샘플 "
        "- Reverse Primary Discordant 부족분을 채운 완화 기준"
    ),
    CONCORDANT: (
        "두 모델 모두 정확한(오차 하위 25%) 샘플 - 두 모델이 일치하는 sanity check"
    ),
}


def _short_id(case_id: str) -> str:
    """`DISCORDANT_P01` -> `P01`: the form typed at the demo prompt."""
    return case_id.split("_", 1)[-1]


class XAISessionError(ValueError):
    """A user-facing error: the CLI prints its message instead of a traceback."""


class XAISession:
    def __init__(
        self,
        result_dir: Path | str = DEFAULT_RESULT_DIR,
        metadata_path: Path | str = DEFAULT_METADATA_PATH,
        predictors=None,
    ):
        result_dir = Path(result_dir)
        with open(result_dir / "model_analysis_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        self._cases = summary["case_studies"]
        self._cases_by_id = {case["case_id"].upper(): case for case in self._cases}
        # Read now so later operations (Testset matching, integrity checks)
        # share one loaded copy; the tracer bullet itself only needs the cache.
        self._meta = pd.read_csv(metadata_path)
        self._predictors = predictors

    @property
    def cached_only(self) -> bool:
        return self._predictors is None

    def cases(self) -> str:
        header = (
            f"{'case_id':<17} {'case_type':<29} {'sample_id':>9} {'true':>7}"
            f" {'pred_A':>7} {'err_A':>7} {'pred_B':>7} {'err_B':>7}"
        )
        lines = [
            f"Case Study {len(self._cases)}개 (점수는 cached)",
            header,
            "-" * len(header),
        ]
        for case in self._cases:
            a, b = case["model_a"], case["model_b"]
            lines.append(
                f"{case['case_id']:<17} {case['case_type']:<29} {case['sample_id']:>9}"
                f" {case['true_score']:>7.3f} {a['pred_raw']:>7.3f} {a['error']:>7.3f}"
                f" {b['pred_raw']:>7.3f} {b['error']:>7.3f}"
            )
        return "\n".join(lines)

    def explain(self, query: str) -> str:
        case = self._resolve_case(query)
        a, b = case["model_a"], case["model_b"]
        case_type = case["case_type"]
        return "\n".join(
            [
                f"{case['case_id']}  (sample_id {case['sample_id']}, {case_type})",
                f"sequence     {case['sequence']}",
                f"true score   {case['true_score']:.3f}",
                f"Model A      pred {a['pred_raw']:.3f}  error {a['error']:.3f}  (cached)",
                f"Model B      pred {b['pred_raw']:.3f}  error {b['error']:.3f}  (cached)",
                f"{case_type}: {CASE_TYPE_EXPLANATIONS.get(case_type, '설명 없음')}",
            ]
        )

    def _resolve_case(self, query: str) -> dict:
        """Find a Case Study by full `case_id` or its short suffix (`P01` -> `DISCORDANT_P01`)."""
        key = query.strip().upper()
        if key in self._cases_by_id:
            return self._cases_by_id[key]
        for case_id, case in self._cases_by_id.items():
            if key and _short_id(case_id) == key:
                return case
        known = ", ".join(_short_id(c["case_id"]) for c in self._cases)
        raise XAISessionError(
            f"알 수 없는 Case Study ID: {query!r} (사용 가능: {known})"
        )
