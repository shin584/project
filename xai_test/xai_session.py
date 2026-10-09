"""XAI demo session: the testable core behind the demo CLI (issues #21, #23-#26).

`XAISession` is the single seam between the export contract and the terminal:
it is constructed from the standardized export directory
(`model_analysis_summary.json`, `model_analysis_arrays.npz`,
`model_b_testset_predictions.csv`) plus `test_metadata.csv`, and its public
operations return rendered text. With no predictors it runs in cached-only
mode and never touches a model.

`explain <CaseID>` renders the cached explanation sections as plain-text
tables: Integrated Gradients (per position, with Distal/Seed/PAM labels and
region shares), Token-grouped SHAP, an ISM summary and complex-mismatch
results by region.

`report()` opens with a Testset integrity check (the export's whole-column
SHA256, every row's `sequence_hash`, every Case Study's sequence against its
Testset row) and Model B's headline Testset metrics computed from the
exported predictions CSV, then the global findings: the ablation with Paired
Bootstrap 95% CIs, the physical-feature SHAP share, mismatch sensitivity by
region and IG attribution share by region, closing with the export handoff.
Every interpretation line is computed from the numbers its section prints;
none asserts a predetermined conclusion.

Unlike the dashboard (ADR 0002), the session shows Model A next to Model B
and surfaces Concordant/Discordant labels, because the presentation's claim
is comparative (ADR 0003).
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from case_study_selection import (
    CONCORDANT,
    PRIMARY_DISCORDANT,
    REVERSE_PRIMARY_DISCORDANT,
    REVERSE_SECONDARY_DISCORDANT,
    SECONDARY_DISCORDANT,
)
from ism_sweep import alt_bases_for
from mismatch_profiling import (
    DISTAL_REGION_END,
    DISTAL_REGION_START,
    PAM_REGION_END,
    PAM_REGION_START,
    SEED_REGION_END,
    SEED_REGION_START,
)
from model_b_testset_export import CASE_STUDY_MATCH_ATOL
from paired_bootstrap import compute_metrics
from shap_analysis import EMBEDDING_DIM, aggregate_physical_contribution_ratio
from shap_grouping import token_grouped_shap

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


# Half-open [start, end) position ranges on the 36bp sequence, 5' to 3'.
REGIONS = (
    ("Distal", DISTAL_REGION_START, DISTAL_REGION_END),
    ("Seed", SEED_REGION_START, SEED_REGION_END),
    ("PAM", PAM_REGION_START, PAM_REGION_END),
)
NO_REGION = "-"
ISM_TOP_POSITIONS = 5

DETERMINISTIC_PROJECTION_NOTE = (
    "Model B IG는 Deterministic Projection Rule 가정: 각 6-mer token의 attribution을 "
    "그 token이 덮는 6개 nucleotide에 균등 분배한 값 "
    "(모델이 nucleotide를 독립적으로 인식한다는 근거가 아님)"
)

# The region the Seed-sensitivity hypothesis predicts mismatches hurt most.
HYPOTHESIS_REGION = "Seed"

# (label, summary key) of each ablation variant, in display order.
ABLATION_VARIANTS = (
    ("Model B-Embedding", "embedding_only"),
    ("Model B-Physical", "physical_only"),
    ("Model B-Full", "full_model"),
)
# (label, CI key suffix) of each variant Model B-Full is bootstrapped against.
ABLATION_BASELINES = (
    ("Embedding-only", "embedding_only"),
    ("Physical-only", "physical_only"),
)
# (label, CI key metric, whether a higher value is better).
ABLATION_CI_METRICS = (("ΔSpearman", "spearman", True), ("ΔMSE", "mse", False))

HANDOFF_FILES = (
    "model_analysis_summary.json",
    "model_analysis_arrays.npz",
    "model_b_testset_predictions.csv",
)

_ARRAY_KEYS = (
    "ism_delta_model_a",
    "ism_delta_model_b",
    "mismatch_single_min_model_a",
    "mismatch_single_min_model_b",
    "mismatch_complex_model_a",
    "mismatch_complex_model_b",
    "shap_values",
)


def _region_of(pos: int) -> str:
    """The Distal/Seed/PAM label of a 0-indexed position, or `NO_REGION`."""
    for name, start, end in REGIONS:
        if start <= pos < end:
            return name
    return NO_REGION


SHARE_NAMES = (*(name for name, _, _ in REGIONS), "other")


def _region_shares(attribution: np.ndarray) -> np.ndarray:
    """Each region's fraction of the absolute attribution mass, in
    `SHARE_NAMES` order (the last entry is everything outside the regions)."""
    mass = np.abs(attribution)
    total = mass.sum()
    region_mass = [mass[start:end].sum() for _, start, end in REGIONS]
    shares = np.array([*region_mass, total - sum(region_mass)])
    return shares / total if total else np.zeros_like(shares)


def _format_region_shares(attribution: np.ndarray) -> str:
    """One-line `Distal 0.250 | Seed ... | other ...` share summary."""
    shares = _region_shares(attribution)
    return " | ".join(f"{name} {s:.3f}" for name, s in zip(SHARE_NAMES, shares))


def _ci_verdict(lower: float, upper: float, higher_is_better: bool) -> str | None:
    """None when the CI includes zero, else whether Full is better or worse."""
    if lower <= 0 <= upper:
        return None
    return "더 좋음" if (lower > 0) == higher_is_better else "더 나쁨"


def _most_sensitive_line(tag: str, model: str, medians: dict[str, float]) -> str:
    """Name the region with the largest median drop and check it against the
    Seed-sensitivity hypothesis, whichever way it comes out. The spread across
    regions is printed too, so a near-flat profile isn't read as a strong peak."""
    region = min(medians, key=medians.get)
    if medians[region] >= 0:
        return f"해석({tag}): {model}는 어느 region에서도 median 감소 없음"
    agreement = "일치" if region == HYPOTHESIS_REGION else "불일치"
    spread = max(medians.values()) - medians[region]
    return (
        f"해석({tag}): {model}는 {region}에서 가장 크게 감소 "
        f"(median {medians[region]:+.3f}, region 간 차이 {spread:.3f})"
        f" - {HYPOTHESIS_REGION} 민감 가설과 {agreement}"
    )


def _sequence_sha256(sequence: str) -> str:
    """Per-row hash; same convention as test_dataset_verifier.generate_sha256."""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _column_sha256(sequences: pd.Series) -> str:
    """Whole-column hash; same convention as test_dataset_verifier.py and
    step5_export.py, so it is comparable to the export's `sha256_checksum`."""
    return hashlib.sha256(pd.util.hash_pandas_object(sequences).values).hexdigest()


def _truncated_id_list(ids, limit: int = 10) -> str:
    shown = ", ".join(str(i) for i in ids[:limit])
    return shown + (f" 외 {len(ids) - limit}개" if len(ids) > limit else "")


def _short_id(case_id: str) -> str:
    """`DISCORDANT_P01` -> `P01`: the form typed at the demo prompt."""
    return case_id.split("_", 1)[-1]


class XAISessionError(ValueError):
    """A user-facing error: the CLI prints its message instead of a traceback."""


class Predictors(NamedTuple):
    """Model A, Model B and the XAI predictor; the CLI constructs them one at
    a time.

    `model_a` / `model_b` only need `predict(list[str]) -> np.ndarray`
    (`Model_A_Predictor`, `Model_B_Predictor`); `model_b_xai` is the
    `Model_B_XAIPredictor` live IG builds on.
    """

    model_a: Any
    model_b: Any
    model_b_xai: Any


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
        self._complex_meta = summary["complex_mismatch_metadata"]
        self._ablation = summary["global_evaluation"]["ablation_results"]
        self._recorded_column_sha256 = summary["metadata"]["sha256_checksum"]
        with np.load(result_dir / "model_analysis_arrays.npz") as arrays:
            self._arrays = {key: arrays[key] for key in _ARRAY_KEYS}
        self._meta = pd.read_csv(metadata_path)
        self._predictions_b = pd.read_csv(
            result_dir / "model_b_testset_predictions.csv"
        )
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
        header = "\n".join(
            [
                f"{case['case_id']}  (sample_id {case['sample_id']}, {case_type})",
                f"sequence     {case['sequence']}",
                f"true score   {case['true_score']:.3f}",
                self._score_line("Model A", a, case["sequence"], "model_a"),
                self._score_line("Model B", b, case["sequence"], "model_b"),
                f"{case_type}: {CASE_TYPE_EXPLANATIONS.get(case_type, '설명 없음')}",
            ]
        )
        sections = (
            self._ig_section(case),
            self._shap_section(case),
            self._ism_section(case),
            self._complex_mismatch_section(case),
        )
        return "\n\n".join((header, *sections))

    def report(self) -> str:
        sections = (
            self._integrity_section(),
            self._performance_section(),
            self._ablation_section(),
            self._physical_shap_section(),
            self._mismatch_region_section(),
            self._ig_region_section(),
            self._handoff_section(),
        )
        return "\n\n".join(sections)

    def _score_line(
        self, model: str, cached: dict, sequence: str, predictor_field: str
    ) -> str:
        """The cached score, plus - when predictors are loaded - the score
        recomputed live, a match indicator and the elapsed time."""
        if self.cached_only:
            return (
                f"{model:<12} pred {cached['pred_raw']:.3f}"
                f"  error {cached['error']:.3f}  (cached)"
            )
        predictor = getattr(self._predictors, predictor_field)
        start = time.perf_counter()
        live = float(predictor.predict([sequence])[0])
        elapsed = time.perf_counter() - start
        diff = live - cached["pred_raw"]
        match = (
            "[일치]" if abs(diff) <= CASE_STUDY_MATCH_ATOL else f"[불일치 Δ{diff:+.4f}]"
        )
        return (
            f"{model:<12} cached {cached['pred_raw']:.3f}"
            f"  cached error {cached['error']:.3f}"
            f"  live {live:.3f} {match} ({elapsed:.2f}s)"
        )

    def _integrity_section(self) -> str:
        lines = ["== Testset integrity (SHA256)"]
        recorded = self._recorded_column_sha256
        actual = _column_sha256(self._meta["sequence"])
        if actual == recorded:
            lines.append(f"PASS  whole-column SHA256 일치 ({actual[:16]}...)")
        else:
            lines.append(
                "FAIL  whole-column SHA256 불일치: export 기록 "
                f"{recorded[:16]}... / 현재 Testset {actual[:16]}..."
            )

        rehashed = self._meta["sequence"].map(_sequence_sha256)
        bad_rows = self._meta.loc[
            rehashed != self._meta["sequence_hash"], "sample_id"
        ].tolist()
        n = len(self._meta)
        if bad_rows:
            lines.append(
                f"FAIL  sequence_hash 불일치 {len(bad_rows)}/{n}행: "
                f"sample_id {_truncated_id_list(bad_rows)}"
            )
        else:
            lines.append(f"PASS  sequence_hash {n}/{n}행 일치")

        testset_seq = dict(zip(self._meta["sample_id"], self._meta["sequence"]))
        bad_cases = [
            f"{case['case_id']} (sample_id {case['sample_id']})"
            for case in self._cases
            if testset_seq.get(case["sample_id"]) != case["sequence"]
        ]
        if bad_cases:
            lines.append(
                f"FAIL  Case Study sequence가 Testset 행과 불일치 "
                f"{len(bad_cases)}/{len(self._cases)}개: {', '.join(bad_cases)}"
            )
        else:
            lines.append(
                f"PASS  Case Study sequence {len(self._cases)}/{len(self._cases)}개가 "
                "Testset 행과 일치"
            )
        return "\n".join(lines)

    def _performance_section(self) -> str:
        joined = self._meta[["sample_id", "true_score"]].merge(
            self._predictions_b[["sample_id", "pred_raw"]],
            on="sample_id",
            validate="one_to_one",
        )
        metrics = compute_metrics(
            joined["true_score"].to_numpy(), joined["pred_raw"].to_numpy()
        )
        lines = [
            (
                "== Model B Testset 성능 "
                f"(model_b_testset_predictions.csv에서 계산, n={len(joined)})"
            ),
            (
                f"Spearman {metrics['spearman']:.3f} | Pearson {metrics['pearson']:.3f}"
                f" | MAE {metrics['mae']:.4f} | MSE {metrics['mse']:.4f}"
            ),
        ]
        # An inner join would otherwise silently score a subset of the Testset.
        missing = sorted(set(self._meta["sample_id"]) - set(joined["sample_id"]))
        if missing:
            lines.append(
                f"FAIL  predictions CSV가 Testset {len(joined)}/{len(self._meta)}행만 "
                f"포함 - 누락 sample_id {_truncated_id_list(missing)}"
            )
        lines.append(
            "주의: production Model B는 Testset과 겹치는 데이터로 학습되어, "
            "clean dev split으로 재학습한 ablation variant와 직접 비교할 수 없음"
        )
        return "\n".join(lines)

    def _ablation_section(self) -> str:
        lines = [
            "== Ablation (clean dev split으로 재학습한 variant, Testset 평가)",
            f"{'variant':<18} {'Spearman':>8} {'Pearson':>8} {'MAE':>7} {'MSE':>7}",
        ]
        for label, key in ABLATION_VARIANTS:
            m = self._ablation[key]
            lines.append(
                f"{label:<18} {m['spearman']:>8.4f} {m['pearson']:>8.4f}"
                f" {m['mae']:>7.4f} {m['mse']:>7.4f}"
            )
        lines.append("Paired Bootstrap 95% CI (Δ = Full - variant)")
        cis = self._ablation["paired_bootstrap_95ci"]
        vs_embedding = []  # significant differences vs Embedding-only
        for baseline, suffix in ABLATION_BASELINES:
            for label, metric, higher_is_better in ABLATION_CI_METRICS:
                lower, upper = cis[f"delta_{metric}_full_vs_{suffix}"]
                verdict = _ci_verdict(lower, upper, higher_is_better)
                status = (
                    "0 포함 → 유의한 차이 없음"
                    if verdict is None
                    else f"0 미포함 → Full이 유의하게 {verdict}"
                )
                lines.append(
                    f"Full vs {baseline} {label} [{lower:+.4f}, {upper:+.4f}]  {status}"
                )
                if suffix == "embedding_only" and verdict is not None:
                    vs_embedding.append(f"{label} {verdict}")
        if vs_embedding:
            finding = (
                "물리 feature를 더한 Full이 Embedding-only 대비 "
                f"{', '.join(vs_embedding)} (CI가 0 미포함)"
            )
        else:
            finding = (
                "Full vs Embedding-only의 모든 CI가 0을 포함 → 물리 feature 추가에 의한 "
                "유의한 차이 없음"
            )
        lines.append(f"해석: {finding}")
        return "\n".join(lines)

    def _physical_shap_section(self) -> str:
        shap = self._arrays["shap_values"]
        share = aggregate_physical_contribution_ratio(shap)
        n_features = shap.shape[1]
        n_physical = n_features - EMBEDDING_DIM
        feature_share = n_physical / n_features
        ratio = share / feature_share
        comparison = "기대치보다 작음" if ratio < 1 else "기대치 이상"
        return "\n".join(
            [
                "== Physical-feature SHAP share (Model B, Testset 전체 |SHAP| 대비)",
                (
                    f"물리 feature {n_physical}개의 |SHAP| 비율 {share:.4%} "
                    f"(feature 수 비율 {n_physical}/{n_features} = {feature_share:.4%})"
                ),
                (
                    f"해석: 물리 feature의 SHAP 기여는 feature 수 비율의 {ratio:.2f}배로, "
                    f"균등 기여 {comparison}"
                ),
            ]
        )

    def _mismatch_region_section(self) -> str:
        single = {
            model: {
                name: float(np.nanmedian(self._arrays[key][:, start:end]))
                for name, start, end in REGIONS
            }
            for model, key in (
                ("Model A", "mismatch_single_min_model_a"),
                ("Model B", "mismatch_single_min_model_b"),
            )
        }
        complex_ = {
            model: {
                region: float(np.nanmedian(self._arrays[key][:, indices]))
                for region, indices in self._complex_scenarios_by_region().items()
            }
            for model, key in (
                ("Model A", "mismatch_complex_model_a"),
                ("Model B", "mismatch_complex_model_b"),
            )
        }
        lines = [
            (
                "== Mismatch sensitivity by region "
                f"(median relative delta, Testset n={len(self._meta)})"
            ),
            (
                "single = position별 3개 alt base 중 worst-case drop, "
                "region 내 모든 sample x position의 median"
            ),
            "complex = scenario region별 모든 sample x scenario의 median",
            f"{'':<20} {'Model A':>8} {'Model B':>8}",
        ]
        for tag, medians in (("single", single), ("complex", complex_)):
            for region in medians["Model A"]:
                lines.append(
                    f"{tag + ' ' + region:<20} {medians['Model A'][region]:>+8.3f}"
                    f" {medians['Model B'][region]:>+8.3f}"
                )
            lines += [_most_sensitive_line(tag, m, medians[m]) for m in medians]
        return "\n".join(lines)

    def _ig_region_section(self) -> str:
        lines = [
            (
                f"== IG attribution share by region (Case Study {len(self._cases)}개 "
                "평균, |attr| 비율)"
            ),
            f"{'model':<8} " + " ".join(f"{name:>6}" for name in SHARE_NAMES),
        ]
        focus = {}
        for model, key in (
            ("Model A", "model_a_norm_attr_36bp"),
            ("Model B", "model_b_phase4_projected_norm_attr_36bp"),
        ):
            shares = np.mean(
                [
                    _region_shares(np.asarray(case["integrated_gradients"][key]))
                    for case in self._cases
                ],
                axis=0,
            )
            lines.append(f"{model:<8} " + " ".join(f"{s:>6.3f}" for s in shares))
            focus[model] = sum(
                shares[SHARE_NAMES.index(name)] for name in ("PAM", "Seed")
            )
        a, b = focus["Model A"], focus["Model B"]
        if np.isclose(a, b):
            verdict = "두 모델의 PAM+Seed 집중도가 같음"
        else:
            verdict = f"{'Model A' if a > b else 'Model B'}가 PAM+Seed에 attribution을 더 집중"
        lines.append(
            f"해석: PAM+Seed share Model A {a:.3f} / Model B {b:.3f} → {verdict}"
        )
        return "\n".join(lines)

    def _handoff_section(self) -> str:
        return "\n".join(
            [
                "== Handoff (visualization layer)",
                "이 report의 수치는 모두 standardized export에서 계산됨: "
                + ", ".join(HANDOFF_FILES),
                (
                    "시각화 layer(dashboard)가 소비하는 contract도 같은 export "
                    "(model_analysis_summary.json + arrays)"
                ),
            ]
        )

    def _complex_scenarios_by_region(self) -> dict[str, list[int]]:
        by_region: dict[str, list[int]] = {}
        for scenario in self._complex_meta:
            by_region.setdefault(scenario["region"], []).append(
                scenario["scenario_index"]
            )
        return by_region

    def _ig_section(self, case: dict) -> str:
        ig = case["integrated_gradients"]
        attr_a = np.asarray(ig["model_a_norm_attr_36bp"])
        attr_b = np.asarray(ig["model_b_phase4_projected_norm_attr_36bp"])
        lines = [
            "== Integrated Gradients (cached, L1-normalized, signed)",
            f"{'pos':>3} {'base':<4} {'region':<6} {'Model A':>8} {'Model B':>8}",
        ]
        for pos, base in enumerate(case["sequence"]):
            lines.append(
                f"{pos:>3} {base:<4} {_region_of(pos):<6}"
                f" {attr_a[pos]:>+8.3f} {attr_b[pos]:>+8.3f}"
            )
        lines += [
            f"Model A share |attr|  {_format_region_shares(attr_a)}",
            f"Model B share |attr|  {_format_region_shares(attr_b)}",
            DETERMINISTIC_PROJECTION_NOTE,
        ]
        return "\n".join(lines)

    def _shap_section(self, case: dict) -> str:
        groups = token_grouped_shap(
            self._arrays["shap_values"][case["sample_id"]],
            case["sequence"],
            case["per_sample_physical_values"],
        )
        lines = [
            f"== Token-grouped SHAP (Model B, {len(groups)} groups, cached)",
            "signed SHAP, |값| 큰 순서 (+는 점수를 올림, -는 내림)",
        ]
        for group in sorted(groups, key=lambda g: abs(g.shap_value), reverse=True):
            lines.append(f"{group.shap_value:+.4f}  {group.label}")
        return "\n".join(lines)

    def _ism_section(self, case: dict) -> str:
        sequence = case["sequence"]
        lines = ["== ISM summary (cached, relative delta = (mutant - WT) / |WT|)"]
        for model, key in (
            ("Model A", "ism_delta_model_a"),
            ("Model B", "ism_delta_model_b"),
        ):
            delta = self._arrays[key][case["sample_id"]]  # (36, 3)
            lines.append(f"{model}: 가장 민감한 위치 {ISM_TOP_POSITIONS}개")
            if np.isnan(delta).all():
                lines.append(
                    "  (|Score_WT|가 너무 작아 relative delta가 정의되지 않음)"
                )
                continue
            magnitude = np.nan_to_num(np.abs(delta), nan=-1.0)
            strongest_alt = magnitude.argmax(axis=1)
            ranked = np.argsort(-magnitude.max(axis=1), kind="stable")
            lines.append(f"{'pos':>3} {'region':<6} {'mut':<4} {'delta':>7}")
            for pos in ranked[:ISM_TOP_POSITIONS]:
                wt = sequence[pos]
                alt = alt_bases_for(wt)[strongest_alt[pos]]
                lines.append(
                    f"{pos:>3} {_region_of(pos):<6} {wt}>{alt:<2}"
                    f" {delta[pos, strongest_alt[pos]]:>+7.3f}"
                )
        return "\n".join(lines)

    def _complex_mismatch_section(self, case: dict) -> str:
        sample_id = case["sample_id"]
        delta_a = self._arrays["mismatch_complex_model_a"][sample_id]
        delta_b = self._arrays["mismatch_complex_model_b"][sample_id]
        by_region = self._complex_scenarios_by_region()
        lines = [
            "== Complex mismatch (cached, relative delta, region별 scenario 평균 / 최저)",
            (
                f"{'region':<12} {'n':>2} {'A mean':>7} {'B mean':>7}"
                f" {'A worst':>8} {'B worst':>8}"
            ),
        ]
        for region, indices in by_region.items():
            a, b = delta_a[indices], delta_b[indices]
            lines.append(
                f"{region:<12} {len(indices):>2} {np.nanmean(a):>+7.3f}"
                f" {np.nanmean(b):>+7.3f} {np.nanmin(a):>+8.3f} {np.nanmin(b):>+8.3f}"
            )
        return "\n".join(lines)

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
