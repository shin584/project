"""Tests for the XAI demo session (issues #23-#28).

Drives only the session's public operations, against a small synthetic
export directory built in the test - no real model is loaded. The only real
files read are the checked-in Testset and export, by the PAM validation tests
(issue #31), in cached-only mode.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xai_session
from mismatch_profiling import (
    DISTAL_REGION_END,
    DISTAL_REGION_START,
    PAM_REGION_END,
    PAM_REGION_START,
    SEED_REGION_END,
    SEED_REGION_START,
)
from paired_bootstrap import compute_metrics
from shap_analysis import aggregate_physical_contribution_ratio
from shap_grouping import HIDDEN_DIM, SHAP_GROUP_COUNT
from xai_session import Predictors, XAISession, XAISessionError

SEQ = "ACGTACGTACGTACGTACGTACGTACGGAGTACGTA"

CASES = [
    ("DISCORDANT_P01", "Primary Discordant", 3, 0.90, 0.55, 0.88),
    ("DISCORDANT_S01", "Secondary Discordant", 4, 0.70, 0.50, 0.62),
    ("DISCORDANT_R01", "Reverse Primary Discordant", 1, 0.40, 0.39, 0.61),
    ("DISCORDANT_RS01", "Reverse Secondary Discordant", 5, 0.30, 0.33, 0.45),
    ("CONCORDANT_C01", "Concordant", 0, 0.37, 0.371, 0.372),
]

# DISCORDANT_P01 (sample_id 3) carries hand-picked explanation values below.
P01_SAMPLE = 3

# Integrated Gradients (already L1-normalized, as exported).
IG_A = np.zeros(36)
IG_A[3], IG_A[20], IG_A[26] = 0.25, -0.25, 0.5  # Distal / Seed / PAM
IG_B = np.zeros(36)
IG_B[18], IG_B[33] = 0.4, -0.6  # Seed / unlabelled 3' flank

PHYSICAL_VALUES = {"mfe": -10.9, "dg": -81.5, "tm": 69.26, "gc": 61.11}

COMPLEX_REGIONS = ["Seed"] * 5 + ["Distal"] * 5 + ["Intermittent"] * 5
COMPLEX_A = np.array([-0.5] * 5 + [-0.1] * 5 + [-0.3] * 5)
COMPLEX_B = np.array([-0.7] * 5 + [-0.05] * 5 + [-0.2] * 5)


def _case_entry(case_id, case_type, sample_id, true, pred_a, pred_b):
    return {
        "case_id": case_id,
        "sample_id": sample_id,
        "original_id": 100 + sample_id,
        "sequence": SEQ,
        "true_score": true,
        "model_a": {"pred_raw": pred_a, "error": abs(true - pred_a)},
        "model_b": {"pred_raw": pred_b, "error": abs(true - pred_b)},
        "case_type": case_type,
        "per_sample_physical_values": PHYSICAL_VALUES,
        "integrated_gradients": {
            "model_a_norm_attr_36bp": IG_A.tolist(),
            "model_b_phase4_projected_norm_attr_36bp": IG_B.tolist(),
        },
    }


def _complex_metadata():
    return [
        {
            "scenario_index": i,
            "mutation_id": f"{region.lower()}_scenario_{i}",
            "region": region,
            "positions": [i],
            "description": f"{region} scenario {i}",
        }
        for i, region in enumerate(COMPLEX_REGIONS)
    ]


def _arrays(n):
    ism_a = np.zeros((n, 36, 3))
    ism_a[P01_SAMPLE, 22, 1] = -0.8
    ism_a[P01_SAMPLE, 5, 0] = 0.3
    ism_b = np.zeros((n, 36, 3))
    ism_b[P01_SAMPLE, 27, 2] = -0.9
    complex_a = np.zeros((n, 15))
    complex_a[P01_SAMPLE] = COMPLEX_A
    complex_b = np.zeros((n, 15))
    complex_b[P01_SAMPLE] = COMPLEX_B
    shap = np.zeros((n, 8964), dtype=np.float32)
    shap[P01_SAMPLE, :HIDDEN_DIM] = 0.05 / HIDDEN_DIM  # [CLS]
    shap[P01_SAMPLE, 2 * HIDDEN_DIM] = -0.2  # Token 2 (pos 6-11)
    shap[P01_SAMPLE, 8960 + 3] = 0.1  # GC
    shap[0, 8960 + 0] = -0.02  # MFE on another sample
    # Testset-wide region patterns for `report`; P01's own rows are kept.
    for sample in range(n):
        if sample != P01_SAMPLE:
            complex_a[sample] = COMPLEX_A_TESTSET
            complex_b[sample] = COMPLEX_B_TESTSET
    return {
        "ism_delta_model_a": ism_a,
        "ism_delta_model_b": ism_b,
        "mismatch_single_min_model_a": _single_min(n, SINGLE_MIN_A),
        "mismatch_single_min_model_b": _single_min(n, SINGLE_MIN_B),
        "mismatch_complex_model_a": complex_a,
        "mismatch_complex_model_b": complex_b,
        "shap_values": shap,
    }


# Single-mismatch worst-case drop per region, identical across samples.
# Model A peaks in PAM (against the Seed-sensitivity hypothesis), Model B in Seed.
SINGLE_MIN_A = {"Distal": -0.1, "Seed": -0.2, "PAM": -0.4}
SINGLE_MIN_B = {"Distal": -0.05, "Seed": -0.3, "PAM": -0.1}
_REGION_SPANS = {
    "Distal": (DISTAL_REGION_START, DISTAL_REGION_END),
    "Seed": (SEED_REGION_START, SEED_REGION_END),
    "PAM": (PAM_REGION_START, PAM_REGION_END),
}


def _single_min(n, by_region):
    values = np.zeros((n, 36))
    for region, value in by_region.items():
        start, end = _REGION_SPANS[region]
        values[:, start:end] = value
    return values


# Complex-mismatch deltas for every non-P01 sample: Model A drops most on
# Intermittent, Model B on Seed.
COMPLEX_A_TESTSET = np.array([-0.02] * 5 + [-0.01] * 5 + [-0.03] * 5)
COMPLEX_B_TESTSET = np.array([-0.04] * 5 + [-0.01] * 5 + [-0.02] * 5)


def _metrics(spearman, pearson, mae, mse):
    return {"spearman": spearman, "pearson": pearson, "mae": mae, "mse": mse}


ABLATION = {
    "embedding_only": _metrics(0.8427, 0.8573, 0.0929, 0.0142),
    "physical_only": _metrics(0.7479, 0.7568, 0.1287, 0.0238),
    "full_model": _metrics(0.8447, 0.8588, 0.0933, 0.0141),
    "paired_bootstrap_95ci": {
        "delta_spearman_full_vs_embedding_only": [-0.0009, 0.0050],
        "delta_mse_full_vs_embedding_only": [-0.0002, 0.0001],
        "delta_spearman_full_vs_physical_only": [0.0711, 0.1236],
        "delta_mse_full_vs_physical_only": [-0.0117, -0.0077],
    },
}


N_SAMPLES = 6
TRUE_SCORES = np.array([0.2, 0.35, 0.5, 0.9, 0.7, 0.3])
PRED_B = np.array([0.25, 0.3, 0.55, 0.85, 0.6, 0.4])


def _sequences():
    # Distinct rows so a tampered sequence is attributable to one sample_id.
    seqs = [SEQ[:i] + "T" + SEQ[i + 1 :] for i in range(N_SAMPLES)]
    for _, _, sample_id, *_ in CASES:
        seqs[sample_id] = SEQ
    return seqs


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _column_sha256(sequences):
    # Same convention as test_dataset_verifier.py / step5_export.py.
    return hashlib.sha256(
        pd.util.hash_pandas_object(pd.Series(sequences, name="sequence")).values
    ).hexdigest()


@pytest.fixture
def export_dir(tmp_path):
    n = N_SAMPLES
    sequences = _sequences()
    summary = {
        "metadata": {"total_samples": n, "sha256_checksum": _column_sha256(sequences)},
        "global_evaluation": {
            "ablation_results": ABLATION,
            # Deliberately stale: the report must recompute from the arrays.
            "aggregate_physical_shap_contribution_ratio": 0.5,
        },
        "complex_mismatch_metadata": _complex_metadata(),
        "case_studies": [_case_entry(*c) for c in CASES],
    }
    (tmp_path / "model_analysis_summary.json").write_text(json.dumps(summary))
    np.savez(tmp_path / "model_analysis_arrays.npz", **_arrays(n))
    pd.DataFrame(
        {"sample_id": range(n), "pred_raw": PRED_B}
        | {key: [value] * n for key, value in PHYSICAL_VALUES.items()}
    ).to_csv(tmp_path / "model_b_testset_predictions.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": range(n),
            "original_id": range(100, 100 + n),
            "sequence": sequences,
            "true_score": TRUE_SCORES,
            "sequence_hash": [_sha256(s) for s in sequences],
        }
    ).to_csv(tmp_path / "test_metadata.csv", index=False)
    return tmp_path


@pytest.fixture
def session(export_dir):
    return XAISession(export_dir, export_dir / "test_metadata.csv")


def test_session_without_predictors_is_cached_only(session):
    assert session.cached_only


def test_cases_lists_every_case_study_with_both_models(session):
    text = session.cases()
    for case_id, case_type, sample_id, true, pred_a, pred_b in CASES:
        line = next(ln for ln in text.splitlines() if case_id in ln)
        assert case_type in line
        assert str(sample_id) in line
        for value in (true, pred_a, abs(true - pred_a), pred_b, abs(true - pred_b)):
            assert f"{value:.3f}" in line
    assert "cached" in text


@pytest.mark.parametrize("query", ["P01", "p01", "DISCORDANT_P01", "discordant_p01"])
def test_explain_resolves_short_and_full_case_ids(session, query):
    text = session.explain(query)
    assert "DISCORDANT_P01" in text
    assert "0.900" in text  # true score
    assert "0.550" in text and "0.350" in text  # Model A pred / error
    assert "0.880" in text and "0.020" in text  # Model B pred / error


def test_explain_labels_scores_as_cached(session):
    header = session.explain("P01").split("\n== ")[0]  # scores, before the sections
    model_lines = [
        ln for ln in header.splitlines() if ln.startswith(("Model A", "Model B"))
    ]
    assert model_lines
    assert all("cached" in ln for ln in model_lines)


@pytest.mark.parametrize(
    "query, case_type, marker",
    [
        ("P01", "Primary Discordant", "Model A"),
        ("S01", "Secondary Discordant", "Model A"),
        ("R01", "Reverse Primary Discordant", "Model B"),
        ("RS01", "Reverse Secondary Discordant", "Model B"),
        ("C01", "Concordant", "두 모델"),
    ],
)
def test_explain_prints_a_case_type_line(session, query, case_type, marker):
    text = session.explain(query)
    line = next(ln for ln in text.splitlines() if ln.startswith(f"{case_type}:"))
    assert marker in line


@pytest.mark.parametrize("query", ["P09", "X01", "DISCORDANT_P09", ""])
def test_explain_unknown_case_id_raises_clear_error(session, query):
    with pytest.raises(XAISessionError, match="Case Study"):
        session.explain(query)


def _section(text, title):
    """The lines of one `explain` section, from its `== title` header to the next."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"== {title}"))
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("== ")),
        len(lines),
    )
    return lines[start + 1 : end]


def _position_row(section, pos):
    return next(ln for ln in section if ln.split()[:1] == [str(pos)])


def test_explain_renders_all_four_sections_in_cached_only_mode(session):
    assert session.cached_only
    text = session.explain("P01")
    for title in (
        "Integrated Gradients",
        "Token-grouped SHAP",
        "ISM",
        "Complex mismatch",
    ):
        assert _section(text, title), title


def test_ig_table_has_every_position_with_base_and_region(session):
    section = _section(session.explain("P01"), "Integrated Gradients")
    for pos in range(36):
        row = _position_row(section, pos).split()
        assert row[1] == SEQ[pos]
        if DISTAL_REGION_START <= pos < DISTAL_REGION_END:
            assert row[2] == "Distal"
        elif SEED_REGION_START <= pos < SEED_REGION_END:
            assert row[2] == "Seed"
        elif PAM_REGION_START <= pos < PAM_REGION_END:
            assert row[2] == "PAM"
        else:
            assert row[2] == "-"
    assert _position_row(section, 26).split()[3:] == ["+0.500", "+0.000"]
    assert _position_row(section, 33).split()[3:] == ["+0.000", "-0.600"]


def test_ig_region_shares_per_model(session):
    section = _section(session.explain("P01"), "Integrated Gradients")
    line_a = next(ln for ln in section if ln.startswith("Model A share"))
    line_b = next(ln for ln in section if ln.startswith("Model B share"))
    for region, share in (("Distal", 0.25), ("Seed", 0.25), ("PAM", 0.5)):
        assert f"{region} {share:.3f}" in line_a
    for region, share in (("Distal", 0.0), ("Seed", 0.4), ("PAM", 0.0)):
        assert f"{region} {share:.3f}" in line_b


def test_ig_section_states_the_deterministic_projection_rule(session):
    section = _section(session.explain("P01"), "Integrated Gradients")
    note = next(ln for ln in section if "Deterministic Projection Rule" in ln)
    assert "Model B" in note


def test_token_grouped_shap_has_eleven_groups_sorted_by_magnitude(session):
    section = _section(session.explain("P01"), "Token-grouped SHAP")
    rows = [ln for ln in section if ln.lstrip()[:1] in "+-" and ln.strip()]
    assert len(rows) == SHAP_GROUP_COUNT
    values = [float(ln.split()[0]) for ln in rows]
    assert values == sorted(values, key=abs, reverse=True)
    assert rows[0].startswith("-0.2000") and "Token 2 (pos 6-11): GTACGT" in rows[0]
    assert rows[1].startswith("+0.1000") and "GC: 61.11" in rows[1]
    assert rows[2].startswith("+0.0500") and "[CLS]" in rows[2]


def test_ism_summary_lists_most_sensitive_positions_per_model(session):
    section = _section(session.explain("P01"), "ISM")
    split = next(i for i, ln in enumerate(section) if ln.startswith("Model B"))
    a_rows = [ln for ln in section[:split] if ln.split()[:1] != ["Model"]]
    b_rows = [ln for ln in section[split:] if ln.split()[:1] != ["Model"]]
    # Ranked by |relative delta|. pos 22 wt G -> alts (A, C, T): index 1 is C;
    # pos 27 wt G -> index 2 is T.
    a_pos = [ln.split()[0] for ln in a_rows if ln.split()[:1] != ["pos"]]
    assert a_pos[:2] == ["22", "5"]
    row_22 = next(ln for ln in a_rows if ln.split()[:1] == ["22"])
    assert "Seed" in row_22 and "G>C" in row_22 and "-0.800" in row_22
    row_27 = next(ln for ln in b_rows if ln.split()[:1] == ["27"])
    assert "PAM" in row_27 and "G>T" in row_27 and "-0.900" in row_27


def test_complex_mismatch_results_by_region_for_both_models(session):
    section = _section(session.explain("P01"), "Complex mismatch")
    for region, a, b in (
        ("Seed", -0.5, -0.7),
        ("Distal", -0.1, -0.05),
        ("Intermittent", -0.3, -0.2),
    ):
        line = next(ln for ln in section if ln.split()[:1] == [region])
        assert f"{a:+.3f}" in line and f"{b:+.3f}" in line


# --- report: Testset integrity + Model B headline performance (issue #25) ---


def _edit_csv(path, edit):
    df = pd.read_csv(path)
    edit(df)
    df.to_csv(path, index=False)


def _edit_summary(export_dir, edit):
    path = export_dir / "model_analysis_summary.json"
    summary = json.loads(path.read_text())
    edit(summary)
    path.write_text(json.dumps(summary))


def _report(export_dir):
    return XAISession(export_dir, export_dir / "test_metadata.csv").report()


def _integrity_lines(text):
    return [ln for ln in _section(text, "Testset integrity") if ln.strip()]


def test_report_integrity_passes_on_an_intact_export(export_dir):
    lines = _integrity_lines(_report(export_dir))
    assert len(lines) == 3
    assert all(ln.startswith("PASS") for ln in lines)
    assert f"{N_SAMPLES}/{N_SAMPLES}" in lines[1]
    assert f"{len(CASES)}/{len(CASES)}" in lines[2]


def test_report_flags_a_tampered_testset_sequence(export_dir):
    def tamper(df):
        df.loc[2, "sequence"] = "G" + df.loc[2, "sequence"][1:]

    _edit_csv(export_dir / "test_metadata.csv", tamper)
    lines = _integrity_lines(_report(export_dir))
    column, rows, _ = lines
    assert column.startswith("FAIL") and "SHA256" in column
    assert rows.startswith("FAIL") and "sample_id 2" in rows


def test_report_flags_a_tampered_row_hash(export_dir):
    def tamper(df):
        df.loc[4, "sequence_hash"] = "f" * 64

    _edit_csv(export_dir / "test_metadata.csv", tamper)
    column, rows, cases = _integrity_lines(_report(export_dir))
    assert column.startswith("PASS")  # the sequences themselves are intact
    assert rows.startswith("FAIL") and "sample_id 4" in rows
    assert cases.startswith("PASS")


def test_report_flags_a_tampered_column_hash(export_dir):
    def tamper(summary):
        summary["metadata"]["sha256_checksum"] = "a" * 64

    _edit_summary(export_dir, tamper)
    column, rows, cases = _integrity_lines(_report(export_dir))
    assert column.startswith("FAIL") and "a" * 8 in column
    assert rows.startswith("PASS") and cases.startswith("PASS")


def test_report_flags_a_case_study_sequence_that_differs_from_its_testset_row(
    export_dir,
):
    def tamper(summary):
        case = summary["case_studies"][0]  # DISCORDANT_P01, sample_id 3
        case["sequence"] = "C" + case["sequence"][1:]

    _edit_summary(export_dir, tamper)
    column, rows, cases = _integrity_lines(_report(export_dir))
    assert column.startswith("PASS") and rows.startswith("PASS")
    assert cases.startswith("FAIL")
    assert "DISCORDANT_P01" in cases and "sample_id 3" in cases


def _performance_values(text):
    section = _section(text, "Model B Testset")
    line = next(ln for ln in section if ln.startswith("Spearman"))
    tokens = line.replace("|", " ").split()
    return dict(zip(tokens[::2], map(float, tokens[1::2]))), section


def test_report_model_b_performance_is_computed_from_the_predictions_csv(
    export_dir,
):
    values, section = _performance_values(_report(export_dir))
    expected = compute_metrics(TRUE_SCORES, PRED_B)
    for name, key in (
        ("Spearman", "spearman"),
        ("Pearson", "pearson"),
        ("MAE", "mae"),
        ("MSE", "mse"),
    ):
        assert values[name] == pytest.approx(expected[key], abs=5e-4)
    assert any("ablation" in ln for ln in section)  # not directly comparable

    # A different CSV yields different numbers: nothing is hard-coded.
    shifted = PRED_B[::-1].copy()

    def replace(df):
        df["pred_raw"] = shifted

    _edit_csv(export_dir / "model_b_testset_predictions.csv", replace)
    values, _ = _performance_values(_report(export_dir))
    assert values["MAE"] == pytest.approx(
        compute_metrics(TRUE_SCORES, shifted)["mae"], abs=5e-4
    )


def test_report_flags_predictions_that_do_not_cover_the_testset(export_dir):
    def drop(df):
        df.drop(index=[1, 4], inplace=True)

    _edit_csv(export_dir / "model_b_testset_predictions.csv", drop)
    _, section = _performance_values(_report(export_dir))
    warning = next(ln for ln in section if ln.startswith("FAIL"))
    assert f"{N_SAMPLES - 2}/{N_SAMPLES}" in warning
    assert "sample_id 1, 4" in warning


# --- report: global findings with computed interpretation lines (issue #26) ---


def _np_arrays(export_dir):
    with np.load(export_dir / "model_analysis_arrays.npz") as arrays:
        return {key: arrays[key] for key in arrays.files}


def _edit_arrays(export_dir, edit):
    arrays = _np_arrays(export_dir)
    edit(arrays)
    np.savez(export_dir / "model_analysis_arrays.npz", **arrays)


def _edit_cis(export_dir, **cis):
    def edit(summary):
        stored = summary["global_evaluation"]["ablation_results"]
        stored["paired_bootstrap_95ci"].update(cis)

    _edit_summary(export_dir, edit)


def _line(section, prefix):
    return next(ln for ln in section if ln.startswith(prefix))


def test_report_renders_every_global_section_in_cached_only_mode(session):
    assert session.cached_only
    text = session.report()
    for title in (
        "Testset integrity",
        "Model B Testset",
        "Ablation",
        "Physical-feature SHAP share",
        "Mismatch sensitivity",
        "IG attribution share",
        "Handoff",
    ):
        assert _section(text, title), title


def test_report_ablation_shows_each_variant_and_ci(session):
    section = _section(session.report(), "Ablation")
    for label, key in (
        ("Model B-Embedding", "embedding_only"),
        ("Model B-Physical", "physical_only"),
        ("Model B-Full", "full_model"),
    ):
        row = _line(section, label)
        metrics = ABLATION[key]
        assert f"{metrics['spearman']:.4f}" in row and f"{metrics['mse']:.4f}" in row
    row = _line(section, "Full vs Embedding-only ΔSpearman")
    assert "[-0.0009, +0.0050]" in row


@pytest.mark.parametrize(
    "ci, verdict",
    [
        ([-0.0009, 0.0050], "0 포함"),
        ([0.0, 0.0050], "0 포함"),  # an endpoint on zero still includes zero
        ([0.0010, 0.0050], "0 미포함"),
        ([-0.0050, -0.0010], "0 미포함"),
    ],
)
def test_report_ablation_states_whether_each_ci_crosses_zero(export_dir, ci, verdict):
    _edit_cis(export_dir, delta_spearman_full_vs_embedding_only=ci)
    section = _section(_report(export_dir), "Ablation")
    assert verdict in _line(section, "Full vs Embedding-only ΔSpearman")
    # The untouched CIs keep their own verdicts.
    assert "0 포함" in _line(section, "Full vs Embedding-only ΔMSE")
    assert "0 미포함" in _line(section, "Full vs Physical-only ΔSpearman")


def test_report_ablation_interpretation_flips_with_the_embedding_ci(export_dir):
    line = _line(_section(_report(export_dir), "Ablation"), "해석")
    assert "유의한 차이 없음" in line

    _edit_cis(export_dir, delta_spearman_full_vs_embedding_only=[0.0010, 0.0050])
    line = _line(_section(_report(export_dir), "Ablation"), "해석")
    assert "유의한 차이 없음" not in line
    assert "ΔSpearman" in line and "더 좋음" in line


def test_report_ablation_reads_a_higher_full_mse_as_worse(export_dir):
    _edit_cis(export_dir, delta_mse_full_vs_embedding_only=[0.0001, 0.0003])
    line = _line(_section(_report(export_dir), "Ablation"), "해석")
    assert "ΔMSE" in line and "더 나쁨" in line


def test_report_physical_shap_share_is_computed_from_the_arrays(export_dir):
    expected = aggregate_physical_contribution_ratio(
        _np_arrays(export_dir)["shap_values"]
    )
    assert 0 < expected < 1
    section = "\n".join(_section(_report(export_dir), "Physical-feature SHAP share"))
    assert f"{expected:.4%}" in section
    assert "50.0000%" not in section  # not the stale summary value
    assert "해석" in section


def _mismatch(text):
    return _section(text, "Mismatch sensitivity")


def test_report_single_mismatch_medians_by_region_for_both_models(session):
    section = _mismatch(session.report())
    for region in ("Distal", "Seed", "PAM"):
        row = _line(section, f"single {region}")
        assert f"{SINGLE_MIN_A[region]:+.3f}" in row
        assert f"{SINGLE_MIN_B[region]:+.3f}" in row


def test_report_single_mismatch_line_names_the_computed_region(session):
    section = _mismatch(session.report())
    line_a = _line(section, "해석(single): Model A")
    assert "PAM" in line_a and "Seed 민감 가설과 불일치" in line_a
    line_b = _line(section, "해석(single): Model B")
    assert "Seed" in line_b and "Seed 민감 가설과 일치" in line_b


def test_report_single_mismatch_uses_medians_not_means(export_dir):
    def outlier(arrays):
        # One sample's huge Distal drop would dominate a mean.
        single = arrays["mismatch_single_min_model_a"]
        single[0, DISTAL_REGION_START:DISTAL_REGION_END] = -50.0

    _edit_arrays(export_dir, outlier)
    section = _mismatch(_report(export_dir))
    assert f"{SINGLE_MIN_A['Distal']:+.3f}" in _line(section, "single Distal")
    assert "PAM" in _line(section, "해석(single): Model A")


def test_report_single_mismatch_line_follows_the_data(export_dir):
    def seed_heavy(arrays):
        single = arrays["mismatch_single_min_model_a"]
        single[:, SEED_REGION_START:SEED_REGION_END] = -0.9

    _edit_arrays(export_dir, seed_heavy)
    line = _line(_mismatch(_report(export_dir)), "해석(single): Model A")
    assert "Seed" in line and "Seed 민감 가설과 일치" in line


def test_report_single_mismatch_without_any_drop(export_dir):
    def flat(arrays):
        arrays["mismatch_single_min_model_b"][:] = 0.1

    _edit_arrays(export_dir, flat)
    line = _line(_mismatch(_report(export_dir)), "해석(single): Model B")
    assert "감소 없음" in line


def test_report_complex_mismatch_medians_and_line(session):
    section = _mismatch(session.report())
    for region, a, b in (
        ("Seed", -0.02, -0.04),
        ("Distal", -0.01, -0.01),
        ("Intermittent", -0.03, -0.02),
    ):
        row = _line(section, f"complex {region}")
        assert f"{a:+.3f}" in row and f"{b:+.3f}" in row
    line_a = _line(section, "해석(complex): Model A")
    assert "Intermittent" in line_a and "Seed 민감 가설과 불일치" in line_a
    line_b = _line(section, "해석(complex): Model B")
    assert "Seed" in line_b and "Seed 민감 가설과 일치" in line_b


def test_report_complex_mismatch_uses_medians_not_means(export_dir):
    def outlier(arrays):
        # One sample's huge Distal drop would dominate a mean.
        arrays["mismatch_complex_model_b"][0, 5:10] = -50.0

    _edit_arrays(export_dir, outlier)
    section = _mismatch(_report(export_dir))
    assert f"{-0.01:+.3f}" in _line(section, "complex Distal").split()[-1]
    assert "Seed" in _line(section, "해석(complex): Model B")


def test_report_mismatch_line_prints_the_region_spread(session):
    line = _line(_mismatch(session.report()), "해석(single): Model A")
    spread = SINGLE_MIN_A["Distal"] - SINGLE_MIN_A["PAM"]
    assert f"region 간 차이 {spread:.3f}" in line


def test_report_ig_share_averaged_over_case_studies(session):
    text = session.report()
    header = _line(text.splitlines(), "== IG attribution share")
    assert f"Case Study {len(CASES)}개" in header
    section = _section(text, "IG attribution share")
    # Every synthetic case carries IG_A / IG_B, so the mean is their own share.
    assert _line(section, "Model A").split()[2:6] == [
        "0.250",
        "0.250",
        "0.500",
        "0.000",
    ]
    assert _line(section, "Model B").split()[2:6] == [
        "0.000",
        "0.400",
        "0.000",
        "0.600",
    ]
    assert "Model A가 PAM+Seed" in _line(section, "해석")


def test_report_ig_line_names_the_model_actually_concentrating_more(export_dir):
    pam_only = np.zeros(36)
    pam_only[PAM_REGION_START] = 1.0

    def edit(summary):
        for case in summary["case_studies"]:
            ig = case["integrated_gradients"]
            ig["model_b_phase4_projected_norm_attr_36bp"] = pam_only.tolist()

    _edit_summary(export_dir, edit)
    line = _line(_section(_report(export_dir), "IG attribution share"), "해석")
    assert "Model B가 PAM+Seed" in line


def test_report_closes_with_the_export_handoff(session):
    text = session.report()
    section = _section(text, "Handoff")
    joined = "\n".join(section)
    assert "model_analysis_summary.json" in joined
    assert "model_analysis_arrays.npz" in joined
    assert text.rstrip().endswith(section[-1])  # the report's last section


# ---------------------------------------------------------------------------
# Live-recomputed scores (issue #27)
# ---------------------------------------------------------------------------


class FakePredictor:
    """Deterministic stand-in for Model_A_Predictor / Model_B_Predictor."""

    def __init__(self, scores_by_sequence):
        self._scores = scores_by_sequence
        self.calls = []

    def predict(self, sequences):
        self.calls.append(list(sequences))
        return np.array([self._scores[s] for s in sequences])


class ExplodingPredictor:
    def predict(self, sequences):
        raise AssertionError("cached-only mode must never call a predictor")


def _live_session(export_dir, live_a, live_b):
    predictors = Predictors(
        model_a=FakePredictor({SEQ: live_a}),
        model_b=FakePredictor({SEQ: live_b}),
        model_b_xai=ExplodingPredictor(),
    )
    return XAISession(
        export_dir, export_dir / "test_metadata.csv", predictors
    ), predictors


def _model_line(text, model):
    return next(ln for ln in text.splitlines() if ln.startswith(model))


def test_session_with_predictors_is_live(export_dir):
    session, _ = _live_session(export_dir, 0.55, 0.88)
    assert not session.cached_only


def test_explain_shows_live_scores_next_to_cached_with_a_match(export_dir):
    session, predictors = _live_session(export_dir, 0.55, 0.88)  # P01's cached pred_raw
    text = session.explain("P01")
    line_a, line_b = _model_line(text, "Model A"), _model_line(text, "Model B")
    assert "cached 0.550" in line_a and "live 0.550" in line_a
    assert "cached 0.880" in line_b and "live 0.880" in line_b
    assert "cached error 0.350" in line_a  # the error is the cached one
    for line in (line_a, line_b):
        assert "[일치]" in line
        assert re.search(r"\d+\.\d+s", line)  # elapsed time
    assert predictors.model_a.calls == [[SEQ]]
    assert predictors.model_b.calls == [[SEQ]]


def test_explain_flags_a_live_score_that_differs_from_the_cache(export_dir):
    session, _ = _live_session(export_dir, 0.60, 0.88)
    text = session.explain("P01")
    line_a = _model_line(text, "Model A")
    assert "live 0.600" in line_a
    assert "[불일치" in line_a and "+0.0500" in line_a
    assert "[일치]" in _model_line(text, "Model B")


def test_live_match_tolerates_float_noise_below_the_tolerance(export_dir):
    session, _ = _live_session(export_dir, 0.55 + 1e-5, 0.88 - 1e-5)
    text = session.explain("P01")
    assert "[일치]" in _model_line(text, "Model A")
    assert "[일치]" in _model_line(text, "Model B")


def test_cached_only_explain_labels_scores_cached_and_shows_no_live_column(session):
    text = session.explain("P01")
    for model in ("Model A", "Model B"):
        line = _model_line(text, model)
        assert "cached" in line
        assert "live" not in line


# ---------------------------------------------------------------------------
# Typed 36bp sequences (issue #28)
# ---------------------------------------------------------------------------

# Testset row 2 is the one non-Case-Study row (see `_sequences`).
TESTSET_ONLY_SAMPLE = 2
TESTSET_ONLY_SEQ = _sequences()[TESTSET_ONLY_SAMPLE]
NOVEL_SEQ = "GGGGACGTACGTACGTACGTACGTACGGAGTACGTA"  # valid PAM, not in the Testset
# Holds the checked-in Testset and export; read only by the PAM validation tests.
CHECKED_IN_DIR = Path(__file__).parent


class DifferentiableModelA:
    """Linear stand-in for Model_A_Predictor: score = sum(w * one_hot)."""

    def __init__(self):
        self.device = torch.device("cpu")
        self._weights = torch.linspace(-1.0, 1.0, 36 * 4).reshape(36, 4)
        self.forward_calls = 0
        self.predict_calls = []

    def encode_one_hot(self, sequences):
        index = {"A": 0, "C": 1, "G": 2, "T": 3}
        one_hot = torch.zeros(len(sequences), 36, 4)
        for i, seq in enumerate(sequences):
            for pos, base in enumerate(seq):
                one_hot[i, pos, index[base]] = 1.0
        return one_hot

    def model(self, x):
        self.forward_calls += 1
        return (x * self._weights).sum(dim=(1, 2))

    def predict(self, sequences):
        self.predict_calls.append(list(sequences))
        with torch.no_grad():
            return (
                (self.encode_one_hot(sequences) * self._weights).sum(dim=(1, 2)).numpy()
            )


class DifferentiableModelBXAI:
    """Linear stand-in for Model_B_XAIPredictor over (batch, 7, 1280) embeddings."""

    def __init__(self):
        self.nt_model = torch.nn.Linear(1, 1)  # IG freezes its parameters
        self._weights = torch.linspace(-1.0, 1.0, 7 * HIDDEN_DIM).reshape(7, HIDDEN_DIM)
        self.forward_calls = 0

    def get_input_embeddings(self, sequences):
        embeddings = torch.ones(len(sequences), 7, HIDDEN_DIM)
        for i, seq in enumerate(sequences):
            for token in range(1, 7):
                kmer = seq[(token - 1) * 6 : token * 6]
                embeddings[i, token] *= 1 + kmer.count("G")
        mask = torch.ones(len(sequences), 7, dtype=torch.long)
        return embeddings.requires_grad_(True), mask

    def classify_from_input_embeddings(self, embeddings, attention_mask):
        self.forward_calls += 1
        return (embeddings * self._weights).sum(dim=(1, 2))


# Tree SHAP contributions the fake XGBoost returns: Token 3 pushes the score
# down, GC up; the trailing column is the bias term `compute_tree_shap` drops.
LIVE_SHAP = np.zeros(8964 + 1, dtype=np.float32)
LIVE_SHAP[3 * HIDDEN_DIM] = -0.3
LIVE_SHAP[8960 + 3] = 0.2
LIVE_SHAP[-1] = 0.5
LIVE_PHYSICAL = np.array([[-7.5, -70.25, 66.0, 47.0]], dtype=np.float32)


class FakeBooster:
    def predict(self, dmatrix, pred_contribs=False):
        assert pred_contribs
        assert dmatrix.num_col() == 8964
        return LIVE_SHAP[None, :]


class FakeXGB:
    def get_booster(self):
        return FakeBooster()


class FakeModelB:
    """Stand-in for Model_B_Predictor: predict() plus the feature pipeline
    live Tree SHAP needs."""

    def __init__(self, score=0.42):
        self._score = score
        self.xgb_model = FakeXGB()
        self.predict_calls = []

    def predict(self, sequences):
        self.predict_calls.append(list(sequences))
        return np.full(len(sequences), self._score)

    def extract_nt_embeddings(self, sequences):
        return np.zeros((len(sequences), 8960), dtype=np.float32)

    def compute_physical_features(self, sequences):
        return np.repeat(LIVE_PHYSICAL, len(sequences), axis=0)


def _typed_session(export_dir, **kwargs):
    predictors = Predictors(
        DifferentiableModelA(), FakeModelB(), DifferentiableModelBXAI()
    )
    session = XAISession(
        export_dir, export_dir / "test_metadata.csv", predictors, **kwargs
    )
    return session, predictors


def _position_row_or_none(section, pos):
    return next((ln for ln in section if ln.split()[:1] == [str(pos)]), None)


@pytest.mark.parametrize(
    "sequence, message",
    [
        (NOVEL_SEQ[:-1], "36"),  # too short
        (NOVEL_SEQ + "A", "36"),  # too long
        (NOVEL_SEQ[:5] + "N" + NOVEL_SEQ[6:], "A/C/G/T"),
        (NOVEL_SEQ[:10] + "U" + NOVEL_SEQ[11:], "A/C/G/T"),
        (NOVEL_SEQ[:27] + "C" + NOVEL_SEQ[28:], "PAM"),  # N N [G] R R N
        (NOVEL_SEQ[:28] + "T" + NOVEL_SEQ[29:], "PAM"),  # first R must be A/G
        (NOVEL_SEQ[:29] + "C" + NOVEL_SEQ[30:], "PAM"),  # second R must be A/G
    ],
)
def test_invalid_typed_sequences_raise_a_clear_error(export_dir, sequence, message):
    session, predictors = _typed_session(export_dir)
    with pytest.raises(XAISessionError, match=message):
        session.explain(sequence)
    assert predictors.model_a.predict_calls == []
    assert predictors.model_b.predict_calls == []


@pytest.mark.parametrize("base", "ACG")
def test_any_base_at_the_last_pam_position_is_accepted(export_dir, base):
    # The Testset's PAM window is NNGRRN: position 30 is not fixed (issue #31).
    sequence = NOVEL_SEQ[:30] + base + NOVEL_SEQ[31:]
    session, predictors = _typed_session(export_dir)
    session.explain(sequence)
    assert predictors.model_a.predict_calls == [[sequence]]


@pytest.fixture(scope="module")
def checked_in_session():
    return XAISession(
        CHECKED_IN_DIR / "final_analysis_result", CHECKED_IN_DIR / "test_metadata.csv"
    )


def test_every_real_testset_sequence_passes_validation(checked_in_session):
    sequences = pd.read_csv(CHECKED_IN_DIR / "test_metadata.csv")["sequence"]
    assert len(sequences) == 514
    for sequence in sequences:
        checked_in_session.explain(sequence)  # raises XAISessionError if refused


def test_every_real_case_study_sequence_passes_validation(checked_in_session):
    summary_path = (
        CHECKED_IN_DIR / "final_analysis_result" / "model_analysis_summary.json"
    )
    cases = json.loads(summary_path.read_text(encoding="utf-8"))["case_studies"]
    assert len(cases) == 15
    for case in cases:
        checked_in_session.explain(case["sequence"])


def test_invalid_typed_sequence_is_rejected_in_cached_only_mode_too(session):
    with pytest.raises(XAISessionError, match="PAM"):
        session.explain(NOVEL_SEQ[:27] + "C" + NOVEL_SEQ[28:])


@pytest.mark.parametrize(
    "sequence",
    [NOVEL_SEQ[:6] + " " + NOVEL_SEQ[7:], NOVEL_SEQ[:6] + "1" + NOVEL_SEQ[7:]],
)
def test_a_pasted_sequence_with_a_stray_character_gets_a_sequence_error(
    session, sequence
):
    with pytest.raises(XAISessionError, match="A/C/G/T"):
        session.explain(sequence)


def test_a_wrong_length_error_states_the_typed_length(session):
    with pytest.raises(XAISessionError, match="35"):
        session.explain(NOVEL_SEQ[:-1])


def test_lowercase_typed_sequence_is_accepted(export_dir):
    session, predictors = _typed_session(export_dir)
    text = session.explain(NOVEL_SEQ.lower())
    assert NOVEL_SEQ in text
    assert predictors.model_a.predict_calls == [[NOVEL_SEQ]]
    assert predictors.model_b.predict_calls == [[NOVEL_SEQ]]


def test_typed_sequence_matching_the_testset_shows_sample_id_and_true_score(
    export_dir,
):
    session, _ = _typed_session(export_dir)
    text = session.explain(TESTSET_ONLY_SEQ)
    match_line = _line(text.splitlines(), "Testset")
    assert f"sample_id {TESTSET_ONLY_SAMPLE}" in match_line
    assert f"true score {TRUE_SCORES[TESTSET_ONLY_SAMPLE]:.3f}" in match_line


def test_typed_sequence_without_a_testset_match_states_no_ground_truth(export_dir):
    session, _ = _typed_session(export_dir)
    text = session.explain(NOVEL_SEQ)
    match_line = _line(text.splitlines(), "Testset")
    assert "ground truth" in match_line
    assert "sample_id" not in text
    assert "true score" not in text


def test_typed_sequence_scores_are_live_with_elapsed_time(export_dir):
    session, predictors = _typed_session(export_dir)
    text = session.explain(NOVEL_SEQ)
    expected_a = float(predictors.model_a.predict([NOVEL_SEQ])[0])
    line_a, line_b = _model_line(text, "Model A"), _model_line(text, "Model B")
    assert f"live {expected_a:.3f}" in line_a
    assert "live 0.420" in line_b
    for line in (line_a, line_b):
        assert "cached" not in line
        assert re.search(r"\(\d+\.\d+s\)", line)


def test_typed_testset_sequence_shows_the_live_error_against_the_true_score(
    export_dir,
):
    session, _ = _typed_session(export_dir)
    text = session.explain(TESTSET_ONLY_SEQ)
    error = abs(0.42 - TRUE_SCORES[TESTSET_ONLY_SAMPLE])
    assert f"error {error:.3f}" in _model_line(text, "Model B")


def test_typed_sequence_live_ig_table_matches_the_cached_layout(export_dir):
    session, _ = _typed_session(export_dir)
    section = _section(session.explain(NOVEL_SEQ), "Integrated Gradients")
    for pos in range(36):
        row = _position_row(section, pos).split()
        assert row[1] == NOVEL_SEQ[pos]
    assert _position_row(section, 26).split()[2] == "PAM"
    assert any(ln.startswith("Model A share") for ln in section)
    assert any(ln.startswith("Model B share") for ln in section)
    assert any("Deterministic Projection Rule" in ln for ln in section)


def test_typed_sequence_live_ig_is_l1_normalized_and_projected(export_dir):
    session, _ = _typed_session(export_dir)
    section = _section(session.explain(NOVEL_SEQ), "Integrated Gradients")
    rows = [_position_row(section, pos).split() for pos in range(36)]
    attr_a = np.array([float(r[3]) for r in rows])
    attr_b = np.array([float(r[4]) for r in rows])
    assert np.abs(attr_a).sum() == pytest.approx(1.0, abs=0.02)
    assert np.abs(attr_b).sum() == pytest.approx(1.0, abs=0.02)
    # Deterministic Projection Rule: constant within each 6-mer token block.
    for block in attr_b.reshape(6, 6):
        assert np.all(block == block[0])


def test_live_ig_prints_the_default_step_count_and_elapsed_time(export_dir):
    session, predictors = _typed_session(export_dir)
    text = session.explain(NOVEL_SEQ)
    assert "steps=50" in _line(text.splitlines(), "== Integrated Gradients")
    joined = "\n".join(_section(text, "Integrated Gradients"))
    assert re.search(r"Model A IG \d+\.\d+s", joined)
    assert re.search(r"Model B IG \d+\.\d+s", joined)
    assert predictors.model_a.forward_calls == 50
    assert predictors.model_b_xai.forward_calls == 50


def test_live_ig_honors_a_lowered_step_count(export_dir):
    session, predictors = _typed_session(export_dir, ig_steps=5)
    text = session.explain(NOVEL_SEQ)
    assert "steps=5" in _line(text.splitlines(), "== Integrated Gradients")
    assert predictors.model_a.forward_calls == 5
    assert predictors.model_b_xai.forward_calls == 5


@pytest.mark.parametrize("steps", [0, -3])
def test_a_non_positive_ig_step_count_is_rejected(export_dir, steps):
    with pytest.raises(XAISessionError, match="IG"):
        _typed_session(export_dir, ig_steps=steps)


def test_default_ig_steps_matches_the_ig_module():
    from integrated_gradients import DEFAULT_IG_STEPS

    assert xai_session.DEFAULT_IG_STEPS == DEFAULT_IG_STEPS


def test_typed_sequence_live_token_grouped_shap(export_dir):
    session, _ = _typed_session(export_dir)
    text = session.explain(NOVEL_SEQ)
    header = _line(text.splitlines(), "== Token-grouped SHAP")
    assert "live" in header
    assert re.search(r"\d+\.\d+s", header)
    rows = [
        ln for ln in _section(text, "Token-grouped SHAP") if ln.startswith(("+", "-"))
    ]
    assert len(rows) == SHAP_GROUP_COUNT
    values = [float(ln.split()[0]) for ln in rows]
    assert sum(values) == pytest.approx(float(LIVE_SHAP[:-1].sum()), abs=1e-3)
    assert rows[0].startswith("-0.3000") and NOVEL_SEQ[12:18] in rows[0]  # Token 3
    assert rows[1].startswith("+0.2000") and "GC: 47.00" in rows[1]


def test_typed_sequence_live_explanation_has_no_cached_sections(export_dir):
    session, _ = _typed_session(export_dir)
    text = session.explain(NOVEL_SEQ)
    assert "== ISM" not in text
    assert "== Complex mismatch" not in text


def test_cached_only_refuses_a_typed_sequence_not_in_the_testset(session):
    with pytest.raises(XAISessionError, match="cached") as excinfo:
        session.explain(NOVEL_SEQ)
    assert "Testset" in str(excinfo.value)


def test_cached_only_answers_a_testset_sequence_from_the_cache(session):
    text = session.explain(TESTSET_ONLY_SEQ.lower())
    match_line = _line(text.splitlines(), "Testset")
    assert f"sample_id {TESTSET_ONLY_SAMPLE}" in match_line
    assert f"true score {TRUE_SCORES[TESTSET_ONLY_SAMPLE]:.3f}" in match_line
    line_b = _model_line(text, "Model B")
    assert "cached" in line_b and f"{PRED_B[TESTSET_ONLY_SAMPLE]:.3f}" in line_b
    assert "live" not in text
    for title in ("Token-grouped SHAP", "ISM", "Complex mismatch"):
        assert _section(text, title), title


def test_cached_only_testset_sequence_states_ig_exists_only_for_case_studies(
    session,
):
    ig = _section(session.explain(TESTSET_ONLY_SEQ), "Integrated Gradients")
    assert f"Case Study {len(CASES)}개" in " ".join(ig)
    assert not any(_position_row_or_none(ig, pos) for pos in range(36))


def test_cached_only_testset_sequence_uses_cached_shap_for_its_sample(session):
    text = session.explain(TESTSET_ONLY_SEQ)
    rows = [
        ln for ln in _section(text, "Token-grouped SHAP") if ln.startswith(("+", "-"))
    ]
    assert len(rows) == SHAP_GROUP_COUNT
    assert all(float(ln.split()[0]) == 0 for ln in rows)  # row 2 has zero SHAP
    assert "MFE: -10.90" in " ".join(rows)  # physical value from the predictions CSV


def test_cached_only_testset_sequence_of_a_case_study_points_to_its_case(session):
    ig = _section(session.explain(SEQ), "Integrated Gradients")
    assert "CONCORDANT_C01" in " ".join(ig)  # sample_id 0, the first SEQ row
