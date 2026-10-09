"""Tests for the XAI demo session (issues #23, #24, #25).

Drives only the session's public operations against a small synthetic
export directory built in the test - no real model or real export is read.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from mismatch_profiling import (
    DISTAL_REGION_END,
    DISTAL_REGION_START,
    PAM_REGION_END,
    PAM_REGION_START,
    SEED_REGION_END,
    SEED_REGION_START,
)
from paired_bootstrap import compute_metrics
from shap_grouping import HIDDEN_DIM, SHAP_GROUP_COUNT
from xai_session import XAISession, XAISessionError

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
    return {
        "ism_delta_model_a": ism_a,
        "ism_delta_model_b": ism_b,
        "mismatch_complex_model_a": complex_a,
        "mismatch_complex_model_b": complex_b,
        "shap_values": shap,
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
        "global_evaluation": {},
        "complex_mismatch_metadata": _complex_metadata(),
        "case_studies": [_case_entry(*c) for c in CASES],
    }
    (tmp_path / "model_analysis_summary.json").write_text(json.dumps(summary))
    np.savez(tmp_path / "model_analysis_arrays.npz", **_arrays(n))
    pd.DataFrame({"sample_id": range(n), "pred_raw": PRED_B}).to_csv(
        tmp_path / "model_b_testset_predictions.csv", index=False
    )
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
