"""Reader checks against the real prepared data; skipped where it is absent.

These are the one check the synthetic fixtures cannot give: that the reader's
profile definitions, head selection and row join reproduce the figures the
analyses themselves recorded. The bulk data is outside Git, so everywhere else
these skip and the fixture tests carry the defence.
"""

from __future__ import annotations

import numpy as np
import pytest

from xai_test.nt_xai_demo import reader as rd
from xai_test.nt_xai_demo import spcas9_views


def _reader(cls, *needed: str):
    data = rd.DEFAULT_DATA_DIR
    if not (data / "manifest.json").exists():
        pytest.skip("실제 데모 자료 없음: `prepare`를 먼저 실행")
    for name in needed:
        if not (data / name).exists():
            pytest.skip(f"git 밖 bulk 자료 없음: {name}")
    return cls(data)


def test_spcas9_reader_reproduces_the_recorded_generalisation_token_profile():
    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")
    head = spcas9.primary_head
    recorded = spcas9.validation_figures("generalisation", head)

    rows = sorted(spcas9.ig_subset_rows("generalisation"))
    profiles = spcas9.ism_profiles(rows, head)
    tokens = rd.aggregate_to_tokens(profiles, spcas9.token_boundaries).mean(axis=0)

    assert len(rows) == recorded["rows"]
    assert tokens == pytest.approx(recorded["ism_token_profile"], rel=1e-6)
    # The PAM-bearing token is the most sensitive one, at this layer and head.
    assert int(np.argmax(tokens)) == len(spcas9.token_names) - 1


def test_spcas9_reader_reproduces_the_recorded_pam_library_token_profile():
    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")
    head = spcas9.primary_head
    recorded = spcas9.validation_figures("pam_library", head)

    rows = sorted(spcas9.ig_subset_rows("pam_library"))
    profiles = spcas9.ism_profiles(rows, head)
    tokens = rd.aggregate_to_tokens(profiles, spcas9.token_boundaries).mean(axis=0)

    assert tokens == pytest.approx(recorded["ism_token_profile"], rel=1e-6)
    assert not np.allclose(
        recorded["ism_token_profile"],
        spcas9.validation_figures("generalisation", head)["ism_token_profile"],
    )


def test_spcas9_ism_and_ig_populations_are_all_test_rows():
    """The measured premise behind showing the split role as a cheap safeguard."""
    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")

    for axis in (spcas9.ism_rows, spcas9.ig_rows):
        roles = {spcas9.split_role(int(row_id)) for row_id in axis.ids}
        assert roles == {"test"}, (axis.source, roles)


def test_spcas9_wt_prediction_matches_the_predictions_table_row_by_row():
    """ISM's own WT prediction is the check that catches a mis-join."""
    spcas9 = _reader(rd.SpCas9Reader, "spcas9/v3_ism/ism.npz")
    head = spcas9.primary_head

    sample = [int(row_id) for row_id in spcas9.ism_rows.ids[::400]]
    for row_id in sample:
        assert spcas9.wt_prediction(row_id, head) == pytest.approx(
            spcas9.prediction(row_id, head), rel=1e-5, abs=1e-6
        )


def test_cas12a_reader_reproduces_the_recorded_position_profile():
    cas12a = _reader(rd.Cas12aReader, "cas12a/ism.npz")
    recorded = cas12a.position_profile("all")

    rows = [int(row_id) for row_id in cas12a.ism_rows.ids]
    profiles = np.stack([cas12a.ism_profile(row_id) for row_id in rows])

    assert len(rows) == recorded["n_available"]
    assert profiles.mean(axis=0) == pytest.approx(
        recorded["mean_abs_raw_delta"], rel=1e-9
    )
    # Position 7, the last base of the 5' PAM (4-7), is the most sensitive.
    assert int(np.argmax(profiles.mean(axis=0))) == 7


def test_cas12a_labels_and_predictions_are_fractions_and_the_claim_limit_is_present():
    cas12a = _reader(rd.Cas12aReader)

    row = int(cas12a.prediction_rows.ids[0])
    record = cas12a.prediction(row)

    assert 0.0 <= record["label"] <= 1.5
    assert cas12a.metrics_for("all")["count"] == 1500
    assert (
        cas12a.metrics_for("TTTV")["count"] + cas12a.metrics_for("TTTT")["count"]
        == 1500
    )
    assert cas12a.rollout_degeneracy("all") == {
        "near_identical_query_count": 1500,
        "n_accepted": 1500,
    }
    assert "[범위와 한계]" in cas12a.claim_limit
    assert "분모" in cas12a.claim_limit


def test_spcas9_claim_limit_is_one_statement_across_its_artifacts():
    spcas9 = _reader(rd.SpCas9Reader)

    assert "physical DNA/protein" in spcas9.claim_limit
    assert spcas9.analysed_heads == ("SpCas9", "SpCas9-NG", "evoCas9")
    assert spcas9.ism_heads.width == 9 and spcas9.ig_heads.width == 3


def test_the_real_spcas9_report_shows_only_the_three_analysed_heads():
    """The six stored-but-unanalysed heads must not reach the screen (#155).

    The synthetic fixture pins this too, but the real ISM archive is where the
    nine actually live, and the head names come from it.
    """
    spcas9 = _reader(rd.SpCas9Reader)
    report = spcas9_views.report(spcas9)

    unanalysed = set(spcas9.ism_heads.names) - set(spcas9.analysed_heads)
    assert len(unanalysed) == 6
    for head in unanalysed:
        assert head not in report, head
    for head in spcas9.analysed_heads:
        assert head in report, head


def test_the_real_spcas9_report_quotes_the_per_head_floor_not_the_pooled_one():
    spcas9 = _reader(rd.SpCas9Reader)
    report = spcas9_views.report(spcas9)
    pooled = float(spcas9.noise_floor["pooled"]["median"])

    for head in spcas9.analysed_heads:
        own = float(spcas9.noise_floor_by_head(head)["median"])
        assert f"{own:.4g}" in report, head
    assert f"{pooled:.4g}" not in report
