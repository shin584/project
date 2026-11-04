"""Tests for v3's cross-head attribution agreement (#137, AC8).

Three groups:

- **Rank statistics** — `rank_rows` / `spearman_rows` are checked against
  `scipy.stats` on random data with ties, because the module implements them
  itself (row-vectorised over 3,851 rows x 3 head pairs) rather than looping
  scipy per row.
- **Reporting discipline** — the constraints #137 states as acceptance
  criteria: the two strata are never summed, the library stratum carries its
  "5 targets" qualifier and a per-target breakdown, single-head coverage is
  named apart from the intersection, the head pairs and position range that
  form the agreement's denominator are in the artifact, and rollout is not
  used as an input (ADR 0008).
- **Real artifact** — against the committed `ism.npz`, skipped when it or the
  split is absent since both are git-ignored.

`test_identical_heads_do_not_separate` is the AC's "분리되지 않으면 그 자체가
결과다" written as a test: a run where the two heads' PAM profiles coincide
produces a report, not an exception.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import v3_agreement
import v3_predict
import v3_strata
from v3_agreement import (
    BASES,
    HEAD_PAIRS,
    NGG_RANGE,
    PAM_RANGE,
    WINDOW_RANGE,
    pair_key,
)
from v3_strata import ANALYSED_HEADS, GENERALISATION, PAM_LIBRARY

SUBTREE_ROOT = Path(v3_agreement.__file__).resolve().parent
XAI_TEST_ROOT = SUBTREE_ROOT.parent
SPLIT_CSV = XAI_TEST_ROOT / "SpCas9" / "published_split" / "test.csv"
ISM_NPZ = SUBTREE_ROOT / "artifacts" / "v3_ism" / "ism.npz"

#: The counts the design pins. AC8 turns on these three being different
#: numbers, so the test names them rather than deriving them.
INTERSECTION_ROWS = 3851
COVERAGE = {"SpCas9": 5021, "SpCas9-NG": 3982, "evoCas9": 5161}
GENERALISATION_ROWS = 895
LIBRARY_ROWS = 1287
LIBRARY_TARGETS = 5


# --- fixtures --------------------------------------------------------------


def sequence(*, protospacer: str = "A" * 20, pam: str = "AGG") -> str:
    return "ACGT" + protospacer + pam + "TCA"


def make_split(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_index": index,
                "sequence": row.get("sequence", sequence()),
                "in_pam_library": row.get("in_pam_library", False),
                **{head: row.get(head, 10.0) for head in v3_predict.HEADS},
            }
            for index, row in enumerate(rows)
        ]
    )


def make_delta(
    profiles: dict[str, np.ndarray],
    sequences: list[str],
    *,
    heads: tuple[str, ...] = ANALYSED_HEADS,
) -> np.ndarray:
    """A `(rows, 30, 4, 9)` delta whose per-position profile is `profiles`.

    Each position's magnitude is spread evenly over its three alternative
    bases, negative, with the reference base's slot left at exactly 0 — the
    layout `v3_ism` writes.
    """
    rows = len(sequences)
    delta = np.zeros((rows, 30, len(BASES), len(v3_predict.HEADS)), dtype=np.float32)
    for head, profile in profiles.items():
        head_axis = v3_predict.HEADS.index(head)
        broadcast = np.broadcast_to(np.asarray(profile, dtype=np.float64), (rows, 30))
        for row, text in enumerate(sequences):
            for position in range(30):
                for base_axis, base in enumerate(BASES):
                    if base != text[position]:
                        delta[row, position, base_axis, head_axis] = -broadcast[
                            row, position
                        ]
    return delta


#: The sidecar keys `v3_agreement.read_ism_sidecar` refuses to do without.
CODE_REF = "0" * 40


def write_ism(
    directory: Path,
    split: pd.DataFrame,
    delta: np.ndarray,
    *,
    sidecar: dict | None = None,
) -> Path:
    sequences = list(split["sequence"])
    reference = np.array(
        [[BASES.index(base) for base in text] for text in sequences], dtype=np.int8
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ism.npz"
    np.savez_compressed(
        path,
        delta=delta.astype(np.float32),
        wt_prediction=np.zeros((len(split), len(v3_predict.HEADS)), dtype=np.float32),
        reference_base=reference,
        row_index=split["row_index"].to_numpy(dtype=np.int64),
    )
    if sidecar is not None:
        (directory / "ism.sidecar.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    return path


def default_sidecar(**overrides) -> dict:
    payload = {
        "heads": list(v3_predict.HEADS),
        "code_ref": CODE_REF,
        "population": "analysis_intersection",
        "population_digest": "f" * 64,
        "delta_scale": "fraction",
    }
    payload.update(overrides)
    return payload


def flat_profile(value: float = 1.0) -> np.ndarray:
    return np.full(30, value)


def ramp_profile() -> np.ndarray:
    return np.linspace(0.1, 3.0, 30)


# --- rank statistics -------------------------------------------------------


def test_rank_rows_averages_ties():
    ranks = v3_agreement.rank_rows(np.array([[10.0, 20.0, 20.0, 40.0]]))
    assert ranks.tolist() == [[1.0, 2.5, 2.5, 4.0]]


def test_rank_rows_is_row_wise():
    ranks = v3_agreement.rank_rows(np.array([[1.0, 2.0], [9.0, 0.0]]))
    assert ranks.tolist() == [[1.0, 2.0], [2.0, 1.0]]


def test_rank_rows_matches_scipy_rankdata_with_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    generator = np.random.default_rng(1137)
    values = np.round(generator.normal(size=(8, 30)), 1)
    expected = [scipy_stats.rankdata(row) for row in values]
    np.testing.assert_allclose(v3_agreement.rank_rows(values), expected)


def test_spearman_rows_matches_scipy_with_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    generator = np.random.default_rng(137)
    # Rounding manufactures ties, which is where a hand-written rank is most
    # likely to drift from scipy's.
    left = np.round(generator.normal(size=(12, 30)), 1)
    right = np.round(generator.normal(size=(12, 30)), 1)
    expected = [
        scipy_stats.spearmanr(a, b).statistic for a, b in zip(left, right, strict=True)
    ]
    np.testing.assert_allclose(
        v3_agreement.spearman_rows(left, right), expected, atol=1e-12
    )


def test_spearman_rows_is_one_for_a_monotone_pair():
    values = np.arange(30.0)[None, :]
    assert v3_agreement.spearman_rows(values, 2 * values + 1)[0] == pytest.approx(1.0)
    assert v3_agreement.spearman_rows(values, -values)[0] == pytest.approx(-1.0)


def test_spearman_rows_is_nan_for_a_constant_row():
    constant = np.ones((1, 30))
    assert np.isnan(v3_agreement.spearman_rows(constant, np.arange(30.0)[None, :])[0])


def test_summarise_values_excludes_undefined_rows_and_counts_them():
    values = np.array([0.0, 0.5, 1.0, np.nan])
    summary = v3_agreement.summarise_values(values)
    assert summary["rows"] == 3
    assert summary["undefined_rows"] == 1
    assert summary["median"] == pytest.approx(0.5)
    assert all(np.isfinite(v) for v in summary.values() if isinstance(v, float))


def test_summarise_values_reports_no_total_to_add_up():
    # Guard against a "sum" or "total" key that a later reader could add
    # across strata; AC3 forbids 895 + 1,287.
    summary = v3_agreement.summarise_values(np.array([0.1, 0.2]))
    assert not [key for key in summary if "sum" in key or "total" in key]


# --- profiles --------------------------------------------------------------


def test_position_profile_is_the_mean_over_three_alternatives():
    delta = make_delta({head: flat_profile(0.3) for head in ANALYSED_HEADS}, ["A" * 30])
    profile = v3_agreement.position_profile(
        v3_agreement.select_heads(delta, v3_predict.HEADS, ANALYSED_HEADS)
    )
    assert profile.shape == (1, 30, len(ANALYSED_HEADS))
    # 0.3 at each of three alternatives; the reference slot's true zero must
    # not dilute the mean to 0.225.
    np.testing.assert_allclose(profile, 0.3, atol=1e-6)


def test_select_heads_takes_the_analysed_three_in_order():
    delta = np.zeros((2, 30, 4, len(v3_predict.HEADS)))
    for index, head in enumerate(v3_predict.HEADS):
        delta[..., index] = index
    taken = v3_agreement.select_heads(delta, v3_predict.HEADS, ANALYSED_HEADS)
    expected = [v3_predict.HEADS.index(head) for head in ANALYSED_HEADS]
    assert [taken[0, 0, 0, i] for i in range(len(ANALYSED_HEADS))] == expected


def test_select_heads_refuses_an_unknown_head():
    delta = np.zeros((1, 30, 4, len(v3_predict.HEADS)))
    with pytest.raises(ValueError, match="nope"):
        v3_agreement.select_heads(delta, v3_predict.HEADS, ("nope",))


def test_position_share_is_scale_free():
    profile = np.zeros((1, 30, 1))
    profile[0, :, 0] = 1.0
    profile[0, NGG_RANGE[0] : NGG_RANGE[1], 0] = 4.0
    share = v3_agreement.position_share(profile, NGG_RANGE)
    doubled = v3_agreement.position_share(profile * 7.0, NGG_RANGE)
    np.testing.assert_allclose(share, doubled)
    # mean over the two NGG positions / mean over the window
    window_mean = (28 * 1.0 + 2 * 4.0) / 30
    assert share[0, 0] == pytest.approx(4.0 / window_mean)


# --- pairwise agreement ----------------------------------------------------


def test_head_pairs_are_the_three_unordered_pairs():
    assert HEAD_PAIRS == (
        ("SpCas9", "SpCas9-NG"),
        ("SpCas9", "evoCas9"),
        ("SpCas9-NG", "evoCas9"),
    )
    assert len(HEAD_PAIRS) == 3


def test_head_pairs_follow_the_heads_they_are_given():
    # Every function here takes `heads`; a pair list read off a constant
    # would correlate a caller's heads against ANALYSED_HEADS' positions.
    assert v3_agreement.head_pairs() == HEAD_PAIRS
    assert v3_agreement.head_pairs(("a", "b")) == (("a", "b"),)
    assert len(v3_agreement.head_pairs(("a", "b", "c", "d"))) == 6


def test_identical_profiles_agree_perfectly():
    profiles = {head: ramp_profile() for head in ANALYSED_HEADS}
    delta = make_delta(profiles, ["A" * 30] * 4)
    profile = v3_agreement.position_profile(
        v3_agreement.select_heads(delta, v3_predict.HEADS, ANALYSED_HEADS)
    )
    agreement = v3_agreement.pair_agreement(profile, ANALYSED_HEADS, WINDOW_RANGE)
    for pair in HEAD_PAIRS:
        np.testing.assert_allclose(agreement[pair_key(*pair)], 1.0)


def test_a_reversed_head_disagrees_perfectly():
    profiles = {head: ramp_profile() for head in ANALYSED_HEADS}
    profiles["evoCas9"] = ramp_profile()[::-1]
    delta = make_delta(profiles, ["A" * 30] * 2)
    profile = v3_agreement.position_profile(
        v3_agreement.select_heads(delta, v3_predict.HEADS, ANALYSED_HEADS)
    )
    agreement = v3_agreement.pair_agreement(profile, ANALYSED_HEADS, WINDOW_RANGE)
    np.testing.assert_allclose(agreement[pair_key("SpCas9", "evoCas9")], -1.0)
    np.testing.assert_allclose(agreement[pair_key("SpCas9", "SpCas9-NG")], 1.0)


def test_pair_agreement_uses_only_the_named_position_range():
    # Two heads that agree on the PAM and disagree everywhere else.
    left, right = ramp_profile(), ramp_profile()[::-1]
    left[PAM_RANGE[0] : PAM_RANGE[1]] = [1.0, 2.0, 3.0]
    right[PAM_RANGE[0] : PAM_RANGE[1]] = [1.0, 2.0, 3.0]
    delta = make_delta(
        {"SpCas9": left, "SpCas9-NG": right, "evoCas9": left}, ["A" * 30]
    )
    profile = v3_agreement.position_profile(
        v3_agreement.select_heads(delta, v3_predict.HEADS, ANALYSED_HEADS)
    )
    key = pair_key("SpCas9", "SpCas9-NG")
    on_pam = v3_agreement.pair_agreement(profile, ANALYSED_HEADS, PAM_RANGE)[key]
    on_window = v3_agreement.pair_agreement(profile, ANALYSED_HEADS, WINDOW_RANGE)[key]
    np.testing.assert_allclose(on_pam, 1.0)
    assert on_window[0] < 0


# --- PAM separation (analysis question 2) ----------------------------------


def separation_inputs(
    spcas9_ngg_weight: float, ng_ngg_weight: float, rows: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profiles = {}
    for head, weight in (
        ("SpCas9", spcas9_ngg_weight),
        ("SpCas9-NG", ng_ngg_weight),
        ("evoCas9", 1.0),
    ):
        profile = flat_profile(1.0)
        profile[NGG_RANGE[0] : NGG_RANGE[1]] = weight
        profiles[head] = profile
    sequences = [sequence(pam="AGG")] * rows
    delta = v3_agreement.select_heads(
        make_delta(profiles, sequences), v3_predict.HEADS, ANALYSED_HEADS
    )
    profile = v3_agreement.position_profile(delta)
    return delta, profile, v3_strata.is_ngg(sequences)


def test_separated_pam_profiles_are_reported_as_separated():
    delta, profile, ngg = separation_inputs(8.0, 1.0)
    report = v3_agreement.pam_separation(delta, profile, ANALYSED_HEADS, ngg)
    assert report["direction_share"] == pytest.approx(1.0)
    assert report["separated"] is True


def test_identical_heads_do_not_separate():
    # The AC's explicit case: no separation is a finding about the shared
    # trunk, so this must produce a report rather than raise.
    delta, profile, ngg = separation_inputs(4.0, 4.0)
    report = v3_agreement.pam_separation(delta, profile, ANALYSED_HEADS, ngg)
    assert report["direction_share"] == pytest.approx(0.0)
    assert report["separated"] is False
    assert "shared trunk" in report["reading"]
    assert report["not_a_bug"] is True


def test_separation_records_its_margin_and_contrast():
    delta, profile, ngg = separation_inputs(8.0, 1.0)
    report = v3_agreement.pam_separation(delta, profile, ANALYSED_HEADS, ngg)
    assert report["contrast"] == ["SpCas9", "SpCas9-NG"]
    assert report["relative_margin"] == v3_agreement.SEPARATION_RELATIVE_MARGIN
    assert report["positions"] == {"start": 25, "stop": 27}
    # Named for what it is: a profile shape over positions 25-26, defined on
    # every row whatever its own PAM. `g_loss_share` is the NGG-only one.
    assert set(report["position_share_25_26"]) == set(ANALYSED_HEADS)


def test_separation_reports_magnitude_and_direction_as_two_claims():
    # A stratum can separate on the median while a minority of rows order the
    # other way, so one boolean covering both would hide that.
    delta, profile, ngg = separation_inputs(8.0, 1.0)
    report = v3_agreement.pam_separation(delta, profile, ANALYSED_HEADS, ngg)
    assert report["separated"] is True
    assert report["direction_consistent"] is True
    assert report["direction_consistent_share"] == (
        v3_agreement.DIRECTION_CONSISTENT_SHARE
    )

    # Half the rows each way: the magnitudes differ, the direction does not
    # hold, and the two verdicts come apart.
    mixed_delta = np.concatenate([delta, delta[:, :, :, [1, 0, 2]]], axis=0)
    mixed_profile = v3_agreement.position_profile(mixed_delta)
    mixed = v3_agreement.pam_separation(
        mixed_delta, mixed_profile, ANALYSED_HEADS, np.ones(len(mixed_delta), bool)
    )
    assert mixed["separated"] is True
    assert mixed["direction_consistent"] is False
    assert mixed["direction_share"] == pytest.approx(0.5)
    assert mixed["paired_difference"]["share_negative"] == pytest.approx(0.5)


def test_separation_also_contrasts_the_one_discriminating_position():
    # NGG and NGA differ at position 26 alone; both require G at 25.
    delta, profile, ngg = separation_inputs(8.0, 1.0)
    report = v3_agreement.pam_separation(delta, profile, ANALYSED_HEADS, ngg)
    discriminating = report["discriminating_position"]
    assert discriminating["positions"] == {"start": 26, "stop": 27}
    assert "position 26 alone" in discriminating["why"]
    assert set(discriminating["position_share"]) == set(ANALYSED_HEADS)
    assert discriminating["direction_share"] == pytest.approx(1.0)


def test_separation_g_loss_is_positive_when_losing_g_lowers_the_prediction():
    delta, profile, ngg = separation_inputs(8.0, 1.0)
    report = v3_agreement.pam_separation(delta, profile, ANALYSED_HEADS, ngg)
    # `make_delta` writes negative deltas, so every substitution lowers the
    # prediction and the loss share is positive.
    assert report["ngg_rows"] == 20
    assert report["g_loss_share"]["SpCas9"]["median"] > 0


def test_separation_on_a_stratum_without_ngg_rows_is_reported_as_absent():
    delta, profile, _ = separation_inputs(8.0, 1.0)
    ngg = np.zeros(profile.shape[0], dtype=bool)
    report = v3_agreement.pam_separation(delta, profile, ANALYSED_HEADS, ngg)
    assert report["ngg_rows"] == 0
    assert report["g_loss_share"] == {}


def test_pam_base_sensitivity_is_three_positions_by_four_bases():
    delta, _, _ = separation_inputs(8.0, 1.0)
    table = v3_agreement.pam_base_sensitivity(delta, ANALYSED_HEADS)
    assert set(table) == set(ANALYSED_HEADS)
    for head in ANALYSED_HEADS:
        assert list(table[head]) == ["24", "25", "26"]
        assert set(table[head]["24"]) == set(BASES)
    # Reference base G at 25 with the AGG PAM: its slot is a true zero.
    assert table["SpCas9"]["25"]["G"] == pytest.approx(0.0)
    assert table["SpCas9"]["25"]["A"] == pytest.approx(8.0, rel=1e-5)


# --- the report ------------------------------------------------------------


def synthetic_report(tmp_path: Path) -> dict:
    rows = (
        [
            {"sequence": sequence(protospacer="A" * 19 + base, pam="AGG")}
            for base in "ACGT"
        ]
        + [
            {
                "sequence": sequence(protospacer="C" * 20, pam=pam),
                "in_pam_library": True,
            }
            for pam in ("AGG", "AGA", "TGG")
        ]
        + [
            {
                "sequence": sequence(protospacer="G" * 20, pam=pam),
                "in_pam_library": True,
            }
            for pam in ("AGG", "AGA")
        ]
        + [
            # One row per head with that head unmeasured, so each head's own
            # coverage comes out larger than the intersection and no coverage
            # count equals it. These rows are outside the intersection and so
            # outside both strata.
            {"sequence": sequence(protospacer="T" * 20, pam="AGA"), head: np.nan}
            for head in ANALYSED_HEADS
        ]
    )
    split = make_split(rows)
    measured = split.loc[:, list(ANALYSED_HEADS)].notna().all(axis=1)
    profiles = {head: ramp_profile() for head in ANALYSED_HEADS}
    profiles["evoCas9"] = ramp_profile()[::-1]
    swept = split.loc[measured].reset_index(drop=True)
    delta = make_delta(profiles, list(swept["sequence"]))
    ism_path = write_ism(tmp_path / "v3_ism", swept, delta, sidecar=default_sidecar())
    split_csv = tmp_path / "test.csv"
    split.to_csv(split_csv, index=False)
    # The synthetic split is not the published one, so its pin is not checked.
    return v3_agreement.build_report(
        ism_npz=ism_path, split_csv=split_csv, split_checksum=None
    )


def test_report_has_one_entry_per_stratum_and_no_pooled_entry(tmp_path):
    report = synthetic_report(tmp_path)
    assert set(report["strata"]) == {GENERALISATION, PAM_LIBRARY}
    assert report["pooling"] == "forbidden"


def test_report_never_sums_the_two_strata(tmp_path):
    report = synthetic_report(tmp_path)
    rows = [report["strata"][name]["rows"] for name in (GENERALISATION, PAM_LIBRARY)]
    pooled = sum(rows)

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
        else:
            yield node

    assert pooled not in [v for v in walk(report["strata"]) if isinstance(v, int)]


def test_report_names_the_denominator_of_the_agreement(tmp_path):
    report = synthetic_report(tmp_path)
    assert report["head_pairs"] == [list(pair) for pair in HEAD_PAIRS]
    assert report["position_ranges"]["window"] == {"start": 0, "stop": 30}
    assert report["position_ranges"]["pam"] == {"start": 24, "stop": 27}
    for stratum in report["strata"].values():
        for name in ("window", "pam"):
            agreement = stratum["agreement"][name]
            assert set(agreement["per_row"]) == {pair_key(*p) for p in HEAD_PAIRS}
            assert agreement["positions"] == report["position_ranges"][name]


def test_report_marks_the_pam_range_as_too_narrow_to_rank(tmp_path):
    # Three positions can only rank to ±1, ±0.5 or 0, so the PAM question is
    # answered by `pam_separation`, not by this correlation.
    report = synthetic_report(tmp_path)
    for stratum in report["strata"].values():
        assert stratum["agreement"]["window"]["width"] == 30
        assert "width_note" not in stratum["agreement"]["window"]
        assert stratum["agreement"]["pam"]["width"] == 3
        assert stratum["agreement"]["pam"]["width_note"] == (
            v3_agreement.NARROW_RANGE_NOTE
        )
        assert stratum["pam_separation"]["rows"] == stratum["rows"]


def test_report_separates_intersection_rows_from_single_head_coverage(tmp_path):
    report = synthetic_report(tmp_path)
    coverage = report["head_coverage"]
    assert set(coverage["rows"]) == set(ANALYSED_HEADS)
    assert coverage["population"] == "own_measured_coverage"
    assert report["population"] == "analysis_intersection"
    # The whole point of naming them apart: no coverage equals the
    # intersection, so an intersection figure can never be read as a
    # coverage figure.
    assert report["rows"] not in coverage["rows"].values()
    assert report["single_head_profile"]["population"] == "analysis_intersection"


def test_report_carries_the_library_qualifier_and_per_target_breakdown(tmp_path):
    report = synthetic_report(tmp_path)
    library = report["strata"][PAM_LIBRARY]
    assert library["qualifier"] == v3_agreement.LIBRARY_QUALIFIER
    assert library["distinct_protospacers"] == 2
    assert [entry["rows"] for entry in library["per_target"]] == [3, 2]
    assert sum(entry["rows"] for entry in library["per_target"]) == library["rows"]
    # The PAM contrast is the only reason this stratum exists, so a pooled
    # verdict over its PAM variants is not a report on 5 targets.
    for entry in library["per_target"]:
        assert entry["pam_separation"]["rows"] == entry["rows"]
        assert set(entry["pam_separation"]["position_share_25_26"]) == set(
            ANALYSED_HEADS
        )
    assert GENERALISATION not in report["strata"][PAM_LIBRARY]
    assert "qualifier" not in report["strata"][GENERALISATION]


def test_report_says_what_the_base_table_zeros_mean(tmp_path):
    # Without the caveat a reader takes G reading 0.0 at 25-26 in an all-NGG
    # population as "the model ignores G", when it is the reference slot.
    report = synthetic_report(tmp_path)
    table = report["strata"][GENERALISATION]["pam_base_sensitivity"]
    assert table["positions"] == {"start": 24, "stop": 27}
    assert "true zero" in table["reference_slot_note"]
    assert set(table["by_head"]) == set(ANALYSED_HEADS)
    assert table["by_head"]["SpCas9"]["25"]["G"] == pytest.approx(0.0)


def test_report_records_the_unanalysed_heads_as_stored_only(tmp_path):
    report = synthetic_report(tmp_path)
    assert report["stored_heads"] == list(v3_predict.HEADS)
    assert set(report["stored_not_analysed"]) == set(v3_predict.HEADS) - set(
        ANALYSED_HEADS
    )
    assert report["nan_policy"] == "forbidden"


def test_report_refuses_an_ism_artifact_with_no_sidecar(tmp_path):
    rows = [{"sequence": sequence(protospacer="A" * 19 + base)} for base in "ACGT"]
    split = make_split(rows)
    delta = make_delta(
        {head: ramp_profile() for head in ANALYSED_HEADS}, list(split["sequence"])
    )
    ism_path = write_ism(tmp_path / "v3_ism", split, delta)
    split_csv = tmp_path / "test.csv"
    split.to_csv(split_csv, index=False)
    with pytest.raises(ValueError, match="no sidecar"):
        v3_agreement.build_report(
            ism_npz=ism_path, split_csv=split_csv, split_checksum=None
        )


@pytest.mark.parametrize("key", ["heads", "code_ref", "population_digest"])
def test_report_refuses_a_sidecar_that_cannot_pin_the_sweep(tmp_path, key):
    # A sweep is addressed by its code_ref and population digest, so a report
    # computed off a sidecar that omits them could not name its own source.
    rows = [{"sequence": sequence(protospacer="A" * 19 + base)} for base in "ACGT"]
    split = make_split(rows)
    delta = make_delta(
        {head: ramp_profile() for head in ANALYSED_HEADS}, list(split["sequence"])
    )
    ism_path = write_ism(
        tmp_path / "v3_ism", split, delta, sidecar=default_sidecar(**{key: None})
    )
    split_csv = tmp_path / "test.csv"
    split.to_csv(split_csv, index=False)
    with pytest.raises(ValueError, match=key):
        v3_agreement.build_report(
            ism_npz=ism_path, split_csv=split_csv, split_checksum=None
        )


def test_report_does_not_read_the_rollout(tmp_path):
    report = synthetic_report(tmp_path)
    assert report["inputs"] == ["ism"]
    assert "rollout" in report["rollout_excluded"]
    text = json.dumps(report).lower()
    assert "rollout.npz" not in text


def test_report_states_the_interpretation_limit_and_claims_no_binding(tmp_path):
    report = synthetic_report(tmp_path)
    assert report["interpretation_limit"] == v3_agreement.INTERPRETATION_LIMIT

    def keys(node, trail=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield f"{trail}/{key}".lower()
                yield from keys(value, f"{trail}/{key}")
        elif isinstance(node, list):
            for value in node:
                yield from keys(value, trail)

    forbidden = ("bind", "docking", "affinity", "contact")
    assert not [key for key in keys(report) if any(word in key for word in forbidden)]


def test_report_records_its_inputs_by_name_and_digest_only(tmp_path):
    # The report is tracked, unlike the `.npz` sidecars, so a path from the
    # machine that wrote it would put that machine in the committed bytes.
    report = synthetic_report(tmp_path)
    consumed = report["consumed"]
    assert consumed["ism"]["name"] == "ism.npz"
    assert consumed["split"]["name"] == "test.csv"
    assert len(consumed["split"]["sha256"]) == 64
    assert consumed["ism"]["sidecar"]["code_ref"] == CODE_REF
    assert consumed["split"]["split_checksum_pin_checked"] is False
    assert str(tmp_path) not in json.dumps(report)
    assert not [
        value
        for entry in (consumed["ism"], consumed["split"])
        for value in entry.values()
        if isinstance(value, str) and ("/" in value or "\\" in value)
    ]


def test_report_is_finite_everywhere(tmp_path):
    report = synthetic_report(tmp_path)

    def floats(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from floats(value)
        elif isinstance(node, list):
            for value in node:
                yield from floats(value)
        elif isinstance(node, float):
            yield node

    assert all(np.isfinite(value) for value in floats(report))


def test_report_refuses_an_ism_artifact_that_is_not_the_intersection(tmp_path):
    rows = [{"sequence": sequence(protospacer="A" * 19 + base)} for base in "ACGT"]
    split = make_split(rows)
    delta = make_delta(
        {head: ramp_profile() for head in ANALYSED_HEADS}, list(split["sequence"])
    )
    # Drop a row from the archive only: the split still has four.
    ism_path = write_ism(
        tmp_path / "v3_ism", split.iloc[:3], delta[:3], sidecar=default_sidecar()
    )
    split_csv = tmp_path / "test.csv"
    split.to_csv(split_csv, index=False)
    with pytest.raises(ValueError, match="analysis intersection"):
        v3_agreement.build_report(
            ism_npz=ism_path, split_csv=split_csv, split_checksum=None
        )


def test_write_report_round_trips(tmp_path):
    report = synthetic_report(tmp_path)
    path = v3_agreement.write_report(tmp_path / "artifacts", report)
    assert path.name == v3_agreement.AGREEMENT_NAME
    assert json.loads(path.read_text(encoding="utf-8")) == report


# --- the real artifact -----------------------------------------------------


@pytest.mark.skipif(
    not (ISM_NPZ.exists() and SPLIT_CSV.exists()),
    reason="ism.npz and test.csv are git-ignored",
)
def test_real_report_reproduces_the_pinned_populations():
    report = v3_agreement.build_report(ism_npz=ISM_NPZ, split_csv=SPLIT_CSV)
    assert report["rows"] == INTERSECTION_ROWS
    assert report["head_coverage"]["rows"] == COVERAGE
    assert INTERSECTION_ROWS not in COVERAGE.values()

    generalisation = report["strata"][GENERALISATION]
    library = report["strata"][PAM_LIBRARY]
    assert generalisation["rows"] == GENERALISATION_ROWS
    assert library["rows"] == LIBRARY_ROWS
    assert library["distinct_protospacers"] == LIBRARY_TARGETS
    assert len(library["per_target"]) == LIBRARY_TARGETS
    assert generalisation["rows"] + library["rows"] != report["rows"]

    for stratum in (generalisation, library):
        for name in ("window", "pam"):
            for key in (pair_key(*pair) for pair in HEAD_PAIRS):
                summary = stratum["agreement"][name]["per_row"][key]
                assert -1.0 <= summary["median"] <= 1.0
                assert summary["rows"] + summary["undefined_rows"] == stratum["rows"]
