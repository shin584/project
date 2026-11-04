"""Tests for the v3 IG subset manifest and case selection (#131).

Carries #129's AC7 (the frozen 400-row IG subset) and AC9 (the six-cell case
table and its non-circularity). Both read predictions and labels only, so
everything here runs locally with no checkpoint and no GPU.

The synthetic strata are built at the shape `v3_strata.join_predictions`
returns, because that is the contract these selectors consume. One pair of
tests runs the real split and the real predictions through `v3_strata.build`
to pin the counts the design commits to, and skips when those files are
absent since both are git-ignored.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import v3_predict
import v3_selection
import v3_strata
from v3_selection import ERRHI, ERRLO, HEADGAP
from v3_strata import ANALYSED_HEADS, GENERALISATION, PAM_LIBRARY, PRIMARY_HEAD

XAI_TEST_ROOT = Path(v3_strata.__file__).resolve().parent.parent
SUBTREE_ROOT = Path(v3_strata.__file__).resolve().parent
SPLIT_CSV = XAI_TEST_ROOT / "SpCas9" / "published_split" / "test.csv"
PREDICTIONS_CSV = (
    SUBTREE_ROOT / "artifacts" / "v3_testset_predictions" / "predictions.csv"
)

#: Four NGG and four non-NGG PAMs, interleaved so NGG membership is
#: independent of the error quartile the variant index drives below. Without
#: the interleave every library cell would be empty in two quartiles and the
#: balanced-allocation tests would be testing the shortfall path instead.
PAM_VARIANTS = ("AGG", "AGA", "CGG", "CGA", "GGG", "GGA", "TGG", "TGA")

NGG_PAM = "AGG"
NON_NGG_PAM = "AGA"


def protospacer(index: int) -> str:
    """A distinct 20nt target per index, stable across tests."""
    return f"{index:020d}".translate(str.maketrans("0123456789", "ACGTACGTAC"))


def window(*, target: str, pam: str) -> str:
    return "ACGT" + target + pam + "TCA"


def joined_frame(rows: list[dict]) -> pd.DataFrame:
    """A frame shaped like `v3_strata.join_predictions`'s output.

    `error` sets the primary head's absolute error and `gap` the cross-head
    spread, which is all the selectors read. The unanalysed heads are left out
    on purpose: a selector that touched one would fail here.
    """
    records = []
    for row_index, row in enumerate(rows):
        error = float(row["error"])
        gap = float(row.get("gap", 0.0))
        sequence = row["sequence"]
        record = {
            "row_index": row.get("row_index", row_index),
            "sequence": sequence,
            "protospacer": sequence[v3_strata.PROTOSPACER],
            "in_pam_library": row.get("in_pam_library", False),
        }
        for position, head in enumerate(ANALYSED_HEADS):
            # Head 0 is the primary; the last analysed head carries the gap.
            head_error = error + (gap if position == len(ANALYSED_HEADS) - 1 else 0.0)
            record[f"pred_{head}"] = 0.1
            record[f"label_{head}"] = 0.1 + head_error
            record[f"abs_error_{head}"] = head_error
        records.append(record)
    return pd.DataFrame(records)


def generalisation_frame(rows: int = 40) -> pd.DataFrame:
    """Non-library NGG rows, every one its own target.

    The gap **peaks in the middle** of the error range, so the `HEADGAP` pool
    is disjoint from both error quartiles. That is the arrangement in which
    the three criteria are genuinely three questions; the correlated case,
    where the widest gaps are also the worst errors, is what the real split
    turns out to look like and is covered on its own below.
    """
    middle = (rows - 1) / 2
    return joined_frame(
        [
            {
                "sequence": window(target=protospacer(index), pam=NGG_PAM),
                "error": index * 0.001,
                "gap": 0.2 - abs(index - middle) * 0.008,
            }
            for index in range(rows)
        ]
    )


#: The library fixture's rows start here so the two synthetic strata are
#: disjoint in `row_index`, as the real ones are: they are two cuts of one
#: split, and a fixture that overlapped would make the subset's
#: drew-a-row-twice guard unreachable.
LIBRARY_ROW_OFFSET = 1000


def library_frame(targets: int = 5) -> pd.DataFrame:
    """Five targets x eight PAM variants, the real stratum's shape.

    The variant index drives the error and the gap peaks at a middle variant,
    so every quartile and the top gap decile each span all five targets while
    staying disjoint from each other. That is what makes the
    distinct-protospacer constraint satisfiable, exactly as the design argues
    it is for the real 1,287 rows.
    """
    return joined_frame(
        [
            {
                "row_index": LIBRARY_ROW_OFFSET + variant * targets + target,
                "sequence": window(target=protospacer(target), pam=pam),
                "in_pam_library": True,
                "error": variant * 0.01 + target * 0.0005,
                "gap": 0.1 - abs(variant - 4) * 0.01 + target * 0.0001,
            }
            for variant, pam in enumerate(PAM_VARIANTS)
            for target in range(targets)
        ]
    )


def both_strata() -> dict[str, pd.DataFrame]:
    return {GENERALISATION: generalisation_frame(), PAM_LIBRARY: library_frame()}


# --- shared primitives -----------------------------------------------------


def test_stratum_prefixes_come_from_the_strata_constants():
    assert v3_selection.stratum_prefix(GENERALISATION) == "GEN"
    assert v3_selection.stratum_prefix(PAM_LIBRARY) == "LIB"


def test_stratum_prefix_refuses_an_unknown_stratum():
    with pytest.raises(ValueError, match="generalization"):
        v3_selection.stratum_prefix("generalization")


def test_error_quartiles_are_rank_based_and_equally_sized():
    frame = joined_frame(
        [
            {"sequence": window(target=protospacer(i), pam=NGG_PAM), "error": i * 0.01}
            for i in range(8)
        ]
    )
    assert v3_selection.error_quartile(frame).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_error_quartiles_break_ties_on_row_index():
    # Every row has the same error: the quartile then has to come from
    # `row_index`, or the manifest's cells shuffle between runs.
    frame = joined_frame(
        [
            {"sequence": window(target=protospacer(i), pam=NGG_PAM), "error": 0.05}
            for i in range(8)
        ]
    )
    assert v3_selection.error_quartile(frame).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]


def test_error_quartiles_read_only_the_primary_head():
    frame = generalisation_frame(8)
    shuffled = frame.copy()
    other = [h for h in ANALYSED_HEADS if h != PRIMARY_HEAD]
    shuffled[[f"abs_error_{h}" for h in other]] = 0.0
    assert v3_selection.error_quartile(frame).tolist() == (
        v3_selection.error_quartile(shuffled).tolist()
    )


def test_head_gap_is_the_spread_of_the_analysed_heads():
    frame = joined_frame(
        [
            {
                "sequence": window(target=protospacer(0), pam=NGG_PAM),
                "error": 0.02,
                "gap": 0.07,
            }
        ]
    )
    assert v3_selection.head_gap(frame) == pytest.approx([0.07])


def test_head_gap_ignores_the_unanalysed_heads():
    frame = joined_frame(
        [
            {
                "sequence": window(target=protospacer(0), pam=NGG_PAM),
                "error": 0.02,
                "gap": 0.07,
            }
        ]
    )
    frame["abs_error_HypaCas9"] = 5.0
    assert v3_selection.head_gap(frame) == pytest.approx([0.07])


# --- balanced allocation ---------------------------------------------------


def test_allocation_is_even_when_every_cell_has_capacity():
    assert v3_selection.allocate_cell_draws(
        {"a": 10, "b": 10, "c": 10, "d": 10}, 8
    ) == {
        "a": 2,
        "b": 2,
        "c": 2,
        "d": 2,
    }


def test_allocation_spends_the_remainder_in_cell_name_order():
    assert v3_selection.allocate_cell_draws({"a": 10, "b": 10, "c": 10}, 8) == {
        "a": 3,
        "b": 3,
        "c": 2,
    }


def test_allocation_redistributes_a_short_cells_shortfall():
    # `a` can only give 1, so the other two carry 4 each rather than the
    # draw silently coming back short.
    assert v3_selection.allocate_cell_draws({"a": 1, "b": 10, "c": 10}, 9) == {
        "a": 1,
        "b": 4,
        "c": 4,
    }


def test_allocation_refuses_a_total_the_cells_cannot_fill():
    with pytest.raises(ValueError, match="only 6 rows"):
        v3_selection.allocate_cell_draws({"a": 3, "b": 3}, 7)


# --- AC7: the IG subset manifest -------------------------------------------


def test_ig_subset_cells_for_the_generalisation_stratum_are_error_quartiles():
    cells = v3_selection.ig_subset_cells(generalisation_frame(8), GENERALISATION)
    assert sorted(set(cells)) == ["errq1", "errq2", "errq3", "errq4"]


def test_ig_subset_cells_for_the_library_stratum_cross_ngg_with_the_quartile():
    cells = v3_selection.ig_subset_cells(library_frame(), PAM_LIBRARY)
    assert sorted(set(cells)) == [
        "ngg-errq1",
        "ngg-errq2",
        "ngg-errq3",
        "ngg-errq4",
        "nonngg-errq1",
        "nonngg-errq2",
        "nonngg-errq3",
        "nonngg-errq4",
    ]
    # Eight equally-filled cells, which is what makes 8 x 25 = 200 reachable
    # on the real stratum.
    assert set(pd.Series(cells).value_counts()) == {5}


def test_ig_subset_is_stored_in_row_index_order():
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    stored = manifest.entries["row_index"].tolist()
    assert stored == sorted(stored)


def test_ig_subset_records_selection_order_stratum_and_cell_as_columns():
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    for column in ("selection_order", "stratum", "cell", "error_quartile", "is_ngg"):
        assert column in manifest.entries.columns
    # The selection order is a recorded field, not the storage axis: it is a
    # permutation of 1..n that does *not* follow the stored order.
    order = manifest.entries["selection_order"].tolist()
    assert sorted(order) == list(range(1, len(order) + 1))
    assert order != sorted(order)


def test_ig_subset_draws_the_budget_from_each_stratum():
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    counts = manifest.entries["stratum"].value_counts().to_dict()
    assert counts == {GENERALISATION: 8, PAM_LIBRARY: 8}


def test_ig_subset_balances_the_cells():
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    library = manifest.entries[manifest.entries["stratum"] == PAM_LIBRARY]
    assert set(library["cell"].value_counts()) == {1}
    generalisation = manifest.entries[manifest.entries["stratum"] == GENERALISATION]
    assert set(generalisation["cell"].value_counts()) == {2}


def test_ig_subset_is_reproducible_for_a_seed():
    first = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8, seed=7)
    second = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8, seed=7)
    assert first.digest == second.digest


def test_ig_subset_changes_with_the_seed():
    first = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8, seed=7)
    other = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8, seed=8)
    assert first.digest != other.digest


def test_ig_subset_draws_the_strata_independently():
    # Each stratum gets its own substream, so growing one does not reshuffle
    # the other's draw. Otherwise re-running the subset after a split fix
    # would silently move rows in the untouched stratum.
    strata = both_strata()
    grown = {**strata, PAM_LIBRARY: library_frame(targets=4)}
    first = v3_selection.select_ig_subset(strata, rows_per_stratum=8)
    second = v3_selection.select_ig_subset(grown, rows_per_stratum=8)
    assert first.rows_for(GENERALISATION).tolist() == (
        second.rows_for(GENERALISATION).tolist()
    )


def test_ig_subset_rows_are_unique():
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    rows = manifest.entries["row_index"]
    assert rows.is_unique


def test_ig_subset_refuses_an_unknown_stratum():
    with pytest.raises(ValueError, match="generalization"):
        v3_selection.select_ig_subset(
            {"generalization": generalisation_frame()}, rows_per_stratum=4
        )


def test_ig_subset_reports_a_stratum_too_small_for_the_budget():
    with pytest.raises(ValueError, match=GENERALISATION):
        v3_selection.select_ig_subset(
            {GENERALISATION: generalisation_frame(4)}, rows_per_stratum=8
        )


def test_ig_subset_default_budget_matches_the_design():
    assert v3_selection.IG_SUBSET_ROWS_PER_STRATUM == 200
    assert v3_selection.IG_SUBSET_ROWS == 400


# --- AC7: the manifest on disk ---------------------------------------------


def test_write_ig_subset_round_trips_through_load(tmp_path):
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    path = v3_selection.write_ig_subset(tmp_path, manifest)
    loaded = v3_selection.load_ig_subset(path)
    assert loaded.digest == manifest.digest
    assert loaded.seed == manifest.seed
    assert loaded.rows_for(PAM_LIBRARY).tolist() == (
        manifest.rows_for(PAM_LIBRARY).tolist()
    )


def test_written_manifest_carries_the_seed_and_the_membership_digest(tmp_path):
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8, seed=7)
    path = v3_selection.write_ig_subset(tmp_path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 7
    assert payload["membership_digest"] == manifest.digest
    assert payload["membership_digest"] == v3_predict.population_digest(
        manifest.entries["row_index"]
    )
    assert payload["rows"] == 16


def test_written_manifest_is_byte_pinned_by_its_checksum_file(tmp_path):
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    path = v3_selection.write_ig_subset(tmp_path, manifest)
    checksum = json.loads(
        (tmp_path / v3_selection.IG_SUBSET_CHECKSUM_NAME).read_text(encoding="utf-8")
    )
    assert checksum["files"][v3_selection.IG_SUBSET_NAME] == v3_predict.sha256_of(path)


def test_load_ig_subset_refuses_an_edited_manifest(tmp_path):
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    path = v3_selection.write_ig_subset(tmp_path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"] = payload["entries"][:-1]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        v3_selection.load_ig_subset(path)


def test_load_ig_subset_refuses_a_manifest_whose_digest_does_not_match(tmp_path):
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    path = v3_selection.write_ig_subset(tmp_path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["membership_digest"] = "0" * 64
    body = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(body, encoding="utf-8", newline="\n")
    # Re-pin the bytes so the failure can only come from the membership check.
    v3_predict.write_sidecar(
        tmp_path,
        v3_selection.IG_SUBSET_CHECKSUM_NAME,
        {"files": {v3_selection.IG_SUBSET_NAME: v3_predict.sha256_of(path)}},
    )
    with pytest.raises(ValueError, match="membership digest"):
        v3_selection.load_ig_subset(path)


def test_manifest_rows_for_refuses_an_unknown_stratum():
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    with pytest.raises(ValueError, match="generalization"):
        manifest.rows_for("generalization")


def test_manifest_rows_for_returns_row_index_ascending():
    manifest = v3_selection.select_ig_subset(both_strata(), rows_per_stratum=8)
    rows = manifest.rows_for(PAM_LIBRARY)
    assert rows.tolist() == sorted(rows.tolist())


# --- AC9: case selection ---------------------------------------------------


def test_cases_cover_six_cells_three_each():
    cases = v3_selection.select_cases(both_strata())
    counts = cases.groupby(["stratum", "criterion"]).size().to_dict()
    assert set(counts) == {
        (stratum, criterion)
        for stratum in (GENERALISATION, PAM_LIBRARY)
        for criterion in (ERRLO, ERRHI, HEADGAP)
    }
    assert set(counts.values()) == {3}
    assert len(cases) == 18


def test_case_ids_follow_the_stratum_criterion_number_scheme():
    cases = v3_selection.select_cases(both_strata())
    assert "GEN-ERRLO-01" in set(cases["case_id"])
    assert "LIB-HEADGAP-03" in set(cases["case_id"])
    assert cases["case_id"].is_unique


def test_errlo_cases_are_the_lowest_errors_in_the_lowest_quartile():
    cases = v3_selection.select_cases({GENERALISATION: generalisation_frame()})
    errlo = cases[cases["criterion"] == ERRLO]
    assert errlo["row_index"].tolist() == [0, 1, 2]
    assert errlo["case_id"].tolist() == ["GEN-ERRLO-01", "GEN-ERRLO-02", "GEN-ERRLO-03"]
    assert set(errlo["error_quartile"]) == {1}


def test_errhi_cases_are_the_highest_errors_in_the_highest_quartile():
    cases = v3_selection.select_cases({GENERALISATION: generalisation_frame()})
    errhi = cases[cases["criterion"] == ERRHI]
    assert errhi["row_index"].tolist() == [39, 38, 37]
    assert set(errhi["error_quartile"]) == {4}


def test_headgap_cases_come_from_the_strata_own_top_decile():
    frame = generalisation_frame()
    cases = v3_selection.select_cases({GENERALISATION: frame})
    headgap = cases[cases["criterion"] == HEADGAP]
    threshold = v3_selection.score_stratum(GENERALISATION, frame).headgap_floor
    assert (headgap["head_gap"] >= threshold).all()
    # The gap peaks mid-range in this fixture, so neither error cell can reach
    # the decile: a selector ranking on error would land on 0/1/2 or 39/38/37.
    # 19 and 20 are equidistant from the peak, so the tie breaks on row_index.
    assert headgap["row_index"].tolist() == [19, 20, 18]


def test_headgap_threshold_is_computed_within_each_stratum():
    # The two strata differ about twofold in gap scale, so one absolute
    # threshold would be a different quantile in each.
    strata = both_strata()
    thresholds = {
        name: v3_selection.score_stratum(name, frame).headgap_floor
        for name, frame in strata.items()
    }
    assert thresholds[GENERALISATION] != thresholds[PAM_LIBRARY]
    cases = v3_selection.select_cases(strata)
    for name, threshold in thresholds.items():
        selected = cases[(cases["stratum"] == name) & (cases["criterion"] == HEADGAP)]
        assert (selected["head_gap"] >= threshold).all()


def test_library_cells_take_one_row_per_distinct_protospacer():
    cases = v3_selection.select_cases({PAM_LIBRARY: library_frame()})
    for criterion in (ERRLO, ERRHI, HEADGAP):
        cell = cases[cases["criterion"] == criterion]
        assert len(cell) == 3
        assert cell["protospacer"].is_unique


def test_generalisation_cells_do_not_impose_the_protospacer_constraint():
    # Not because it would be wrong there, but because the stratum is already
    # one row per target: the constraint has to be the library's, not global,
    # or a library-shaped bug would be invisible.
    frame = generalisation_frame()
    frame["protospacer"] = protospacer(0)
    cases = v3_selection.select_cases({GENERALISATION: frame})
    assert len(cases[cases["criterion"] == ERRLO]) == 3


def correlated_frame(rows: int = 120) -> pd.DataFrame:
    """The real split's shape: the widest gaps are the worst-predicted rows.

    Measured on the real test split, two of the three `HEADGAP` extremes per
    stratum were already `ERRHI` cases. This fixture reproduces that so the
    yielding rule is tested against the arrangement that motivated it. It is
    120 rows because the rule needs a decile deep enough to still hold three
    unclaimed rows, which the real strata's ~90-row deciles are.
    """
    return joined_frame(
        [
            {
                "sequence": window(target=protospacer(index), pam=NGG_PAM),
                "error": index * 0.01,
                "gap": index * 0.01,
            }
            for index in range(rows)
        ]
    )


def test_a_row_is_a_case_only_once():
    cases = v3_selection.select_cases({GENERALISATION: correlated_frame()})
    assert cases["row_index"].is_unique
    assert len(cases) == 9


def test_headgap_yields_to_errhi_rather_than_the_other_way_round():
    """The error cells keep their extremes; `HEADGAP` takes what is left.

    A row already reported as max-error says nothing about multi-head
    disagreement, so the novel criterion is the one that moves down its rank.
    """
    cases = v3_selection.select_cases({GENERALISATION: correlated_frame()})
    errhi = cases[cases["criterion"] == ERRHI]["row_index"].tolist()
    headgap = cases[cases["criterion"] == HEADGAP]["row_index"].tolist()
    assert errhi == [119, 118, 117]
    assert headgap == [116, 115, 114]


def test_headgap_still_comes_from_the_decile_after_yielding():
    frame = correlated_frame()
    threshold = v3_selection.score_stratum(GENERALISATION, frame).headgap_floor
    cases = v3_selection.select_cases({GENERALISATION: frame})
    headgap = cases[cases["criterion"] == HEADGAP]
    assert (headgap["head_gap"] >= threshold).all()


def test_cell_overlap_is_measured_before_dedup():
    """The collision rate is a result about the model, so it is reported.

    De-duplicating without recording it would turn "cross-head disagreement
    sits in the high-error tail" into an invisible implementation detail.
    """
    overlap = v3_selection.cell_overlap({GENERALISATION: correlated_frame()})
    assert overlap[GENERALISATION][ERRLO] == 0
    assert overlap[GENERALISATION][ERRHI] == 0
    assert overlap[GENERALISATION][HEADGAP] == 3
    # The decorrelated fixture collides nowhere.
    clean = v3_selection.cell_overlap({GENERALISATION: generalisation_frame()})
    assert set(clean[GENERALISATION].values()) == {0}


def test_cases_json_records_the_overlap(tmp_path):
    strata = {GENERALISATION: correlated_frame()}
    path = v3_selection.write_cases(tmp_path, v3_selection.select_cases(strata), strata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["one_case_per_row"] is True
    assert payload["cell_overlap_before_dedup"][GENERALISATION][HEADGAP] == 3


def test_every_library_case_carries_the_five_target_qualifier():
    cases = v3_selection.select_cases(both_strata())
    library = cases[cases["stratum"] == PAM_LIBRARY]
    assert (library["reporting_qualifier"] == v3_selection.LIBRARY_QUALIFIER).all()
    generalisation = cases[cases["stratum"] == GENERALISATION]
    assert (generalisation["reporting_qualifier"] == "").all()


def test_cases_break_ties_on_row_index():
    frame = joined_frame(
        [
            {
                "sequence": window(target=protospacer(i), pam=NGG_PAM),
                "error": 0.01,
                "gap": 0.01,
            }
            for i in range(40)
        ]
    )
    cases = v3_selection.select_cases({GENERALISATION: frame})
    errlo = cases[cases["criterion"] == ERRLO]
    assert errlo["row_index"].tolist() == [0, 1, 2]


def test_cases_carry_the_labels_predictions_and_errors_they_were_chosen_on():
    cases = v3_selection.select_cases({GENERALISATION: generalisation_frame()})
    for head in ANALYSED_HEADS:
        assert f"label_{head}" in cases.columns
        assert f"pred_{head}" in cases.columns
        assert f"abs_error_{head}" in cases.columns


def test_cases_come_back_short_rather_than_reaching_outside_a_cell():
    # Two targets cannot fill a three-case library cell under the
    # distinct-protospacer rule. Fewer cases is the design's "at most 18";
    # a third PAM variant of an already-used target would not be.
    cases = v3_selection.select_cases({PAM_LIBRARY: library_frame(targets=2)})
    assert len(cases[cases["criterion"] == ERRLO]) == 2


def test_select_cases_refuses_an_unknown_stratum():
    with pytest.raises(ValueError, match="generalization"):
        v3_selection.select_cases({"generalization": generalisation_frame()})


# --- AC9: cases.json and non-circularity -----------------------------------


def test_write_cases_records_the_thresholds_each_cell_used(tmp_path):
    strata = both_strata()
    path = v3_selection.write_cases(tmp_path, v3_selection.select_cases(strata), strata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, frame in strata.items():
        thresholds = payload["strata"][name]
        assert thresholds["head_gap_p90"] == pytest.approx(
            v3_selection.score_stratum(name, frame).headgap_floor
        )
        assert thresholds["head_gap_median"] == pytest.approx(
            float(np.median(v3_selection.head_gap(frame)))
        )
        assert thresholds["rows"] == len(frame)


def test_cases_json_does_not_reference_any_attribution_artifact(tmp_path):
    strata = both_strata()
    path = v3_selection.write_cases(tmp_path, v3_selection.select_cases(strata), strata)
    body = path.read_text(encoding="utf-8")
    for token in ("attribution", "ism", "rollout", "npz", "integrated_gradients"):
        assert token not in body.lower()


def test_write_cases_refuses_a_payload_that_names_an_attribution_artifact(tmp_path):
    strata = both_strata()
    cases = v3_selection.select_cases(strata)
    cases["ism_delta_rank"] = 1
    with pytest.raises(ValueError, match="ism"):
        v3_selection.write_cases(tmp_path, cases, strata)


def test_attribution_reference_guard_looks_at_values_not_only_keys():
    with pytest.raises(ValueError, match="attribution"):
        v3_selection.assert_no_attribution_reference(
            {"note": "chosen from the IG attribution outliers"}
        )


def test_attribution_reference_guard_passes_a_clean_payload():
    v3_selection.assert_no_attribution_reference(
        {"cases": [{"case_id": "GEN-ERRLO-01", "note": "mechanism of the error"}]}
    )


def test_written_cases_are_byte_pinned_and_ordered_by_case_id(tmp_path):
    strata = both_strata()
    path = v3_selection.write_cases(tmp_path, v3_selection.select_cases(strata), strata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [case["case_id"] for case in payload["cases"]]
    assert ids == sorted(ids)
    checksum = json.loads(
        (tmp_path / v3_selection.CASES_CHECKSUM_NAME).read_text(encoding="utf-8")
    )
    assert checksum["files"][v3_selection.CASES_NAME] == v3_predict.sha256_of(path)


def test_cases_json_names_its_own_inputs_so_the_direction_is_readable(tmp_path):
    strata = both_strata()
    path = v3_selection.write_cases(tmp_path, v3_selection.select_cases(strata), strata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["inputs"] == ["predictions", "labels"]
    assert payload["non_circularity"]


# --- the real split and the real predictions -------------------------------


def real_strata() -> dict[str, pd.DataFrame]:
    if not SPLIT_CSV.exists() or not PREDICTIONS_CSV.exists():
        pytest.skip("the split or the predictions export is absent")
    return v3_strata.build(pd.read_csv(SPLIT_CSV), pd.read_csv(PREDICTIONS_CSV))


def test_real_ig_subset_is_four_hundred_rows_two_hundred_per_stratum():
    manifest = v3_selection.select_ig_subset(real_strata())
    assert len(manifest.entries) == v3_selection.IG_SUBSET_ROWS
    counts = manifest.entries["stratum"].value_counts().to_dict()
    assert counts == {
        GENERALISATION: v3_selection.IG_SUBSET_ROWS_PER_STRATUM,
        PAM_LIBRARY: v3_selection.IG_SUBSET_ROWS_PER_STRATUM,
    }
    assert manifest.entries["row_index"].is_unique


def test_real_ig_subset_never_overdraws_a_cell():
    strata = real_strata()
    manifest = v3_selection.select_ig_subset(strata)
    for name, frame in strata.items():
        capacity = pd.Series(v3_selection.ig_subset_cells(frame, name)).value_counts()
        drawn = manifest.entries[manifest.entries["stratum"] == name][
            "cell"
        ].value_counts()
        for cell, count in drawn.items():
            assert count <= capacity[cell]


def test_real_library_subset_is_drawn_from_all_five_targets():
    manifest = v3_selection.select_ig_subset(real_strata())
    library = manifest.entries[manifest.entries["stratum"] == PAM_LIBRARY]
    assert library["protospacer"].nunique() == 5


def test_real_case_table_fills_all_eighteen_cells():
    cases = v3_selection.select_cases(real_strata())
    assert len(cases) == 6 * v3_selection.CASES_PER_CELL
    assert cases["case_id"].is_unique
    library = cases[cases["stratum"] == PAM_LIBRARY]
    for criterion in (ERRLO, ERRHI, HEADGAP):
        assert library[library["criterion"] == criterion]["protospacer"].is_unique


# --- the driver ------------------------------------------------------------


def test_consumed_inputs_refuses_a_split_that_misses_its_pin(tmp_path):
    split = tmp_path / "test.csv"
    split.write_text("row_index\n0\n", encoding="utf-8")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("row_index\n0\n", encoding="utf-8")
    pin = tmp_path / "split_checksum.json"
    pin.write_text(json.dumps({"files": {"test.csv": "0" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="split_checksum.json"):
        v3_selection.consumed_inputs(split, predictions, split_checksum=pin)


def test_consumed_inputs_records_both_digests_and_whether_the_pin_was_checked(
    tmp_path,
):
    split = tmp_path / "test.csv"
    split.write_text("row_index\n0\n", encoding="utf-8")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("row_index\n1\n", encoding="utf-8")
    consumed = v3_selection.consumed_inputs(split, predictions, split_checksum=None)
    assert consumed["digests"]["test.csv"] == v3_predict.sha256_of(split)
    assert consumed["digests"]["predictions.csv"] == v3_predict.sha256_of(predictions)
    assert consumed["split_checksum_pin_checked"] is False


def test_build_selections_writes_both_frozen_lists(tmp_path):
    if not SPLIT_CSV.exists() or not PREDICTIONS_CSV.exists():
        pytest.skip("the split or the predictions export is absent")
    written = v3_selection.build_selections(artifacts_root=tmp_path)
    subset = v3_selection.load_ig_subset(written["ig_subset"])
    assert len(subset.entries) == v3_selection.IG_SUBSET_ROWS
    cases = json.loads(written["cases"].read_text(encoding="utf-8"))
    assert len(cases["cases"]) == 6 * v3_selection.CASES_PER_CELL
    # Both name the same consumed bytes, which is why they are drawn together.
    assert (
        cases["consumed"]["digests"]
        == json.loads(written["ig_subset"].read_text(encoding="utf-8"))["consumed"][
            "digests"
        ]
    )


def test_committed_artifacts_match_a_fresh_draw(tmp_path):
    """The tracked lists are reproducible from the tracked inputs.

    This is what "frozen" has to mean: re-running the driver on the same split
    and the same predictions reproduces the committed membership digest, so a
    reviewer can check the artifact rather than trust the run that wrote it.
    """
    committed = (
        SUBTREE_ROOT
        / "artifacts"
        / v3_selection.IG_SUBSET_DIR
        / v3_selection.IG_SUBSET_NAME
    )
    if not committed.exists() or not SPLIT_CSV.exists():
        pytest.skip("the committed subset or the split is absent")
    fresh = v3_selection.build_selections(artifacts_root=tmp_path)
    assert (
        v3_selection.load_ig_subset(fresh["ig_subset"]).digest
        == v3_selection.load_ig_subset(committed).digest
    )
