"""Tests for the 9-Hydra v3 analysis populations (#129).

The tests against the real split and the real predictions carry AC2 and AC3:
they pin the row and distinct-target counts the design commits to. They skip
rather than fail where those files are absent, since both are git-ignored.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import v3_predict
import v3_strata
from v3_strata import ANALYSED_HEADS, GENERALISATION, PAM_LIBRARY

XAI_TEST_ROOT = Path(v3_strata.__file__).resolve().parent.parent
SUBTREE_ROOT = Path(v3_strata.__file__).resolve().parent
SPLIT_CSV = XAI_TEST_ROOT / "SpCas9" / "published_split" / "test.csv"
PREDICTIONS_CSV = (
    SUBTREE_ROOT / "artifacts" / "v3_testset_predictions" / "predictions.csv"
)

OTHER_HEADS = [h for h in v3_predict.HEADS if h not in ANALYSED_HEADS]


def sequence(*, protospacer: str = "A" * 20, pam: str = "AGG") -> str:
    return "ACGT" + protospacer + pam + "TCA"


def make_split(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "row_index": index,
                "sequence": row.get("sequence", sequence()),
                "in_pam_library": row.get("in_pam_library", False),
                "trna_feature": 0,
                **{head: row.get(head, 10.0) for head in v3_predict.HEADS},
            }
            for index, row in enumerate(rows)
        ]
    )
    frame["protospacer"] = v3_strata.protospacers(frame["sequence"])
    return frame


def make_predictions(split: pd.DataFrame, value: float = 0.1) -> pd.DataFrame:
    frame = pd.DataFrame({"row_index": split["row_index"]})
    for head in v3_predict.HEADS:
        frame[f"pred_{head}"] = value
    return frame


# --- NGG and window slices -------------------------------------------------


def test_window_slices_tile_the_30bp_window():
    covered = []
    for part in (
        v3_strata.LEFT_FLANK,
        v3_strata.PROTOSPACER,
        v3_strata.PAM,
        v3_strata.RIGHT_FLANK,
    ):
        covered.extend(range(part.start, part.stop))
    assert covered == list(range(30))


def test_ngg_reads_positions_25_and_26():
    # The PAM is 24-26; position 24 is the free N.
    assert v3_strata.is_ngg([sequence(pam="AGG")]).tolist() == [True]
    assert v3_strata.is_ngg([sequence(pam="TGG")]).tolist() == [True]
    assert v3_strata.is_ngg([sequence(pam="AGA")]).tolist() == [False]
    assert v3_strata.is_ngg([sequence(pam="AAG")]).tolist() == [False]


def test_ngg_ignores_the_right_flank():
    assert v3_strata.is_ngg(["ACGT" + "A" * 20 + "AGG" + "GGG"]).tolist() == [True]
    assert v3_strata.is_ngg(["ACGT" + "A" * 20 + "AAA" + "GGG"]).tolist() == [False]


def test_protospacer_is_positions_4_to_23():
    assert v3_strata.protospacers([sequence(protospacer="C" * 20)]).tolist() == [
        "C" * 20
    ]


# --- intersection ----------------------------------------------------------


def test_analysis_intersection_keeps_only_rows_with_every_analysed_head():
    split = make_split([{}, {}, {}])
    split.loc[1, "SpCas9-NG"] = np.nan
    kept = v3_strata.analysis_intersection(split)
    assert kept["row_index"].tolist() == [0, 2]


def test_analysis_intersection_ignores_unanalysed_heads():
    # A row missing only HypaCas9 stays: that head is not explained.
    split = make_split([{}, {}])
    split.loc[0, OTHER_HEADS] = np.nan
    assert len(v3_strata.analysis_intersection(split)) == 2


def test_analysis_intersection_reports_a_missing_head_column():
    split = make_split([{}]).drop(columns=["evoCas9"])
    with pytest.raises(ValueError, match="evoCas9"):
        v3_strata.analysis_intersection(split)


# --- stratification --------------------------------------------------------


def test_generalisation_stratum_is_non_library_and_ngg():
    split = make_split(
        [
            {"in_pam_library": False, "sequence": sequence(pam="AGG")},  # keep
            {"in_pam_library": False, "sequence": sequence(pam="AGA")},  # dropped
            {"in_pam_library": True, "sequence": sequence(pam="AGG")},  # library
        ]
    )
    strata = v3_strata.stratify(split)
    assert strata[GENERALISATION]["row_index"].tolist() == [0]


def test_pam_library_stratum_keeps_non_ngg_rows():
    # The PAM contrast is this stratum's object of study, so dropping non-NGG
    # would delete the question.
    split = make_split(
        [
            {"in_pam_library": True, "sequence": sequence(pam="AGG")},
            {"in_pam_library": True, "sequence": sequence(pam="AGA")},
            {"in_pam_library": True, "sequence": sequence(pam="TTT")},
        ]
    )
    strata = v3_strata.stratify(split)
    assert strata[PAM_LIBRARY]["row_index"].tolist() == [0, 1, 2]
    assert v3_strata.describe_stratum(strata[PAM_LIBRARY])["non_ngg_rows"] == 2


def test_strata_are_disjoint_and_deliberately_incomplete():
    split = make_split(
        [
            {"in_pam_library": False, "sequence": sequence(pam="AGG")},
            {"in_pam_library": False, "sequence": sequence(pam="AGA")},
            {"in_pam_library": True, "sequence": sequence(pam="AGA")},
        ]
    )
    strata = v3_strata.stratify(split)
    left = set(strata[GENERALISATION]["row_index"])
    right = set(strata[PAM_LIBRARY]["row_index"])
    assert left & right == set()
    # Row 1 (non-library, non-NGG) belongs to neither.
    assert left | right == {0, 2}


def test_summarise_offers_no_pooled_total():
    split = make_split([{"in_pam_library": False, "sequence": sequence(pam="AGG")}])
    report = v3_strata.summarise(v3_strata.stratify(split))
    assert set(report) == {GENERALISATION, PAM_LIBRARY}
    assert not {"total", "all", "pooled", "combined"} & set(report)


def test_summarise_rejects_an_unknown_stratum():
    with pytest.raises(ValueError, match="invented"):
        v3_strata.summarise({"invented": make_split([{}])})


def test_describe_stratum_always_reports_distinct_targets():
    # Five rows, one target: n is not the evidence count.
    split = make_split(
        [
            {"in_pam_library": True, "sequence": sequence(pam=pam)}
            for pam in ("AGG", "AGA", "AGC", "AGT", "TGG")
        ]
    )
    report = v3_strata.describe_stratum(v3_strata.stratify(split)[PAM_LIBRARY])
    assert report["rows"] == 5
    assert report["distinct_protospacers"] == 1
    assert report["distinct_sequences"] == 5
    assert list(report["rows_per_protospacer"].values()) == [5]


# --- labels, predictions, error --------------------------------------------


def test_join_predictions_converts_labels_to_fractions():
    split = make_split([{"SpCas9": 49.6}])
    predictions = make_predictions(split, value=0.497)
    joined = v3_strata.join_predictions(split, predictions)
    assert joined["label_SpCas9"].tolist() == pytest.approx([0.496])
    assert joined["pred_SpCas9"].tolist() == pytest.approx([0.497])
    assert joined["abs_error_SpCas9"].tolist() == pytest.approx([0.001], abs=1e-9)


def test_join_predictions_does_not_rescale_the_predictions():
    split = make_split([{"SpCas9": 0.0}])
    joined = v3_strata.join_predictions(split, make_predictions(split, value=0.25))
    assert joined["pred_SpCas9"].tolist() == pytest.approx([0.25])


def test_join_predictions_joins_on_row_index_not_position():
    split = make_split([{"SpCas9": 10.0}, {"SpCas9": 90.0}])
    predictions = make_predictions(split)
    predictions["pred_SpCas9"] = [0.1, 0.9]
    shuffled = predictions.iloc[::-1].reset_index(drop=True)
    joined = v3_strata.join_predictions(split, shuffled)
    assert joined["pred_SpCas9"].tolist() == pytest.approx([0.1, 0.9])


def test_join_predictions_refuses_a_row_without_a_prediction():
    split = make_split([{}, {}])
    predictions = make_predictions(split).iloc[:1]
    with pytest.raises(ValueError, match="no prediction"):
        v3_strata.join_predictions(split, predictions)


def test_join_predictions_reports_missing_prediction_columns():
    split = make_split([{}])
    predictions = make_predictions(split).drop(columns=["pred_evoCas9"])
    with pytest.raises(ValueError, match="pred_evoCas9"):
        v3_strata.join_predictions(split, predictions)


# --- the real populations: AC2 and AC3 -------------------------------------


@pytest.fixture(scope="module")
def real_split() -> pd.DataFrame:
    if not SPLIT_CSV.exists():
        pytest.skip(f"{SPLIT_CSV} is not available (git-ignored)")
    return pd.read_csv(SPLIT_CSV)


def test_real_analysis_intersection_is_3851_rows(real_split):
    assert len(v3_strata.analysis_intersection(real_split)) == 3851


def test_real_head_coverage_differs_from_the_intersection(real_split):
    # Single-head figures use their own coverage; these are named apart from
    # the intersection precisely because they are not equal to it.
    coverage = {head: int(real_split[head].notna().sum()) for head in ANALYSED_HEADS}
    assert coverage == {"SpCas9": 5021, "SpCas9-NG": 3982, "evoCas9": 5161}
    assert 3851 not in set(coverage.values())


def test_real_strata_match_the_design(real_split):
    report = v3_strata.summarise(
        v3_strata.stratify(v3_strata.analysis_intersection(real_split))
    )
    assert report[GENERALISATION]["rows"] == 895
    assert report[GENERALISATION]["distinct_protospacers"] == 818
    assert report[GENERALISATION]["non_ngg_rows"] == 0

    assert report[PAM_LIBRARY]["rows"] == 1287
    assert report[PAM_LIBRARY]["distinct_protospacers"] == 5
    assert report[PAM_LIBRARY]["ngg_rows"] == 134


def test_real_strata_do_not_cover_the_intersection(real_split):
    # 1,669 non-library non-NGG rows are excluded on purpose.
    intersection = v3_strata.analysis_intersection(real_split)
    strata = v3_strata.stratify(intersection)
    assert sum(len(frame) for frame in strata.values()) == 895 + 1287
    assert len(intersection) == 3851


def test_real_build_attaches_errors(real_split):
    if not PREDICTIONS_CSV.exists():
        pytest.skip(f"{PREDICTIONS_CSV} is not available (git-ignored)")
    strata = v3_strata.build(real_split, pd.read_csv(PREDICTIONS_CSV))
    for name, expected_rows in ((GENERALISATION, 895), (PAM_LIBRARY, 1287)):
        frame = strata[name]
        assert len(frame) == expected_rows
        for head in ANALYSED_HEADS:
            assert frame[f"abs_error_{head}"].notna().all()
            # Both sides are fractions, so an error above 1 means a scale slip.
            assert frame[f"abs_error_{head}"].max() < 1.0
