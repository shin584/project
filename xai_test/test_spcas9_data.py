import numpy as np
import pandas as pd
import pytest
from SpCas9.spcas9_data import (
    HEAD_NAMES,
    filter_all_heads,
    filter_any_head,
    head_labels,
    is_published_test,
    load_spcas9_table,
    normalize_table,
    pams,
    protospacers,
)

# The spreadsheet's own column names, including its column order, which puts
# eSpCas9(1.1) before SpCas9-HF.1 - the reverse of the model's output order.
SEQUENCE_COLUMN = (
    "Input sequence = Target sequence + neighboring sequences = \n"
    "(4 nt left neighboring sequence + 20 nt protospacer (blue lowercase) "
    "+ 3 nt PAM (RED BOLD UPPERCASE)+ 3nt right neighboring sequence)"
)
USE_COLUMN = "Use (Training or Test)"
SPREADSHEET_HEAD_ORDER = [
    "SpCas9",
    "SpCas9-NG",
    "VRQR variant",
    "xCas",
    "Sniper-Cas9",
    "eSpCas9(1.1)",
    "SpCas9-HF.1",
    "HypaCas9",
    "evoCas9",
]


def make_sequence(
    protospacer: str = "A" * 20, pam: str = "TGG", tail: str = "CCC"
) -> str:
    """A 30bp row: 4nt left flank + 20nt protospacer + 3nt PAM + 3nt right flank."""
    return "ACGT" + protospacer + pam + tail


def _raw_table(rows: list[dict]) -> pd.DataFrame:
    """A frame shaped like the spreadsheet. Each row dict may set any head."""
    return pd.DataFrame(
        [
            {
                SEQUENCE_COLUMN: row.get("sequence", make_sequence()),
                "tRNA feature \n(1 = tRNA-N20 sgRNA, \n0 = (G/g)N19 sgRNA)": 0,
                USE_COLUMN: row.get("use", "Training"),
                **{head: row.get(head, 1.0) for head in SPREADSHEET_HEAD_ORDER},
            }
            for row in rows
        ],
        columns=[
            SEQUENCE_COLUMN,
            "tRNA feature \n(1 = tRNA-N20 sgRNA, \n0 = (G/g)N19 sgRNA)",
            USE_COLUMN,
            *SPREADSHEET_HEAD_ORDER,
        ],
    )


@pytest.fixture(scope="module")
def real_table() -> pd.DataFrame:
    return load_spcas9_table()


# --- unit: loading and filtering ---


def test_normalize_table_drops_sequences_that_are_not_30bp():
    raw = _raw_table(
        [
            {"sequence": make_sequence()},
            {"sequence": "ACGT"},
            {"sequence": make_sequence() + "A"},
            {"sequence": make_sequence(protospacer="C" * 20)},
        ]
    )

    table = normalize_table(raw)

    assert table["sequence"].tolist() == [
        make_sequence(),
        make_sequence(protospacer="C" * 20),
    ]


def test_normalize_table_upper_cases_and_strips_sequences():
    raw = _raw_table([{"sequence": "  " + make_sequence().lower() + " "}])

    table = normalize_table(raw)

    assert table["sequence"].tolist() == [make_sequence()]


def test_normalize_table_turns_the_na_literal_into_nan():
    raw = _raw_table([{"SpCas9": "n.a.", "evoCas9": 12.5}])

    table = normalize_table(raw)

    assert np.isnan(table["SpCas9"].iloc[0])
    assert table["evoCas9"].iloc[0] == 12.5


def test_filters_drop_exactly_the_na_targets_and_the_non_30bp_sequences():
    complete = make_sequence(protospacer="A" * 20)
    one_head_missing = make_sequence(protospacer="C" * 20)
    all_heads_missing = make_sequence(protospacer="G" * 20)
    raw = _raw_table(
        [
            {"sequence": complete},
            {"sequence": one_head_missing, "SpCas9": "n.a."},
            {
                "sequence": all_heads_missing,
                **{head: "n.a." for head in SPREADSHEET_HEAD_ORDER},
            },
            {"sequence": "ACGT"},
        ]
    )
    table = normalize_table(raw)

    assert filter_all_heads(table)["sequence"].tolist() == [complete]
    assert filter_any_head(table)["sequence"].tolist() == [
        complete,
        one_head_missing,
    ]


def test_filters_leave_the_frame_reindexed_from_zero():
    raw = _raw_table([{"SpCas9": "n.a."}, {"sequence": make_sequence(pam="AGG")}])

    filtered = filter_all_heads(normalize_table(raw))

    assert filtered.index.tolist() == [0]
    assert filtered["sequence"].tolist() == [make_sequence(pam="AGG")]


# --- unit: target window ---


def test_protospacer_and_pam_come_from_the_spcas9_target_window():
    table = normalize_table(
        _raw_table([{"sequence": make_sequence(protospacer="G" * 20, pam="CGG")}])
    )

    assert protospacers(table).tolist() == ["G" * 20]
    assert pams(table).tolist() == ["CGG"]


# --- unit: labels keyed by head name ---


def test_head_labels_are_keyed_by_head_name_not_column_position():
    # The two heads the spreadsheet and the model disagree about, given
    # values that make a positional swap visible.
    raw = _raw_table([{"eSpCas9(1.1)": 20.0, "SpCas9-HF.1": 80.0}])

    labels = head_labels(normalize_table(raw))

    assert set(labels) == set(HEAD_NAMES)
    assert labels["eSpCas9(1.1)"] == pytest.approx([0.2])
    assert labels["SpCas9-HF.1"] == pytest.approx([0.8])


def test_head_labels_scale_percentages_the_way_the_checkpoint_was_trained():
    raw = _raw_table([{"SpCas9": 95.4}, {"SpCas9": -7.5}])

    labels = head_labels(normalize_table(raw))

    assert labels["SpCas9"] == pytest.approx([0.954, -0.075])


def test_is_published_test_marks_only_the_papers_held_out_rows():
    raw = _raw_table([{"use": "Test"}, {"use": "Training"}, {"use": " Test "}])

    flags = is_published_test(normalize_table(raw))

    assert flags.tolist() == [True, False, True]


# --- regression: the real kim_2020_table_s8.xlsx ---


def test_real_table_filters_to_the_pinned_row_counts(real_table):
    assert len(filter_any_head(real_table)) == 37_996
    assert len(filter_all_heads(real_table)) == 29_303


def test_real_table_sequences_are_all_30bp(real_table):
    assert set(real_table["sequence"].str.len()) == {30}
