import numpy as np
import pandas as pd
import pytest

from SpCas9.spcas9_data import (
    HEAD_NAMES,
    TRNA_COLUMN,
    filter_all_heads,
    filter_any_head,
    head_labels,
    is_published_test,
    load_spcas9_table,
    normalize_table,
    one_hot,
    output_name,
    pam_library_protospacers,
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
TRNA_SPREADSHEET_COLUMN = "tRNA feature \n(1 = tRNA-N20 sgRNA, \n0 = (G/g)N19 sgRNA)"
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


def numbered_protospacer(i: int) -> str:
    """A distinct 20nt protospacer per `i`, numbered in base-4 over ACGT."""
    bases = "ACGT"
    return "".join(bases[(i >> (2 * k)) & 0b11] for k in range(20))


def _raw_table(rows: list[dict]) -> pd.DataFrame:
    """A frame shaped like the spreadsheet. Each row dict may set any head."""
    return pd.DataFrame(
        [
            {
                SEQUENCE_COLUMN: row.get("sequence", make_sequence()),
                TRNA_SPREADSHEET_COLUMN: row.get("trna_feature", 0),
                USE_COLUMN: row.get("use", "Training"),
                **{head: row.get(head, 1.0) for head in SPREADSHEET_HEAD_ORDER},
            }
            for row in rows
        ],
        columns=[
            SEQUENCE_COLUMN,
            TRNA_SPREADSHEET_COLUMN,
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


# --- unit: the tRNA feature passes through as metadata ---


def test_normalize_table_passes_the_trna_feature_through():
    raw = _raw_table([{"trna_feature": 1}, {"trna_feature": 0}])

    table = normalize_table(raw)

    assert table[TRNA_COLUMN].tolist() == [1, 0]


def test_trna_feature_is_not_treated_as_a_head_column():
    # A row missing every head must still drop out of both filters, however
    # the tRNA column reads - it is metadata, not a label.
    raw = _raw_table(
        [{"trna_feature": 1, **{head: "n.a." for head in SPREADSHEET_HEAD_ORDER}}]
    )
    table = normalize_table(raw)

    assert TRNA_COLUMN not in HEAD_NAMES
    assert len(filter_any_head(table)) == 0
    assert len(filter_all_heads(table)) == 0


# --- unit: the PAM library ---


def test_pam_library_holds_only_protospacers_seen_with_more_than_one_pam():
    library = pam_library_protospacers(
        [
            make_sequence(protospacer="G" * 20, pam="TGG"),
            make_sequence(protospacer="G" * 20, pam="CGG"),
            make_sequence(protospacer="C" * 20, pam="TGG"),
            make_sequence(protospacer="C" * 20, pam="TGG", tail="AAA"),
            make_sequence(protospacer="T" * 20, pam="AGG"),
        ]
    )

    assert library == {"G" * 20}


# --- regression: the real kim_2020_table_s8.xlsx ---


def test_real_table_filters_to_the_pinned_row_counts(real_table):
    assert len(filter_any_head(real_table)) == 37_996
    assert len(filter_all_heads(real_table)) == 29_303


def test_real_table_sequences_are_all_30bp(real_table):
    assert set(real_table["sequence"].str.len()) == {30}


def test_real_trna_feature_is_a_zero_one_flag_on_every_row(real_table):
    assert set(real_table[TRNA_COLUMN].unique()) == {0, 1}


def test_real_pam_library_holds_the_pinned_thirty_protospacers(real_table):
    library = pam_library_protospacers(filter_any_head(real_table)["sequence"])

    assert len(library) == 30


# --- unit: the model plumbing every 9-Hydra version shares ---


def test_one_hot_uses_acgt_channel_order_and_zeros_other_bases():
    encoded = one_hot(["ACGTN"])
    assert encoded.shape == (1, 5, 4)
    assert encoded.dtype == np.float32
    assert encoded[0].tolist() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]


def test_output_name_cleans_the_two_heads_with_punctuation():
    assert output_name("SpCas9") == "Output_SpCas9"
    assert output_name("SpCas9-HF.1") == "Output_SpCas9_HF_1"
    assert output_name("eSpCas9(1.1)") == "Output_eSpCas9_1_1_"
