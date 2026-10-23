import json

import numpy as np
import pandas as pd
import pytest
from test_spcas9_data import make_sequence, numbered_protospacer

from SpCas9.spcas9_data import HEAD_NAMES
from SpCas9.spcas9_eval import (
    METADATA_COLUMNS,
    NON_LIBRARY,
    PAM_LIBRARY,
    PREDICTIONS_FILENAME,
    TRNA_VALUES,
    available,
    build_predictions,
    build_report,
    empty_strata,
    head_report,
    write_predictions,
)
from SpCas9.spcas9_metrics import pred_column, true_column
from SpCas9.spcas9_split import (
    OUTPUT_DIR as SPLIT_DIR,
)
from SpCas9.spcas9_split import (
    PAM_LIBRARY_COLUMN,
    ROW_INDEX_COLUMN,
    SPLIT_COLUMN,
    TEST,
)
from SpCas9.spcas9_train import load_published_split

OTHER_HEADS = [head for head in HEAD_NAMES if head != "SpCas9"]

# Per-head test coverage the 9-Hydra Testset is pinned to (issue #48).
PINNED_COVERAGE = {
    "SpCas9": 5021,
    "SpCas9-NG": 3982,
    "VRQR variant": 3979,
    "xCas": 5170,
    "Sniper-Cas9": 5170,
    "SpCas9-HF.1": 5177,
    "eSpCas9(1.1)": 5176,
    "HypaCas9": 5177,
    "evoCas9": 5161,
}


def _test_frame(rows: list[dict]) -> pd.DataFrame:
    """A test-split-shaped frame. A head left out of a row reads `n.a.` (NaN)."""
    return pd.DataFrame(
        [
            {
                ROW_INDEX_COLUMN: row.get("row_index", i),
                "sequence": row.get(
                    "sequence", make_sequence(protospacer=numbered_protospacer(i))
                ),
                "protospacer": row.get("protospacer", numbered_protospacer(i)),
                PAM_LIBRARY_COLUMN: row.get("in_pam_library", False),
                "trna_feature": row.get("trna_feature", 0),
                "use": "Test",
                SPLIT_COLUMN: TEST,
                **{head: row.get(head, np.nan) for head in HEAD_NAMES},
            }
            for i, row in enumerate(rows)
        ]
    )


def _predictions(frame: pd.DataFrame, offsets: dict[str, float] | None = None):
    """Predictions a fixed offset away from each head's label.

    An unmeasured label would make the offset NaN, so the masked rows are
    given a deliberately absurd prediction instead: any metric that averaged
    over them would be visibly wrong rather than merely NaN.
    """
    offsets = offsets or {}
    preds = {}
    for head in HEAD_NAMES:
        label = frame[head].to_numpy(dtype=float) / 100.0
        preds[head] = np.where(np.isfinite(label), label + offsets.get(head, 0.1), 99.0)
    return preds


@pytest.fixture
def toy():
    """Twelve rows where `SpCas9` is measured on nine and missing on three."""
    frame = _test_frame(
        [{"SpCas9": float(i)} if i % 4 else {"xCas": float(i)} for i in range(12)]
    )
    return build_predictions(frame, _predictions(frame))


# --- unit: a head's metrics cover only that head's available rows ---


def test_available_keeps_exactly_the_rows_whose_label_is_measured(toy):
    rows = available(toy, "SpCas9")

    assert len(rows) == 9
    assert rows[true_column("SpCas9")].notna().all()


def test_head_metrics_count_and_score_only_the_available_rows(toy):
    report = head_report(toy, "SpCas9")

    assert report["n"] == 9
    assert report["mae"] == pytest.approx(0.1)
    assert report["mse"] == pytest.approx(0.01)


def test_a_masked_row_cannot_move_a_heads_metrics():
    """The same labels, scored twice with different predictions where masked."""
    frame = _test_frame([{"SpCas9": float(i)} if i % 3 else {} for i in range(15)])
    preds = _predictions(frame)
    wrecked = {head: values.copy() for head, values in preds.items()}
    masked = ~np.isfinite(frame["SpCas9"].to_numpy(dtype=float))
    wrecked["SpCas9"][masked] = -1234.0

    assert head_report(build_predictions(frame, preds), "SpCas9") == head_report(
        build_predictions(frame, wrecked), "SpCas9"
    )


def test_every_head_is_reported_even_one_with_no_measured_row(toy):
    report = build_report(toy)

    assert list(report["heads"]) == list(HEAD_NAMES)
    assert report["heads"]["evoCas9"]["n"] == 0
    assert report["heads"]["evoCas9"]["mae"] is None


# --- unit: the two strata, read from the split metadata ---


def test_the_pam_library_strata_partition_the_heads_available_rows():
    frame = _test_frame(
        [{"SpCas9": float(i), "in_pam_library": i % 2 == 0} for i in range(20)]
    )
    predictions = build_predictions(frame, _predictions(frame))

    report = head_report(predictions, "SpCas9")
    library = report["strata"][PAM_LIBRARY_COLUMN]

    assert library[PAM_LIBRARY]["n"] + library[NON_LIBRARY]["n"] == report["n"] == 20
    assert library[PAM_LIBRARY]["n"] == 10


def test_the_trna_strata_partition_the_heads_available_rows():
    frame = _test_frame(
        [{"SpCas9": float(i), "trna_feature": i % 2} for i in range(20)]
    )
    predictions = build_predictions(frame, _predictions(frame))

    report = head_report(predictions, "SpCas9")
    trna = report["strata"]["trna_feature"]

    assert [trna[value]["n"] for value in TRNA_VALUES] == [10, 10]
    assert sum(trna[value]["n"] for value in TRNA_VALUES) == report["n"]


def test_a_strata_only_counts_rows_the_head_has_measured():
    frame = _test_frame(
        [
            {"SpCas9": float(i), "in_pam_library": True} if i < 6 else {"xCas": 1.0}
            for i in range(10)
        ]
    )
    predictions = build_predictions(frame, _predictions(frame))

    library = head_report(predictions, "SpCas9")["strata"][PAM_LIBRARY_COLUMN]

    assert library[PAM_LIBRARY]["n"] == 6
    assert library[NON_LIBRARY]["n"] == 0


def test_the_strata_follow_the_metadata_column_not_a_fresh_derivation():
    """`in_pam_library` is read from the split, never recomputed here.

    Every row below carries a distinct protospacer with one PAM, so a
    re-derivation would put all twelve rows outside the library; the column
    says otherwise for four of them, and the column is what the report has to
    honour.
    """
    frame = _test_frame(
        [{"SpCas9": float(i), "in_pam_library": i < 4} for i in range(12)]
    )
    predictions = build_predictions(frame, _predictions(frame))

    library = head_report(predictions, "SpCas9")["strata"][PAM_LIBRARY_COLUMN]

    assert library[PAM_LIBRARY]["n"] == 4
    assert library[NON_LIBRARY]["n"] == 8


def test_empty_strata_are_named_so_a_missing_comparison_is_not_read_as_a_tie():
    frame = _test_frame(
        [
            {"SpCas9": float(i), "in_pam_library": i % 2 == 0, "trna_feature": 0}
            for i in range(12)
        ]
    )
    predictions = build_predictions(frame, _predictions(frame))

    assert empty_strata(predictions, "SpCas9") == [f"trna_feature={TRNA_VALUES[1]}"]
    assert build_report(predictions)["empty_strata"]["SpCas9"] == [
        f"trna_feature={TRNA_VALUES[1]}"
    ]


# --- unit: the predictions frame joins back to the split metadata ---


def test_build_predictions_carries_the_metadata_columns_through(toy):
    assert list(toy.columns)[: len(METADATA_COLUMNS)] == METADATA_COLUMNS
    assert toy[ROW_INDEX_COLUMN].tolist() == list(range(12))


def test_build_predictions_keeps_an_unmeasured_label_missing(toy):
    labels = toy[true_column("SpCas9")]

    assert labels.isna().sum() == 3
    assert toy[pred_column("SpCas9")].notna().all()


def test_build_predictions_puts_each_head_on_the_scale_the_model_was_trained_on():
    frame = _test_frame([{"SpCas9": 50.0}, {"SpCas9": -7.5}])

    predictions = build_predictions(frame, _predictions(frame))

    assert predictions[true_column("SpCas9")].tolist() == pytest.approx([0.5, -0.075])


def test_build_predictions_rejects_a_prediction_of_the_wrong_length():
    frame = _test_frame([{"SpCas9": 1.0}, {"SpCas9": 2.0}])
    preds = _predictions(frame)
    preds["SpCas9"] = preds["SpCas9"][:1]

    with pytest.raises(ValueError, match="SpCas9"):
        build_predictions(frame, preds)


# --- unit: the report is strict JSON and says what it scored ---


def test_report_is_strict_json(toy):
    json.dumps(build_report(toy), allow_nan=False)


def test_report_states_the_scale_and_the_scored_row_count(toy):
    report = build_report(toy)

    assert report["rows"] == 12
    assert "100" in report["scale"]


# --- unit: the predictions file and its checksum sidecar ---


@pytest.fixture
def written(tmp_path, toy):
    checkpoint = tmp_path / "9hydra_v2.keras"
    checkpoint.write_bytes(b"not a real checkpoint")
    record = write_predictions(
        toy, checkpoint_path=checkpoint, output_dir=tmp_path, split_dir=SPLIT_DIR
    )
    return tmp_path, record


def test_the_checksum_sidecar_pins_the_predictions_the_checkpoint_and_the_split(
    written,
):
    _, record = written

    assert record["rows"] == 12
    assert record["file"] == PREDICTIONS_FILENAME
    assert record["checkpoint"] == "9hydra_v2.keras"
    assert len(record["checkpoint_sha256"]) == 64
    assert len(record["file_sha256"]) == 64
    assert record["split"]["rows"][TEST] == 5186


def test_the_predictions_csv_is_written_with_a_fixed_line_ending(written):
    tmp_path, _ = written

    raw = (tmp_path / PREDICTIONS_FILENAME).read_bytes()

    assert b"\r\n" not in raw


def test_the_written_predictions_read_back_identically(written):
    tmp_path, record = written

    reread = pd.read_csv(tmp_path / PREDICTIONS_FILENAME)

    assert len(reread) == record["rows"]
    assert reread[true_column("SpCas9")].isna().sum() == 3


# --- pinned regression over the real 9-Hydra Testset (no model loaded) ---


@pytest.fixture(scope="module")
def real_test_rows():
    return load_published_split(SPLIT_DIR)[TEST]


def test_the_real_testset_has_the_pinned_per_head_coverage(real_test_rows):
    coverage = {head: int(real_test_rows[head].notna().sum()) for head in HEAD_NAMES}

    assert len(real_test_rows) == 5186
    assert coverage == PINNED_COVERAGE


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_the_real_strata_partition_every_heads_coverage(real_test_rows):
    frame = build_predictions(
        real_test_rows,
        {head: np.zeros(len(real_test_rows)) for head in HEAD_NAMES},
    )

    for head in HEAD_NAMES:
        report = head_report(frame, head)
        library = report["strata"][PAM_LIBRARY_COLUMN]
        trna = report["strata"]["trna_feature"]
        assert report["n"] == PINNED_COVERAGE[head]
        assert library[PAM_LIBRARY]["n"] + library[NON_LIBRARY]["n"] == report["n"]
        assert sum(trna[value]["n"] for value in TRNA_VALUES) == report["n"]


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_the_two_censored_heads_have_no_trna_rows_to_compare(real_test_rows):
    """`SpCas9-NG` and `VRQR variant` are measured only on tRNA=0 rows.

    Not a quirk of our slicing: the spreadsheet has no tRNA=1 measurement for
    either head, so their tRNA stratum carries no comparison at all.
    """
    frame = build_predictions(
        real_test_rows,
        {head: np.zeros(len(real_test_rows)) for head in HEAD_NAMES},
    )

    censored = {
        head: empty_strata(frame, head)
        for head in HEAD_NAMES
        if empty_strata(frame, head)
    }

    assert censored == {
        "SpCas9-NG": ["trna_feature=1"],
        "VRQR variant": ["trna_feature=1"],
    }
