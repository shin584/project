import json

import numpy as np
import pandas as pd
import pytest
from test_spcas9_data import make_sequence, numbered_protospacer

from SpCas9.spcas9_data import HEAD_NAMES, output_name
from SpCas9.spcas9_split import (
    CHECKSUM_FILENAME,
    SPLIT_FILENAMES,
    SPLITS,
    TEST,
    TRAIN,
    VAL,
    build_split,
    file_sha256,
    write_split,
)
from SpCas9.spcas9_train import (
    encode_inputs,
    head_coverage,
    load_published_split,
    masked_labels_and_weights,
    subsample,
)

# The two heads the spreadsheet and the model order differently.
ESPCAS9 = "eSpCas9(1.1)"
HF1 = "SpCas9-HF.1"


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A split-shaped frame. A head left out of a row dict reads `n.a.` (NaN)."""
    return pd.DataFrame(
        [
            {
                "sequence": row.get("sequence", make_sequence()),
                **{head: row.get(head, np.nan) for head in HEAD_NAMES},
            }
            for row in rows
        ]
    )


# --- unit: the sample weight masks exactly the missing heads ---


def test_weight_is_zero_exactly_where_the_head_is_missing():
    frame = _frame(
        [
            {"SpCas9": 10.0, "evoCas9": 20.0},
            {"SpCas9": 30.0},
            {"evoCas9": 40.0},
        ]
    )

    _, weights = masked_labels_and_weights(frame)

    assert weights[output_name("SpCas9")].tolist() == [1.0, 1.0, 0.0]
    assert weights[output_name("evoCas9")].tolist() == [1.0, 0.0, 1.0]
    assert weights[output_name("xCas")].tolist() == [0.0, 0.0, 0.0]


def test_masked_label_is_filled_to_zero_and_measured_labels_are_scaled():
    frame = _frame([{"SpCas9": 50.0}, {}, {"SpCas9": -7.5}])

    labels, _ = masked_labels_and_weights(frame)

    assert labels[output_name("SpCas9")].tolist() == pytest.approx([0.5, 0.0, -0.075])
    assert labels[output_name("xCas")].tolist() == [0.0, 0.0, 0.0]


def test_every_masked_label_slot_is_finite_so_keras_can_consume_it():
    frame = _frame([{"SpCas9": 1.0}, {}])

    labels, weights = masked_labels_and_weights(frame)

    for name in labels:
        assert np.isfinite(labels[name]).all()
        assert set(np.unique(weights[name])) <= {0.0, 1.0}


# --- unit: both dicts are keyed by output layer name, never by position ---


def test_both_dicts_are_keyed_by_output_layer_name():
    labels, weights = masked_labels_and_weights(_frame([{"SpCas9": 1.0}]))

    expected = {output_name(head) for head in HEAD_NAMES}
    assert set(labels) == expected
    assert set(weights) == expected
    assert "SpCas9-HF.1" not in labels
    assert "Output_SpCas9_HF_1" in labels


def test_labels_follow_the_head_name_not_the_column_order():
    """A transposition of the two reordered heads would fail here.

    The spreadsheet puts `eSpCas9(1.1)` before `SpCas9-HF.1`; the model emits
    them the other way round. Giving the two heads different values and
    different `n.a.` patterns catches a positional assembly either way.
    """
    frame = _frame([{ESPCAS9: 11.0, HF1: 99.0}, {ESPCAS9: 22.0}])
    frame = frame[[c for c in frame.columns if c != HF1] + [HF1]]
    assert list(frame.columns).index(ESPCAS9) < list(frame.columns).index(HF1)

    labels, weights = masked_labels_and_weights(frame)

    assert labels[output_name(ESPCAS9)].tolist() == pytest.approx([0.11, 0.22])
    assert labels[output_name(HF1)].tolist() == pytest.approx([0.99, 0.0])
    assert weights[output_name(ESPCAS9)].tolist() == [1.0, 1.0]
    assert weights[output_name(HF1)].tolist() == [1.0, 0.0]


def test_head_coverage_counts_measured_rows_per_head():
    frame = _frame([{"SpCas9": 1.0, HF1: 2.0}, {"SpCas9": 3.0}])

    assert head_coverage(frame)["SpCas9"] == 2
    assert head_coverage(frame)[HF1] == 1
    assert head_coverage(frame)["evoCas9"] == 0


# --- unit: inputs and the smoke subsample ---


def test_encode_inputs_has_the_input_layer_shape():
    frame = _frame([{"SpCas9": 1.0}, {"SpCas9": 2.0}])

    encoded = encode_inputs(frame)

    assert encoded.shape == (2, 30, 4)
    assert encoded.dtype == np.float32


def test_subsample_is_seeded_and_keeps_source_row_order():
    frame = _frame([{"SpCas9": float(i)} for i in range(50)])

    sample = subsample(frame, 10, seed=42)

    assert len(sample) == 10
    assert sample["SpCas9"].is_monotonic_increasing
    pd.testing.assert_frame_equal(sample, subsample(frame, 10, seed=42))


def test_subsample_returns_every_row_when_asked_for_more_than_it_has():
    frame = _frame([{"SpCas9": 1.0}, {"SpCas9": 2.0}])

    assert len(subsample(frame, 10)) == 2


# --- unit: the split is read from disk and checked, not re-derived ---


@pytest.fixture
def written_split(tmp_path):
    """A real published-split directory, written by `spcas9_split` itself."""
    frame = pd.DataFrame(
        [
            {
                "sequence": make_sequence(protospacer=numbered_protospacer(i)),
                "use": "Training" if i < 40 else "Test",
                "trna_feature": 0,
                **{head: float(i) for head in HEAD_NAMES},
            }
            for i in range(50)
        ]
    )
    write_split(build_split(frame), output_dir=tmp_path)
    return tmp_path


def test_load_published_split_returns_the_three_written_frames(written_split):
    frames = load_published_split(written_split)

    with open(written_split / CHECKSUM_FILENAME, encoding="utf-8") as f:
        record = json.load(f)
    assert set(frames) == set(SPLITS)
    for name in SPLITS:
        assert len(frames[name]) == record["rows"][name]
    assert len(frames[TRAIN]) + len(frames[VAL]) + len(frames[TEST]) == 50


def test_load_published_split_refuses_a_csv_whose_digest_moved(written_split):
    path = written_split / SPLIT_FILENAMES[TEST]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_published_split(written_split)


def test_load_published_split_refuses_a_row_count_that_moved(written_split):
    with open(written_split / CHECKSUM_FILENAME, encoding="utf-8") as f:
        record = json.load(f)
    record["rows"][TEST] += 1
    record["files"][SPLIT_FILENAMES[TEST]] = file_sha256(
        written_split / SPLIT_FILENAMES[TEST]
    )
    with open(written_split / CHECKSUM_FILENAME, "w", encoding="utf-8") as f:
        json.dump(record, f)

    with pytest.raises(ValueError, match="rows"):
        load_published_split(written_split)


# --- regression: the real published split, as training will read it ---


@pytest.fixture(scope="module")
def real_split() -> dict[str, pd.DataFrame]:
    return load_published_split()


def test_real_split_matches_the_pinned_row_counts(real_split):
    assert len(real_split[TRAIN]) == 29134
    assert len(real_split[VAL]) == 3676
    assert len(real_split[TEST]) == 5186


def test_real_test_coverage_per_head_is_what_the_masking_keeps(real_split):
    coverage = head_coverage(real_split[TEST])

    assert coverage["SpCas9"] == 5021
    assert coverage["SpCas9-NG"] == 3982
    assert coverage["VRQR variant"] == 3979
    assert min(coverage.values()) == 3979


def test_real_split_masks_rather_than_drops_the_partially_measured_rows(real_split):
    train = real_split[TRAIN]
    _, weights = masked_labels_and_weights(train)

    all_nine = train[list(HEAD_NAMES)].notna().all(axis=1).sum()
    assert all_nine < len(train)
    # Every row carries at least one measured head, so no row is wholly masked.
    assert (
        np.stack([weights[output_name(h)] for h in HEAD_NAMES]).sum(axis=0).min() >= 1
    )
