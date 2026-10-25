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
    EPOCHS,
    LOG_FILENAME,
    TRAIN_LIBRARIES,
    build_training_log,
    encode_inputs,
    head_coverage,
    hyperparameter_record,
    library_versions,
    load_published_split,
    masked_labels_and_weights,
    split_record,
    stopping_epoch,
    subsample,
    write_training_log,
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


def test_reference_loss_uses_binary_mask_all_heads_and_validation_row_denominator():
    from SpCas9.spcas9_train import reference_validation_loss

    frame = _frame([{ESPCAS9: 100.0, HF1: 200.0}, {ESPCAS9: 300.0}])
    labels, weights = masked_labels_and_weights(frame)
    predictions = {output_name(h): np.zeros(2) for h in reversed(HEAD_NAMES)}
    predictions[output_name(HF1)] = np.array([0.0, 9999.0])
    # (1**2 + 3**2 + 2**2) / 2 rows = 7, not /3 measurements or /9 heads.
    assert reference_validation_loss(labels, weights, predictions) == 7.0
    # A placeholder must never enter the criterion, even if non-finite.
    labels[output_name(HF1)][1] = np.nan
    assert reference_validation_loss(labels, weights, predictions) == 7.0


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


# --- unit: the training log records what reproduces the run (#47) ---


def _history(val_losses: list[float]) -> dict[str, list[float]]:
    """A `History.history`-shaped dict with one metric per epoch."""
    return {
        "loss": [v + 0.01 for v in val_losses],
        "val_loss": list(val_losses),
        "learning_rate": [0.001] * len(val_losses),
    }


def test_stopping_epoch_is_the_best_val_loss_epoch_one_indexed():
    record = stopping_epoch(_history([0.5, 0.3, 0.4, 0.35, 0.36, 0.37]))

    assert record["epochs_run"] == 6
    assert record["best_epoch"] == 2
    assert record["best_val_loss"] == pytest.approx(0.3)


def test_stopping_epoch_reports_early_stopping_against_the_epoch_budget():
    stopped = stopping_epoch(_history([0.5] * 7), epochs=EPOCHS)
    ran_out = stopping_epoch(_history([0.5] * 50), epochs=EPOCHS)

    assert stopped["early_stopped"] is True
    assert ran_out["early_stopped"] is False


def test_hyperparameter_record_pins_v1s_recipe():
    record = hyperparameter_record()

    assert record["optimizer"] == "Adam"
    assert record["learning_rate"] == 0.001
    assert record["batch_size"] == 32
    assert record["epochs"] == 50
    assert record["seed"] == 42
    assert record["early_stopping"] == {
        "monitor": "val_loss",
        "patience": 5,
        "restore_best_weights": True,
    }
    assert record["reduce_lr_on_plateau"] == {
        "monitor": "val_loss",
        "factor": 0.2,
        "patience": 3,
        "min_lr": 1e-6,
    }


def test_the_log_reports_the_four_libraries_the_run_depends_on():
    versions = library_versions(TRAIN_LIBRARIES)

    assert set(versions) == {"tensorflow", "keras", "numpy", "pandas"}
    assert all(v is not None for v in versions.values())


def test_split_record_carries_the_checksums_that_tie_the_run_to_one_split(
    written_split,
):
    record = split_record(written_split)

    with open(written_split / CHECKSUM_FILENAME, encoding="utf-8") as f:
        pinned = json.load(f)
    for name in SPLITS:
        filename = SPLIT_FILENAMES[name]
        assert record["files"][filename] == pinned["files"][filename]
        assert record["files"][filename] == file_sha256(written_split / filename)
    assert record["rows"] == pinned["rows"]
    assert record["checksum_sha256"] == file_sha256(written_split / CHECKSUM_FILENAME)


def test_build_training_log_records_every_field_the_issue_asks_for(
    written_split, tmp_path
):
    checkpoint = tmp_path / "9hydra_v2.keras"
    checkpoint.write_bytes(b"not a real checkpoint")
    splits = load_published_split(written_split)

    log = build_training_log(
        history=_history([0.5, 0.3, 0.4]),
        splits=splits,
        checkpoint_path=checkpoint,
        split_dir=written_split,
        duration_seconds=12.5,
    )

    assert log["hyperparameters"] == hyperparameter_record()
    assert log["hyperparameters"]["seed"] == 42
    assert log["history"]["val_loss"] == [0.5, 0.3, 0.4]
    assert log["stopping"]["best_epoch"] == 2
    assert (
        log["versions"]["tensorflow"] == library_versions(TRAIN_LIBRARIES)["tensorflow"]
    )
    assert log["split"]["files"][SPLIT_FILENAMES[TRAIN]] == file_sha256(
        written_split / SPLIT_FILENAMES[TRAIN]
    )
    assert log["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert log["rows_fitted"][TRAIN] == len(splits[TRAIN])
    assert (
        log["head_coverage"][TRAIN]["SpCas9"] == head_coverage(splits[TRAIN])["SpCas9"]
    )
    assert log["duration_seconds"] == 12.5
    assert log["hyperparameters"]["batch_size"] == 32
    assert log["device"]["gpus_visible"] == 0


def test_build_training_log_is_json_serialisable(written_split, tmp_path):
    checkpoint = tmp_path / "9hydra_v2.keras"
    checkpoint.write_bytes(b"not a real checkpoint")

    log = build_training_log(
        history=_history([0.5, 0.3]),
        splits=load_published_split(written_split),
        checkpoint_path=checkpoint,
        split_dir=written_split,
        duration_seconds=1.0,
    )

    assert json.loads(json.dumps(log))["stopping"]["epochs_run"] == 2


def test_a_smoke_log_reports_the_epochs_it_asked_for_not_the_full_budget():
    """The log must state what the run requested, not the module constant."""
    record = hyperparameter_record(epochs=1)

    assert record["epochs"] == 1
    assert record["learning_rate"] == 0.001


def test_build_training_log_names_the_fitted_rows_apart_from_the_split(
    written_split, tmp_path
):
    """A subsampled run must not read as if the whole split went through it."""
    checkpoint = tmp_path / "9hydra_v2.keras"
    checkpoint.write_bytes(b"not a real checkpoint")
    splits = load_published_split(written_split)
    subsampled = {name: subsample(frame, 3) for name, frame in splits.items()}

    log = build_training_log(
        history=_history([0.5]),
        splits=subsampled,
        checkpoint_path=checkpoint,
        split_dir=written_split,
        epochs=1,
    )

    assert log["rows_fitted"] == {name: 3 for name in SPLITS}
    assert log["split"]["rows"][TRAIN] == len(splits[TRAIN])


# --- unit: the tracked artifact is written byte-reproducibly ---


def test_write_training_log_fixes_the_line_ending_so_the_tracked_file_is_stable(
    written_split, tmp_path
):
    """LF on every platform - the same rule as the split's checksum sidecar."""
    checkpoint = tmp_path / "9hydra_v2.keras"
    checkpoint.write_bytes(b"not a real checkpoint")
    log = build_training_log(
        history=_history([0.5]),
        splits=load_published_split(written_split),
        checkpoint_path=checkpoint,
        split_dir=written_split,
    )

    path = write_training_log(log, tmp_path / "out" / LOG_FILENAME)

    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8"))["stopping"]["epochs_run"] == 1


def test_write_training_log_refuses_a_nan_rather_than_writing_invalid_json(
    written_split, tmp_path
):
    checkpoint = tmp_path / "9hydra_v2.keras"
    checkpoint.write_bytes(b"not a real checkpoint")
    log = build_training_log(
        history={"loss": [float("nan")], "val_loss": [float("nan")]},
        splits=load_published_split(written_split),
        checkpoint_path=checkpoint,
        split_dir=written_split,
    )

    with pytest.raises(ValueError):
        write_training_log(log, tmp_path / LOG_FILENAME)
