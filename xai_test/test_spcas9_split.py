import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from test_spcas9_data import make_sequence, numbered_protospacer

from SpCas9.spcas9_data import (
    HEAD_NAMES,
    TRNA_COLUMN,
    filter_any_head,
    load_spcas9_table,
)
from SpCas9.spcas9_split import (
    CHECKSUM_FILENAME,
    OUTPUT_COLUMNS,
    OUTPUT_DIR,
    SPLIT_COLUMN,
    SPLITS,
    TEST,
    TRAIN,
    VAL,
    build_split,
    carve_validation_protospacers,
    main,
    partition,
    summarize,
    write_split,
)


def _toy_frame(rows: list[dict]) -> pd.DataFrame:
    """The columns `build_split` reads off the at-least-one-head frame.

    Each row dict may set `sequence`, `use`, `trna_feature` or any head; the
    heads default to a measured value so every row survives `filter_any_head`.
    """
    return pd.DataFrame(
        [
            {
                "sequence": row.get("sequence", make_sequence()),
                "use": row.get("use", "Training"),
                TRNA_COLUMN: row.get("trna_feature", 0),
                **{head: row.get(head, 1.0) for head in HEAD_NAMES},
            }
            for row in rows
        ]
    )


def _numbered_frame(n_train: int, n_test: int = 0) -> pd.DataFrame:
    """`n_train` Training rows then `n_test` Test rows, all distinct protospacers."""
    return _toy_frame(
        [
            {
                "sequence": make_sequence(protospacer=numbered_protospacer(i)),
                "use": "Training" if i < n_train else "Test",
            }
            for i in range(n_train + n_test)
        ]
    )


# --- unit: the Use column is honoured verbatim ---


def test_test_rows_are_exactly_the_use_column_test_rows():
    frame = _toy_frame(
        [
            {"sequence": make_sequence(protospacer=numbered_protospacer(i)), "use": u}
            for i, u in enumerate(["Training", "Test", "Training", "Test", "Test"])
        ]
    )

    split = build_split(frame)

    assert (split[SPLIT_COLUMN] == TEST).tolist() == [False, True, False, True, True]


def test_no_test_row_is_ever_moved_into_train_or_val():
    frame = _numbered_frame(n_train=200, n_test=50)

    split = build_split(frame)
    is_test_use = frame["use"] == "Test"

    assert (split.loc[is_test_use, SPLIT_COLUMN] == TEST).all()
    assert (split.loc[~is_test_use, SPLIT_COLUMN] != TEST).all()


def test_an_unexpected_use_value_is_refused_rather_than_guessed():
    frame = _toy_frame([{"use": "Training"}, {"use": "Validation"}])

    with pytest.raises(ValueError, match="Validation"):
        build_split(frame)


# --- unit: the grouped validation carve-out ---


def test_validation_protospacers_are_about_a_tenth_of_the_training_ones():
    carved = carve_validation_protospacers(
        [numbered_protospacer(i) for i in range(200)]
    )

    assert len(carved) == 20


def test_validation_carve_does_not_depend_on_row_order():
    protospacers = [numbered_protospacer(i) for i in range(200)]
    shuffled = list(np.random.default_rng(7).permutation(protospacers))

    assert carve_validation_protospacers(protospacers) == (
        carve_validation_protospacers(shuffled)
    )


def test_validation_carve_is_repeatable_under_the_same_seed():
    protospacers = [numbered_protospacer(i) for i in range(200)]

    assert carve_validation_protospacers(protospacers, seed=42) == (
        carve_validation_protospacers(protospacers, seed=42)
    )
    assert carve_validation_protospacers(protospacers, seed=42) != (
        carve_validation_protospacers(protospacers, seed=7)
    )


def test_no_protospacer_appears_in_both_train_and_val():
    frame = _numbered_frame(n_train=200, n_test=20)

    parts = partition(build_split(frame))
    train_protospacers = set(parts[TRAIN]["protospacer"])
    val_protospacers = set(parts[VAL]["protospacer"])

    assert val_protospacers
    assert not train_protospacers & val_protospacers


def test_every_row_of_a_protospacer_lands_in_the_same_split():
    # One protospacer repeated across four rows that differ only at the PAM,
    # as the PAM library does.
    repeated = numbered_protospacer(0)
    frame = _toy_frame(
        [
            *[
                {"sequence": make_sequence(protospacer=repeated, pam=pam)}
                for pam in ("TGG", "AGG", "CGG", "GGG")
            ],
            *[
                {"sequence": make_sequence(protospacer=numbered_protospacer(i))}
                for i in range(1, 50)
            ],
        ]
    )

    split = build_split(frame)
    repeated_rows = split[split["protospacer"] == repeated]

    assert repeated_rows[SPLIT_COLUMN].nunique() == 1


# --- unit: the three splits partition the frame ---


def test_the_three_splits_partition_every_row_exactly_once():
    frame = _numbered_frame(n_train=200, n_test=50)

    split = build_split(frame)
    parts = partition(split)

    assert len(split) == len(frame)
    assert set(split[SPLIT_COLUMN]) <= set(SPLITS)
    assert sum(len(part) for part in parts.values()) == len(frame)
    recombined = pd.concat(parts.values())["row_index"]
    assert sorted(recombined) == list(range(len(frame)))


def test_partition_keeps_the_rows_in_source_order_within_each_split():
    frame = _numbered_frame(n_train=200, n_test=50)

    parts = partition(build_split(frame))

    for part in parts.values():
        assert part["row_index"].is_monotonic_increasing


# --- unit: metadata the later steps stratify on ---


def test_metadata_carries_protospacer_pam_library_membership_and_trna():
    library_protospacer = numbered_protospacer(0)
    frame = _toy_frame(
        [
            {
                "sequence": make_sequence(protospacer=library_protospacer, pam="TGG"),
                "trna_feature": 1,
            },
            {
                "sequence": make_sequence(protospacer=library_protospacer, pam="AGG"),
                "trna_feature": 1,
            },
            {
                "sequence": make_sequence(protospacer=numbered_protospacer(1)),
                "trna_feature": 0,
            },
        ]
    )

    split = build_split(frame)

    assert list(split.columns) == OUTPUT_COLUMNS
    assert split["protospacer"].tolist() == [
        library_protospacer,
        library_protospacer,
        numbered_protospacer(1),
    ]
    assert split["in_pam_library"].tolist() == [True, True, False]
    assert split[TRNA_COLUMN].tolist() == [1, 1, 0]


def test_metadata_carries_the_nine_head_labels_so_training_need_not_reopen_the_file():
    frame = _toy_frame([{"SpCas9": 95.4, "evoCas9": np.nan}])

    split = build_split(frame)

    assert split["SpCas9"].iloc[0] == 95.4
    assert np.isnan(split["evoCas9"].iloc[0])


# --- unit: the written files and the checksum sidecar ---


def test_write_split_writes_one_csv_per_split_plus_the_checksum(tmp_path):
    write_split(build_split(_numbered_frame(200, 50)), tmp_path)

    assert {p.name for p in tmp_path.iterdir()} == {
        "train.csv",
        "val.csv",
        "test.csv",
        CHECKSUM_FILENAME,
    }


def test_checksum_records_the_seed_counts_and_hashes(tmp_path):
    record = write_split(build_split(_numbered_frame(200, 50)), tmp_path, seed=42)

    assert record["seed"] == 42
    assert set(record["rows"]) == {*SPLITS, "total"}
    assert set(record["protospacers"]) == {*SPLITS, "total"}
    assert record["rows"]["total"] == 250
    assert set(record["files"]) == {"train.csv", "val.csv", "test.csv"}
    for name, digest in record["files"].items():
        assert digest == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert json.loads((tmp_path / CHECKSUM_FILENAME).read_text()) == record


def test_checksum_restates_the_counts_the_way_the_published_column_cuts_them(tmp_path):
    record = write_split(build_split(_numbered_frame(200, 50)), tmp_path)

    # `rows.train` is post-carve; `published.rows.train` folds val back in,
    # which is the figure ADR 0004 and CONTEXT.md quote.
    assert record["published"]["rows"][TRAIN] == 200
    assert record["published"]["rows"][TEST] == 50
    assert record["published"]["rows"][TRAIN] == (
        record["rows"][TRAIN] + record["rows"][VAL]
    )
    assert record["published"]["protospacers"][TRAIN] == 200


def test_main_refuses_a_non_default_seed_over_the_published_split(tmp_path):
    # ADR 0004 pins the published split; a sweep must name its own directory.
    with pytest.raises(SystemExit):
        main(["--seed", "7"])

    assert main(["--seed", "7", "--output-dir", str(tmp_path)])["seed"] == 7


def test_checksum_pins_the_spreadsheet_it_was_built_from(tmp_path):
    spreadsheet = tmp_path / "pinned.xlsx"
    spreadsheet.write_bytes(b"not really a spreadsheet")

    record = write_split(
        build_split(_numbered_frame(200, 50)),
        tmp_path,
        dataset_path=spreadsheet,
    )

    assert (
        record["dataset_sha256"] == hashlib.sha256(spreadsheet.read_bytes()).hexdigest()
    )


def test_rerunning_on_an_unchanged_frame_reproduces_identical_files(tmp_path):
    frame = _numbered_frame(200, 50)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"

    first = write_split(build_split(frame), first_dir)
    second = write_split(build_split(frame), second_dir)

    assert first == second
    for name in ("train.csv", "val.csv", "test.csv"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


# --- regression: the real kim_2020_table_s8.xlsx ---


@pytest.fixture(scope="module")
def real_split() -> pd.DataFrame:
    return build_split(filter_any_head(load_spcas9_table()))


def test_real_train_and_test_match_the_published_row_counts(real_split):
    counts = summarize(real_split)["rows"]

    assert counts[TRAIN] + counts[VAL] == 32_810
    assert counts[TEST] == 5_186
    assert counts["total"] == 37_996
    # The carve is 10% of the training protospacers, which lands near - not
    # exactly on - 10% of the training rows.
    assert counts[TRAIN] == 29_134
    assert counts[VAL] == 3_676


def test_real_train_protospacer_count_is_pinned(real_split):
    published_train = real_split[real_split[SPLIT_COLUMN] != TEST]

    assert published_train["protospacer"].nunique() == 20_759


def test_real_train_and_test_share_no_protospacer_and_no_exact_30mer(real_split):
    parts = partition(real_split)
    published_train = pd.concat([parts[TRAIN], parts[VAL]])

    assert not set(published_train["protospacer"]) & set(parts[TEST]["protospacer"])
    assert not set(published_train["sequence"]) & set(parts[TEST]["sequence"])


def test_real_val_is_a_clean_grouped_carve_out_of_the_training_rows(real_split):
    parts = partition(real_split)

    assert not set(parts[TRAIN]["protospacer"]) & set(parts[VAL]["protospacer"])
    assert len(parts[VAL]) > 0


def test_real_split_reproduces_the_committed_checksum_sidecar(real_split, tmp_path):
    """The sidecar is the file under version control; the CSVs are ignored.

    Rebuilding into a scratch directory must land on the same row counts, the
    same protospacer counts and the same per-file digests as the committed
    `split_checksum.json`, which is what makes the published split a fixed
    artifact rather than something each step re-derives.
    """
    committed = json.loads((OUTPUT_DIR / CHECKSUM_FILENAME).read_text())

    rebuilt = write_split(real_split, tmp_path)

    assert rebuilt == committed


def test_real_pam_library_protospacers_stay_intact_on_one_published_side(real_split):
    library = real_split[real_split["in_pam_library"]]
    side = np.where(library[SPLIT_COLUMN] == TEST, "test", "train")
    by_protospacer = pd.DataFrame(
        {"protospacer": library["protospacer"].to_numpy(), "side": side}
    ).drop_duplicates()

    assert library["protospacer"].nunique() == 30
    assert len(by_protospacer) == 30  # no protospacer spans both sides
    assert by_protospacer["side"].value_counts().to_dict() == {"train": 25, "test": 5}
