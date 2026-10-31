"""NT preparation behavior through the public workflow and its saved evidence."""

import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cas12a.data import SEQUENCE_COLUMN, TARGET_COLUMN, read_cas12a_table
from cas12a.nt_workflow import NTWorkflow, SplitConfig


def synthetic_table(count=10):
    # Base-four row numbers give unique protospacers without random fixtures.
    sequences = []
    for number in range(count):
        bases = "".join("ACGT"[(number >> (2 * i)) & 3] for i in range(23))
        sequences.append("ACGT" + "TTTT" + bases + "GCA")
    return pd.DataFrame(
        {
            "library_id": [f"target-{i}" for i in range(count)],
            SEQUENCE_COLUMN: sequences,
            TARGET_COLUMN: [float(10 * i) for i in range(count)],
        }
    )


def workflow(tmp_path, table=None, *, split=None):
    source = tmp_path / "source.csv"
    if table is not None:
        table.to_csv(source, index=False)
    return NTWorkflow(
        source=source,
        partition_path=tmp_path / "partition.json",
        split=split or SplitConfig.fixture(8, 1, 1),
        local_work_root=tmp_path / "local",
        drive_output_root=tmp_path / "drive",
    )


def test_preparation_preserves_source_and_freezes_training_baseline(tmp_path):
    table = synthetic_table().astype(object)
    table.loc[0, SEQUENCE_COLUMN] = "  " + table.loc[0, SEQUENCE_COLUMN].lower() + " "
    table.loc[0, TARGET_COLUMN] = -5
    table.loc[9, TARGET_COLUMN] = 140
    table.loc[10] = ["no-sequence", None, 50]
    table.loc[11] = ["no-target", table.loc[1, SEQUENCE_COLUMN], None]
    table.loc[12] = ["bad-target", table.loc[2, SEQUENCE_COLUMN], "bad"]
    runner = workflow(tmp_path, table)

    run = runner.prepare(run_id="fixture-first")

    data = json.loads((run.local_dir / "data.json").read_text(encoding="utf-8"))
    partition = json.loads(runner.partition_path.read_text(encoding="utf-8"))
    assert data["columns"] == list(table.columns)
    assert data["rows"][0]["raw"][SEQUENCE_COLUMN] == table.loc[0, SEQUENCE_COLUMN]
    assert data["rows"][0]["sequence"] == table.loc[0, SEQUENCE_COLUMN].strip().upper()
    assert data["rows"][0]["target"] == 0
    assert data["rows"][9]["target"] == 1
    assert [row["source_row"] for row in data["rows"]] == list(range(10))
    assert [(row["source_row"], row["reasons"]) for row in data["excluded"]] == [
        (10, ["missing_sequence"]),
        (11, ["missing_target"]),
        (12, ["unparseable_target"]),
    ]
    assert partition["partitions"] == {
        "train": [5, 6, 0, 7, 3, 2, 4, 9],
        "validation": [1],
        "test": [8],
    }
    assert partition["baseline_train_mean"] == {
        "value": pytest.approx(0.4625),
        "training_rows": 8,
        "training_partition_sha256": partition["training_partition_sha256"],
    }
    assert (
        data["source_file_sha256"]
        == hashlib.sha256(runner.source.read_bytes()).hexdigest()
    )
    assert run.status["state"] == "prepared"
    assert run.status["run_kind"] == "fixture"
    assert run.status["model_completed"] is False
    assert run.checkpoint_dir == run.local_dir / "checkpoints"
    for filename in (
        "data.json",
        "partition.json",
        "configuration.json",
        "status.json",
    ):
        assert (run.local_dir / filename).read_bytes() == (
            run.durable_dir / filename
        ).read_bytes()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("short", "34 bp ACGT"),
        ("long", "34 bp ACGT"),
        ("unknown-base", "34 bp ACGT"),
        ("internal-space", "34 bp ACGT"),
        ("wrong-pam", "TTTN PAM"),
        ("infinity", "nonfinite target"),
        ("negative-infinity", "nonfinite target"),
        ("nan-text", "nonfinite target"),
        ("duplicate-sequence", "duplicate sequence"),
        ("duplicate-protospacer", "duplicate protospacer"),
        ("extra-row", "usable row count"),
        ("missing-row", "usable row count"),
        ("wrong-schema", "source schema"),
    ],
)
def test_bad_libraries_are_rejected_before_reserving_a_run(tmp_path, change, message):
    table = synthetic_table()
    sequence = table.loc[0, SEQUENCE_COLUMN]
    if change == "short":
        table.loc[0, SEQUENCE_COLUMN] = sequence[:-1]
    elif change == "long":
        table.loc[0, SEQUENCE_COLUMN] = sequence + "A"
    elif change == "unknown-base":
        table.loc[0, SEQUENCE_COLUMN] = "N" + sequence[1:]
    elif change == "internal-space":
        table.loc[0, SEQUENCE_COLUMN] = sequence[:10] + " " + sequence[11:]
    elif change == "wrong-pam":
        table.loc[0, SEQUENCE_COLUMN] = sequence[:4] + "ATTT" + sequence[8:]
    elif change in ("infinity", "negative-infinity", "nan-text"):
        table[TARGET_COLUMN] = table[TARGET_COLUMN].astype(object)
        table.loc[0, TARGET_COLUMN] = {
            "infinity": "inf",
            "negative-infinity": "-inf",
            "nan-text": "NaN",
        }[change]
    elif change == "duplicate-sequence":
        table.loc[1, SEQUENCE_COLUMN] = sequence.lower()
    elif change == "duplicate-protospacer":
        table.loc[1, SEQUENCE_COLUMN] = "TGCA" + sequence[4:]
    elif change == "extra-row":
        table = synthetic_table(11)
    elif change == "missing-row":
        table = table.iloc[:-1]
    elif change == "wrong-schema":
        table = table.rename(columns={TARGET_COLUMN: "wrong target"})
    runner = workflow(tmp_path, table)
    with pytest.raises(ValueError, match=message):
        runner.prepare(run_id="rejected")
    assert not runner.partition_path.exists()
    assert not (runner.local_work_root / "rejected").exists()
    assert not (runner.drive_output_root / "rejected").exists()


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".xlsx"])
def test_equivalent_repeated_reads_preserve_all_columns_and_identities(
    tmp_path, suffix
):
    table = synthetic_table()
    table.loc[0, TARGET_COLUMN] = 73.4
    runner = workflow(tmp_path)
    source = runner.source.with_suffix(suffix)
    if suffix == ".xlsx":
        table.to_excel(source, index=False)
    else:
        table.to_csv(source, sep="\t" if suffix == ".tsv" else ",", index=False)
    runner = NTWorkflow(
        source,
        runner.partition_path,
        runner.local_work_root,
        runner.drive_output_root,
        runner.split,
    )
    first = runner.prepare(run_id="first")
    saved = runner.partition_path.read_bytes()
    second = runner.prepare(run_id="second")
    assert runner.partition_path.read_bytes() == saved
    assert (first.local_dir / "data.json").read_bytes() == (
        second.local_dir / "data.json"
    ).read_bytes()
    data = json.loads((first.local_dir / "data.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["target"] == pytest.approx(0.734)
    assert {row["sequence"][4:8] for row in data["rows"]} == {"TTTT"}


def test_duplicate_source_column_is_a_schema_error(tmp_path):
    table = synthetic_table()
    table["duplicate"] = table["library_id"]
    table.columns = ["library_id", SEQUENCE_COLUMN, TARGET_COLUMN, "library_id"]
    runner = workflow(tmp_path, table)
    with pytest.raises(ValueError, match="source schema"):
        runner.prepare()


def test_surplus_source_fields_cannot_become_an_unrecorded_index(tmp_path):
    runner = workflow(tmp_path, synthetic_table())
    records = list(csv.reader(io.StringIO(runner.source.read_text())))
    with runner.source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(records[0])
        writer.writerows(["hidden-index", *record] for record in records[1:])
    with pytest.raises(ValueError, match="source schema"):
        runner.prepare()


def test_nt_validation_does_not_change_the_permissive_cnn_loader(tmp_path):
    table = synthetic_table()
    table.loc[0, SEQUENCE_COLUMN] = table.loc[0, SEQUENCE_COLUMN][:-1]
    runner = workflow(tmp_path, table)
    legacy = read_cas12a_table(runner.source)
    assert legacy.loc[0, "sequence"].endswith("N")
    with pytest.raises(ValueError, match="34 bp ACGT"):
        runner.prepare()
    assert read_cas12a_table(runner.source).equals(legacy)


def test_baseline_uses_no_validation_or_test_labels(tmp_path):
    original = tmp_path / "original"
    changed = tmp_path / "changed"
    original.mkdir()
    changed.mkdir()
    table = synthetic_table()
    first = workflow(original, table).prepare()
    # Seed-42 fixture has validation row 1 and test row 8.
    table.loc[1, TARGET_COLUMN] = 100
    table.loc[8, TARGET_COLUMN] = 0
    second = workflow(changed, table).prepare()
    first_partition = json.loads((first.local_dir / "partition.json").read_text())
    second_partition = json.loads((second.local_dir / "partition.json").read_text())
    assert first_partition["baseline_train_mean"]["value"] == pytest.approx(0.45)
    assert (
        first_partition["baseline_train_mean"]
        == second_partition["baseline_train_mean"]
    )


def test_exclusions_keep_gapped_source_identities_and_multiple_reasons(tmp_path):
    table = synthetic_table().astype(object)
    table.loc[1, SEQUENCE_COLUMN] = "  "
    table.loc[1, TARGET_COLUMN] = ""
    table.loc[10] = ["replacement", synthetic_table(11).loc[10, SEQUENCE_COLUMN], 40]
    runner = workflow(tmp_path, table)
    run = runner.prepare()
    partition = json.loads((run.local_dir / "partition.json").read_text())
    assert partition["partitions"] == {
        "train": [6, 7, 0, 8, 4, 3, 5, 10],
        "validation": [2],
        "test": [9],
    }
    data = json.loads((run.local_dir / "data.json").read_text())
    assert data["excluded"][0]["reasons"] == ["missing_sequence", "missing_target"]


def test_production_does_not_accept_fixture_counts(tmp_path):
    runner = workflow(tmp_path, synthetic_table(), split=SplitConfig())
    with pytest.raises(ValueError, match="expected 15000, got 10"):
        runner.prepare()


def test_preparation_needs_no_training_runtime(tmp_path):
    runner = workflow(tmp_path, synthetic_table())
    script = f"""
import sys
from cas12a.nt_workflow import NTWorkflow, SplitConfig
NTWorkflow(source={str(runner.source)!r}, partition_path={str(runner.partition_path)!r},
    local_work_root={str(runner.local_work_root)!r}, drive_output_root={str(runner.drive_output_root)!r},
    split=SplitConfig.fixture(8, 1, 1)).prepare(run_id='no-runtime')
assert not any(name in sys.modules for name in ('torch', 'tensorflow', 'transformers'))
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )


def test_failed_durable_copy_records_failure_in_both_locations(tmp_path, monkeypatch):
    runner = workflow(tmp_path, synthetic_table())
    original_open = Path.open

    def disk_failure(path, mode="r", *args, **kwargs):
        if (
            path == runner.drive_output_root / "interrupted" / "data.json"
            and mode == "xb"
        ):
            raise OSError("simulated Drive write failure")
        return original_open(path, mode, *args, **kwargs)

    # Filesystem is the external boundary; no workflow collaborator is mocked.
    monkeypatch.setattr(Path, "open", disk_failure)
    with pytest.raises(OSError, match="simulated Drive write failure"):
        runner.prepare(run_id="interrupted")
    for root in (runner.local_work_root, runner.drive_output_root):
        status = json.loads((root / "interrupted" / "status.json").read_text())
        assert status["state"] == "preparation_failed"
        assert status["model_completed"] is False
        assert "simulated Drive write failure" in status["error"]


@pytest.mark.parametrize(
    "change",
    [
        "source-bytes",
        "source-metadata",
        "source-target",
        "source-sequence",
        "source-digest",
        "table-digest",
        "row-identity",
        "sequence-digest",
        "membership",
        "rehashed-membership",
        "duplicate-member",
        "missing-member",
        "seed",
        "algorithm",
        "count",
        "baseline",
        "training-identity",
        "version",
        "unreadable",
        "split-configuration",
    ],
)
def test_reuse_refuses_changed_evidence_without_repair_or_resplitting(tmp_path, change):
    runner = workflow(tmp_path, synthetic_table())
    first = runner.prepare(run_id="original")
    original_artifacts = {
        p.name: p.read_bytes() for p in first.durable_dir.glob("*.json")
    }
    manifest = json.loads(runner.partition_path.read_text())
    if change == "source-bytes":
        with runner.source.open("ab") as handle:
            handle.write(b"\n")
    elif change.startswith("source-") and change != "source-digest":
        table = synthetic_table()
        if change == "source-metadata":
            table.loc[0, "library_id"] = "renamed"
        elif change == "source-target":
            table.loc[0, TARGET_COLUMN] = 25
        elif change == "source-sequence":
            table.loc[0, SEQUENCE_COLUMN] = table.loc[0, SEQUENCE_COLUMN][:-3] + "AAA"
        table.to_csv(runner.source, index=False)
    elif change == "split-configuration":
        runner = workflow(tmp_path, split=SplitConfig.fixture(7, 2, 1))
    else:
        if change == "source-digest":
            manifest["source_file_sha256"] = "0" * 64
        elif change == "table-digest":
            manifest["table_sha256"] = "0" * 64
        elif change == "row-identity":
            manifest["rows"][0]["source_row"] = 999
        elif change == "sequence-digest":
            manifest["rows"][0]["sequence_sha256"] = "0" * 64
        elif change in ("membership", "rehashed-membership"):
            manifest["partitions"]["train"][0], manifest["partitions"]["test"][0] = (
                manifest["partitions"]["test"][0],
                manifest["partitions"]["train"][0],
            )
            if change == "rehashed-membership":
                manifest["partition_sha256"] = hashlib.sha256(
                    json.dumps(
                        manifest["partitions"], sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
        elif change == "duplicate-member":
            manifest["partitions"]["train"][0] = manifest["partitions"]["train"][1]
        elif change == "missing-member":
            manifest["partitions"]["train"].pop()
        elif change == "seed":
            manifest["seed"] = 43
        elif change == "algorithm":
            manifest["algorithm"] = "different"
        elif change == "count":
            manifest["counts"]["train"] = 7
        elif change == "baseline":
            manifest["baseline_train_mean"]["value"] = 0.9
        elif change == "training-identity":
            manifest["baseline_train_mean"]["training_partition_sha256"] = "0" * 64
        elif change == "version":
            manifest["serialization"] = "v999"
        runner.partition_path.write_text(
            "{" if change == "unreadable" else json.dumps(manifest)
        )
    saved_bytes = runner.partition_path.read_bytes()
    with pytest.raises(ValueError, match="refusing to resplit"):
        runner.prepare(run_id="must-not-exist")
    assert runner.partition_path.read_bytes() == saved_bytes
    assert not (runner.local_work_root / "must-not-exist").exists()
    assert not (runner.drive_output_root / "must-not-exist").exists()
    assert original_artifacts == {
        p.name: p.read_bytes() for p in first.durable_dir.glob("*.json")
    }


def test_full_library_partition_has_pinned_membership_and_no_sequence_leakage(tmp_path):
    runner = workflow(
        tmp_path, synthetic_table(15000), split=SplitConfig.fixture(12000, 1500, 1500)
    )
    run = runner.prepare(run_id="full-size-fixture")
    manifest = json.loads(runner.partition_path.read_text())
    assert manifest["counts"] == {"train": 12000, "validation": 1500, "test": 1500}
    assert manifest["seed"] == 42
    # Golden SHA-256 of compact JSON arrays from the agreed NumPy permutation.
    expected = {
        "train": "9d47d685941abf7bb976a9c7a97e7eb475c85eeacd696d39a2a35b3993abea02",
        "validation": "d5a3216523d23a47ba60321a25ae606d26b7493ecdfc735b38445c22f119c5f5",
        "test": "fcd5d6544572b0472debdaca7b8c6de2dcc5a460953ca05c83e6b04201ad1455",
    }
    data = json.loads((run.local_dir / "data.json").read_text())
    by_id = {row["source_row"]: row for row in data["rows"]}
    seen_rows, seen_sequences, seen_protospacers = set(), set(), set()
    for name, members in manifest["partitions"].items():
        assert (
            hashlib.sha256(
                json.dumps(members, separators=(",", ":")).encode()
            ).hexdigest()
            == expected[name]
        )
        sequences = {by_id[i]["sequence"] for i in members}
        protospacers = {by_id[i]["protospacer"] for i in members}
        assert len(members) == len(set(members)) == len(sequences) == len(protospacers)
        assert not (
            set(members) & seen_rows
            or sequences & seen_sequences
            or protospacers & seen_protospacers
        )
        seen_rows.update(members)
        seen_sequences.update(sequences)
        seen_protospacers.update(protospacers)
    assert seen_rows == set(range(15000))
    assert run.status["run_kind"] == "fixture"
    assert run.status["model_completed"] is False


@pytest.mark.parametrize("location", ["local", "durable"])
def test_existing_run_destinations_are_preserved(tmp_path, location):
    runner = workflow(tmp_path, synthetic_table())
    root = runner.local_work_root if location == "local" else runner.drive_output_root
    destination = root / "occupied"
    destination.mkdir(parents=True)
    (destination / "status.json").write_text("previous run")
    with pytest.raises(FileExistsError, match="run destination already exists"):
        runner.prepare(run_id="occupied")
    assert (destination / "status.json").read_text() == "previous run"
    assert not runner.partition_path.exists()
    assert not (
        (runner.drive_output_root if location == "local" else runner.local_work_root)
        / "occupied"
    ).exists()


@pytest.mark.parametrize("run_id", ["../escape", "", "a/b", "a\\b", "a" * 81])
def test_run_identity_cannot_escape_its_destinations(tmp_path, run_id):
    runner = workflow(tmp_path, synthetic_table())
    with pytest.raises(ValueError, match="run_id"):
        runner.prepare(run_id=run_id)


def test_work_and_drive_locations_cannot_overlap(tmp_path):
    with pytest.raises(ValueError, match="non-overlapping"):
        NTWorkflow(
            source=tmp_path / "source.csv",
            partition_path=tmp_path / "partition.json",
            local_work_root=tmp_path / "same",
            drive_output_root=tmp_path / "same" / "nested",
        )


def test_canonical_evidence_has_independently_verifiable_hashes(tmp_path):
    runner = workflow(tmp_path, synthetic_table())
    run = runner.prepare()
    data = json.loads((run.local_dir / "data.json").read_text())
    assert data["serialization"] == "cas12a-nt-json-v1"
    canonical_table = {
        "serialization": "cas12a-nt-json-v1",
        "columns": data["columns"],
        "rows": [
            {"source_row": row["source_row"], "raw": row["raw"]} for row in data["rows"]
        ],
    }
    encoded = json.dumps(
        canonical_table,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == data["table_sha256"]
    manifest = json.loads(runner.partition_path.read_text())
    for row, identity in zip(data["rows"], manifest["rows"]):
        assert identity == {
            "source_row": row["source_row"],
            "sequence_sha256": hashlib.sha256(
                row["sequence"].encode("ascii")
            ).hexdigest(),
        }
