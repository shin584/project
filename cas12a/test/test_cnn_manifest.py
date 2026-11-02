"""Reading the saved NT membership back, and refusing anything else (#116).

No TensorFlow here: `cas12a.cnn_manifest` answers "which rows, what labels"
and that answer has to be checkable without a deep-learning framework, the
same contract `test_data.py` and `test_features.py` hold for the loader.
`test_cnn_single.py` is where the fit itself is tested.

Every mismatch below is a single tampered field on an otherwise valid manifest,
because that is the failure mode that matters: a manifest that is obviously
broken fails anywhere, while one that is right in every respect but the
membership is exactly what would silently train the wrong model.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cas12a.cnn_manifest import (
    PRODUCTION_IDENTITY,
    ManifestMismatch,
    fitting_set,
    load_library,
    manifest_counts,
    partition_arrays,
    split_config_for,
    verify_against_library,
    verify_manifest,
    verify_production_identity,
)
from cas12a.data import SEQUENCE_COLUMN, TARGET_COLUMN
from cas12a.features import FEATURE_DIM, SEQUENCE_LENGTH
from cas12a.nt_data import (
    PRODUCTION_COUNTS,
    SplitConfig,
    canonical_bytes,
    digest,
    partition_library,
    read_library,
)

#: The #110 NT run's own partition manifest - the production membership #115
#: reuses. Present in a working tree that has the run output and absent in a
#: clean checkout, so the tests that read it skip rather than fail.
NT_PRODUCTION_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "NT_cas12a"
    / "result"
    / "20261101-nt500m-full"
    / "partition.json"
)

#: The fixture library: 8 train / 2 validation / 1 test. Small enough that a
#: failing assertion can name the row, large enough that validation and test
#: hold more than one row between them.
FIXTURE_SPLIT = SplitConfig.fixture(8, 2, 1)
FIXTURE_ROWS = sum(FIXTURE_SPLIT.counts)


def synthetic_rows(count=FIXTURE_ROWS):
    """A TTTN library whose protospacers are distinct by construction.

    Base-four row numbers, as `test_nt_workflow.py` does it - unique without a
    random fixture, so a failure is reproducible from the row number alone.
    """
    rows = []
    for number in range(count):
        protospacer = "".join("ACGT"[(number >> (2 * i)) & 3] for i in range(23))
        rows.append(("ACGT" + "TTTT" + protospacer + "GCA", float(number) * 7.5))
    return rows


def write_source(path, rows=None):
    rows = synthetic_rows() if rows is None else rows
    import pandas as pd

    pd.DataFrame(
        {
            SEQUENCE_COLUMN: [sequence for sequence, _ in rows],
            TARGET_COLUMN: [target for _, target in rows],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def library(tmp_path):
    """`(source_path, manifest)` for the synthetic fixture library.

    The manifest is produced by `nt_data.partition_library` - the module that
    owns splitting - and then only ever read. That is the arrangement under
    test: #116's path consumes what the NT path froze.
    """
    source = write_source(tmp_path / "source.csv")
    data = read_library(source, FIXTURE_SPLIT)
    return source, partition_library(data, FIXTURE_SPLIT)


def tampered(manifest, **changes):
    """A deep copy of `manifest` with top-level keys replaced."""
    return {**json.loads(canonical_bytes(manifest)), **changes}


class TestManifestStructure:
    """`verify_manifest`: self-consistency, before any file is read."""

    def test_a_written_manifest_verifies_and_returns_its_membership(self, library):
        _source, manifest = library

        train, validation, test = verify_manifest(manifest)

        assert (len(train), len(validation), len(test)) == FIXTURE_SPLIT.counts
        assert train == manifest["partitions"]["train"]
        assert sorted(train + validation + test) == list(range(FIXTURE_ROWS))

    def test_counts_decide_the_split_configuration_not_the_caller(self, library):
        """A fixture manifest must not be able to pose as a production run.

        `SplitConfig` refuses non-production counts unless told they are a
        fixture, so reading that flag off the manifest is what keeps an
        11-row rehearsal from being recorded as the 15,000-row training set.
        """
        _source, manifest = library

        assert manifest_counts(manifest) == FIXTURE_SPLIT.counts
        assert split_config_for(manifest).run_kind == "fixture"
        assert (
            split_config_for(
                tampered(
                    manifest,
                    counts=dict(
                        zip(("train", "validation", "test"), PRODUCTION_COUNTS)
                    ),
                )
            ).run_kind
            == "production"
        )

    def test_a_fixture_cannot_pose_as_the_production_run(self, library):
        """11 rows relabelled 12,000/1,500/1,500 fail on the membership length.

        It is the counts that decide `run_kind` and therefore whether the
        pinned production identity is demanded, so this is the attempt worth
        closing: claim the counts, and the membership that is actually there
        no longer matches them.
        """
        _source, manifest = library
        counts = dict(zip(("train", "validation", "test"), PRODUCTION_COUNTS))

        with pytest.raises(ManifestMismatch, match="counts says 12000"):
            verify_manifest(tampered(manifest, counts=counts))

    def test_a_membership_that_does_not_match_its_digest_is_refused(self, library):
        """The digest is recomputed, so a hand-moved row cannot pass.

        Moving one identity from train to validation keeps the counts wrong, so
        this swaps two rows *within* train: counts, coverage and disjointness
        all still hold, and only `partition_sha256` can tell.
        """
        _source, manifest = library
        partitions = json.loads(canonical_bytes(manifest["partitions"]))
        partitions["train"][0], partitions["train"][1] = (
            partitions["train"][1],
            partitions["train"][0],
        )

        with pytest.raises(ManifestMismatch, match="partition_sha256"):
            verify_manifest(tampered(manifest, partitions=partitions))

    def test_an_overlapping_partition_is_refused(self, library):
        _source, manifest = library
        partitions = json.loads(canonical_bytes(manifest["partitions"]))
        partitions["validation"][0] = partitions["train"][0]
        broken = tampered(manifest, partitions=partitions)

        with pytest.raises(ManifestMismatch, match="overlap"):
            verify_manifest(tampered(broken, partition_sha256=digest(partitions)))

    def test_a_foreign_split_algorithm_or_seed_is_refused(self, library):
        """Same shape, different provenance, different population.

        #115 reuses NT's membership specifically; a manifest written by
        another splitter describes rows this CNN has no reason to call "the
        same rows NT trained on", however well-formed the file is.
        """
        _source, manifest = library

        with pytest.raises(ManifestMismatch, match="pinned NT split"):
            verify_manifest(tampered(manifest, algorithm="sklearn.KFold/v1"))
        with pytest.raises(ManifestMismatch, match="pinned NT split"):
            verify_manifest(tampered(manifest, seed=7))

    def test_a_missing_partition_name_is_refused(self, library):
        _source, manifest = library
        partitions = json.loads(canonical_bytes(manifest["partitions"]))
        partitions.pop("test")

        with pytest.raises(ManifestMismatch, match="partitions must be exactly"):
            verify_manifest(tampered(manifest, partitions=partitions))

    def test_counts_disagreeing_with_the_membership_length_are_refused(self, library):
        _source, manifest = library
        counts = {**manifest["counts"], "validation": 3}

        with pytest.raises(ManifestMismatch, match="counts says 3"):
            verify_manifest(tampered(manifest, counts=counts))

    def test_row_identities_must_cover_the_same_rows_as_the_membership(self, library):
        _source, manifest = library
        rows = json.loads(canonical_bytes(manifest["rows"]))
        rows[-1]["source_row"] = FIXTURE_ROWS + 50

        with pytest.raises(ManifestMismatch, match="cover different rows"):
            verify_manifest(tampered(manifest, rows=rows))

    def test_a_frozen_baseline_for_other_rows_is_refused(self, library):
        _source, manifest = library
        baseline = {**manifest["baseline_train_mean"], "training_rows": 3}

        with pytest.raises(ManifestMismatch, match="frozen training baseline"):
            verify_manifest(tampered(manifest, baseline_train_mean=baseline))

    def test_a_non_manifest_is_refused_before_anything_is_indexed(self):
        for value in (None, [], "partition.json", 42):
            with pytest.raises(ManifestMismatch, match="must be a JSON object"):
                verify_manifest(value)


class TestProductionIdentity:
    """The pinned identity of NT's frozen 12,000/1,500/1,500 partition.

    Self-consistency is not enough on its own. `partition_sha256` describes
    whatever membership the file happens to carry, so a forger who moves a row
    and re-digests satisfies every internal check - and a swap between
    `validation` and `test` would then pass the training digest too, because
    the training rows never moved. The pins close that, and only for a
    production manifest: a fixture has no frozen identity to be pinned to.
    """

    def test_the_pins_are_fields_of_a_manifest_not_of_a_run_directory(self):
        assert set(PRODUCTION_IDENTITY) == {
            "manifest_sha256",
            "partition_sha256",
            "training_partition_sha256",
            "source_file_sha256",
            "table_sha256",
        }

    def test_a_production_manifest_from_another_source_library_is_refused(self):
        for key in (
            "source_file_sha256",
            "table_sha256",
            "partition_sha256",
            "training_partition_sha256",
        ):
            with pytest.raises(ManifestMismatch, match=f"production {key}"):
                verify_production_identity({**PRODUCTION_IDENTITY, key: "0" * 64})

    def test_a_re_digested_forgery_fails_the_whole_document_digest(self):
        """Every named pin satisfied, and the document still is not NT's.

        This is the case the document digest exists for: the four field pins
        can all be copied across verbatim, so the only thing a moved
        `validation`/`test` row cannot survive is a digest over the whole file.
        """
        with pytest.raises(ManifestMismatch, match="membership has been altered"):
            verify_production_identity(
                {**PRODUCTION_IDENTITY, "partitions": {"train": [], "test": [0]}}
            )

    @pytest.mark.skipif(
        not NT_PRODUCTION_MANIFEST.exists(),
        reason=f"NT run manifest absent: {NT_PRODUCTION_MANIFEST}",
    )
    def test_the_nt_run_s_own_manifest_verifies_against_the_pins(self):
        """The pins are transcribed from a file, so the file is checked too.

        Without this, a typo in `PRODUCTION_IDENTITY` would reject the real
        partition and the only symptom would be #117 refusing to train.
        """
        manifest = json.loads(NT_PRODUCTION_MANIFEST.read_text(encoding="utf-8"))

        train, validation, test = verify_manifest(manifest)

        assert (len(train), len(validation), len(test)) == PRODUCTION_COUNTS
        assert manifest["run_kind"] == "production"
        assert split_config_for(manifest).run_kind == "production"

    @pytest.mark.skipif(
        not NT_PRODUCTION_MANIFEST.exists(),
        reason=f"NT run manifest absent: {NT_PRODUCTION_MANIFEST}",
    )
    def test_a_validation_test_swap_in_the_real_manifest_is_refused(self):
        """The leak the training digest alone cannot see.

        A Test row inside `validation` is chosen on, not fitted on, so the
        training rows and their digest are untouched, and re-digesting the
        membership satisfies the manifest's *internal* `partition_sha256`
        check. What refuses it is that the pinned `partition_sha256` is a
        constant: the recomputed digest is self-consistent and is not NT's.
        """
        manifest = json.loads(NT_PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
        partitions = json.loads(canonical_bytes(manifest["partitions"]))
        partitions["validation"][0], partitions["test"][0] = (
            partitions["test"][0],
            partitions["validation"][0],
        )
        broken = tampered(
            manifest, partitions=partitions, partition_sha256=digest(partitions)
        )

        with pytest.raises(ManifestMismatch, match="production partition_sha256"):
            verify_manifest(broken)


class TestLibraryIdentity:
    """`verify_against_library`: the manifest against the actual sheet."""

    def test_the_library_it_was_written_from_verifies(self, library):
        source, manifest = library

        by_identity = verify_against_library(manifest, load_library(source, manifest))

        assert sorted(by_identity) == list(range(FIXTURE_ROWS))
        assert all(0.0 <= row["target"] <= 1.0 for row in by_identity.values())

    def test_a_different_source_file_is_refused(self, tmp_path, library):
        """One changed label is a different library, even at the same shape.

        The target of row 0 moves, so every per-row sequence checksum still
        matches and only the source/table digests disagree - which is the point
        of recording them separately from the membership.
        """
        _source, manifest = library
        rows = synthetic_rows()
        rows[0] = (rows[0][0], 99.0)
        other = write_source(tmp_path / "other.csv", rows)

        with pytest.raises(ManifestMismatch, match="source_file_sha256 differs"):
            verify_against_library(manifest, load_library(other, manifest))

    def test_a_relabelled_training_row_fails_the_frozen_baseline(
        self, tmp_path, library
    ):
        """The baseline mean is the training partition's only label evidence.

        Bypassing the file digests (they are asserted above) leaves the
        training-partition digest and the frozen mean as the two things that
        still have to agree, and they are what a changed label moves.
        """
        _source, manifest = library
        rows = synthetic_rows()
        train_row = manifest["partitions"]["train"][0]
        rows[train_row] = (rows[train_row][0], 100.0)
        other = write_source(tmp_path / "relabelled.csv", rows)
        data = load_library(other, manifest)

        with pytest.raises(ManifestMismatch, match="training_partition_sha256"):
            verify_against_library(
                tampered(
                    manifest,
                    source_file_sha256=data["source_file_sha256"],
                    table_sha256=data["table_sha256"],
                ),
                data,
            )

    def test_a_changed_sequence_fails_its_recorded_checksum(self, tmp_path, library):
        _source, manifest = library
        rows = synthetic_rows()
        rows[0] = (
            rows[0][0][:33] + ("T" if rows[0][0][33] != "T" else "A"),
            rows[0][1],
        )
        data = load_library(write_source(tmp_path / "mutated.csv", rows), manifest)

        with pytest.raises(ManifestMismatch, match="sequence checksums"):
            verify_against_library(
                tampered(
                    manifest,
                    source_file_sha256=data["source_file_sha256"],
                    table_sha256=data["table_sha256"],
                ),
                data,
            )

    def test_a_duplicate_protospacer_never_reaches_the_partition_check(
        self, tmp_path, library
    ):
        """`read_library` is the gate for duplicates, and it stays the gate.

        A protospacer shared between two rows would put the same guide in two
        partitions, so this asserts the library loader refuses the sheet
        outright rather than leaving the check to the manifest comparison.
        """
        _source, manifest = library
        rows = synthetic_rows()
        # Row 0's protospacer under a different right flank, so the 34 bp
        # sequences still differ and the duplicate is the protospacer alone.
        rows[5] = (rows[5][0][:8] + rows[0][0][8:31] + "TAC", rows[5][1])

        with pytest.raises(ValueError, match="duplicate protospacer"):
            load_library(write_source(tmp_path / "duplicate.csv", rows), manifest)


class TestFittingSet:
    """`fitting_set`: the arrays a single fit is handed, and nothing else."""

    def test_only_train_and_validation_come_back_as_arrays(self, library):
        source, manifest = library

        prepared = fitting_set(source, manifest)

        assert sorted(prepared["partitions"]) == ["train", "validation"]
        assert prepared["held_out"]["name"] == "test"
        assert prepared["held_out"]["source_rows"] == manifest["partitions"]["test"]
        assert set(prepared["held_out"]) == {"name", "source_rows"}

    def test_the_arrays_are_the_manifest_s_rows_in_the_manifest_s_order(self, library):
        """Row order is membership order, so an array offset can be joined back.

        Nothing downstream may infer identity from position alone, but the
        mapping has to exist and be stated: `source_rows[i]` is the row that
        produced `X_seq[i]`.
        """
        source, manifest = library
        prepared = fitting_set(source, manifest)

        for name, members in (
            ("train", manifest["partitions"]["train"]),
            ("validation", manifest["partitions"]["validation"]),
        ):
            partition = prepared["partitions"][name]
            assert partition["source_rows"] == members
            assert partition["X_seq"].shape == (len(members), SEQUENCE_LENGTH, 4)
            assert partition["X_feat"].shape == (len(members), FEATURE_DIM)
            assert partition["y"].shape == (len(members),)
            for offset, identity in enumerate(members):
                expected = prepared["by_identity"][identity]
                assert partition["y"][offset] == pytest.approx(expected["target"])

    def test_the_encoding_is_the_canonical_one(self, library):
        """One-hot rows sum to 1 and the 22 dinucleotide pairs are all set.

        Both follow from `cas12a.features` being the single definition of the
        encoding; the assertion is here because a drifting copy would produce
        arrays of the right shape and the wrong contents (#91).
        """
        source, manifest = library
        train = fitting_set(source, manifest)["partitions"]["train"]

        assert np.all(train["X_seq"].sum(axis=2) == 1)
        assert np.all(train["X_feat"][:, 2:].sum(axis=1) == 22)
        assert train["X_seq"].dtype == train["X_feat"].dtype == np.float32

    def test_a_test_row_moved_into_training_is_refused(self, library):
        """A re-digested swap still has to answer to the frozen train digest.

        Swapping a train row for a Test one keeps counts, coverage and
        disjointness intact, and the forgery re-digests the membership - so
        `partition_sha256` is satisfied. What is left is
        `training_partition_sha256`, which digests the training *rows* rather
        than their indices, and that is what refuses the fit.
        """
        source, manifest = library
        partitions = json.loads(canonical_bytes(manifest["partitions"]))
        held_out = partitions["test"][0]
        partitions["train"][0], partitions["test"][0] = held_out, partitions["train"][0]
        broken = tampered(
            manifest, partitions=partitions, partition_sha256=digest(partitions)
        )

        with pytest.raises(ManifestMismatch, match="training_partition_sha256"):
            fitting_set(source, broken)

    def test_no_test_row_s_features_or_label_are_built(self, library):
        source, manifest = library
        prepared = fitting_set(source, manifest)

        fitted = {
            identity
            for partition in prepared["partitions"].values()
            for identity in partition["source_rows"]
        }
        assert not fitted.intersection(manifest["partitions"]["test"])
        assert len(fitted) == sum(FIXTURE_SPLIT.counts[:2])

    def test_building_arrays_rejects_a_row_the_features_cannot_encode(self, library):
        """A non-ACGT row would one-hot to zeros and feature to a short sum.

        `read_library` rejects such a row, so this calls `partition_arrays`
        directly: the guard belongs to the array builder too, because #115
        re-encodes mutated sequences through it and a silently zeroed position
        there would read as a biological result.
        """
        source, manifest = library
        by_identity = dict(fitting_set(source, manifest)["by_identity"])
        identity = manifest["partitions"]["train"][0]
        row = by_identity[identity]
        by_identity[identity] = {**row, "sequence": row["sequence"][:20] + "N" * 14}

        with pytest.raises(ManifestMismatch, match="dinucleotide pairs"):
            partition_arrays(by_identity, [identity])


def test_the_manifest_path_never_writes_or_resplits(library, monkeypatch):
    """#116's rule, enforced rather than reviewed.

    `partition_library` and `save_or_verify_partition` are how a new membership
    comes into existence, and `KFold` is how the historical five-fold path
    makes one. If `fitting_set` can run with all three replaced by explosives,
    it read the saved membership and derived nothing.
    """
    import sklearn.model_selection

    from cas12a import cnn_manifest, nt_data

    source, manifest = library

    def forbidden(*args, **kwargs):
        raise AssertionError("the manifest path must not construct a split")

    monkeypatch.setattr(nt_data, "partition_library", forbidden)
    monkeypatch.setattr(nt_data, "save_or_verify_partition", forbidden)
    monkeypatch.setattr(sklearn.model_selection, "KFold", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)

    prepared = fitting_set(source, manifest)

    assert len(prepared["partitions"]["train"]["source_rows"]) == 8
    assert not hasattr(cnn_manifest, "KFold")
