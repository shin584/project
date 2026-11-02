"""Reading - never writing - the saved NT partition for a CNN fit (#116).

`cas12a/nt_data.py` is the module that *creates* the 12,000/1,500/1,500
membership and freezes it; this one only reads it back. The distinction is the
point of #116: the CNN that #115 compares against NT has to be fitted on the
same rows NT was fitted on, and a second splitter - even one seeded the same
way - would quietly produce a different answer to "which rows are these".
So nothing here calls `partition_library`, `save_or_verify_partition` or
`sklearn`'s `KFold`, and `verify_manifest` rejects a manifest whose recorded
`algorithm`/`seed` are not the ones `nt_data` wrote.

Verification runs in two halves because they fail for different reasons:

- `verify_manifest` is structure and self-consistency, and needs no source
  file: names, counts, integer identities, disjointness, and the
  `partition_sha256` recomputed from the membership it is supposed to describe.
- `verify_against_library` is identity against the actual Kim 2018 sheet, read
  through `nt_data.read_library` so the strict sequence/PAM/label checks are
  the same ones the NT run applied: source and table digests, per-row sequence
  checksums, the training-partition digest and its frozen baseline mean.

Pure NumPy/pandas, like `features` and `data` - the package contract in
`cas12a/__init__.py` keeps TensorFlow out of the path that answers "which rows
and what are their labels". `cas12a/cnn_single.py` is where TensorFlow enters.
"""

import numpy as np

from cas12a.features import (
    DINUCLEOTIDE_POSITIONS,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    extract_seq_features,
    one_hot_encode,
    protospacer_of,
)
from cas12a.nt_data import (
    ALGORITHM,
    PARTITION_NAMES,
    PRODUCTION_COUNTS,
    SEED,
    SERIALIZATION,
    SplitConfig,
    digest,
    read_library,
)
from cas12a.units import SCORE_MAX, SCORE_MIN

#: The partitions a CNN fit is allowed to touch. `test` is read - its
#: membership is needed to prove the other two exclude it - and never returned
#: as arrays by `partition_arrays`.
FITTING_NAMES = ("train", "validation")
HELD_OUT_NAME = "test"

#: The identity of NT's frozen production partition, as the #110 run
#: (`cas12a/models/NT_cas12a/result/20261101-nt500m-full/`) wrote it. These are
#: pinned rather than merely recomputed because self-consistency cannot catch a
#: *consistently* altered manifest: `partition_sha256` covers the membership,
#: so anyone who moves a row and re-digests passes every internal check, and a
#: swap between `validation` and `test` would then put a Test row into model
#: selection with nothing else to object. `manifest_sha256` is the whole
#: document - the value NT's own `configuration.json` carries as
#: `partition_manifest_sha256` - so it fixes all three memberships at once.
#: A fixture manifest has no pinned identity and is checked for consistency
#: only; `run_kind` is what separates the two, and it is derived from the
#: counts rather than read as a free-text field.
PRODUCTION_IDENTITY = {
    "manifest_sha256": (
        "e45c8b1128f85ba90b6f2bd17921b64b069081686321bc2809afc1fc8a3bdc9c"
    ),
    "partition_sha256": (
        "680fc75f023e49ce6baae41a581e00d9ba06da024294691964f4ff1f03942f6d"
    ),
    "training_partition_sha256": (
        "383bb97d91be31ce8220d95e9c0c39c908a4309f6cd9323dd465809894d23029"
    ),
    "source_file_sha256": (
        "c465925148023dcfe01e388c79c17ccaf73bc454c815eaef72595ba6ee71d028"
    ),
    "table_sha256": (
        "25ac3a282b31193a31067168bd3c7ec9cdf2b34ce2ffb30803f2f4bf5e640c3c"
    ),
}

#: Dinucleotide block width of a clean 23 bp protospacer: every one of the 22
#: overlapping pairs sets exactly one of its 16 slots. A row that misses it has
#: a non-ACGT base, which `read_library` already rejects - checking the sum
#: here catches the other direction, a feature function that stopped encoding.
CLEAN_DINUCLEOTIDE_SUM = float(DINUCLEOTIDE_POSITIONS)


class ManifestMismatch(ValueError):
    """The saved membership is not what it claims to be, or not what is here.

    Its own type because the caller's response differs from any other failure:
    a mismatch means the partition on disk and the library in front of it
    disagree, so no fit may proceed and nothing may be rewritten to agree.
    """


def split_config_for(manifest):
    """The `SplitConfig` describing an already-written manifest.

    Production counts give the production config; anything else is a fixture,
    which is what lets `cas12a/test/` exercise this path on 11 synthetic rows.
    `SplitConfig` itself refuses non-production counts unless told they are a
    fixture, so this is the one place that decision is read off the manifest
    rather than chosen by a caller.
    """
    counts = manifest_counts(manifest)
    if counts == PRODUCTION_COUNTS:
        return SplitConfig()
    return SplitConfig.fixture(*counts)


def manifest_counts(manifest):
    """`(train, validation, test)` as the manifest's `counts` records them."""
    try:
        counts = manifest["counts"]
        return tuple(int(counts[name]) for name in PARTITION_NAMES)
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestMismatch(f"unreadable partition counts: {error}") from error


def _identities(manifest, name):
    members = manifest["partitions"][name]
    if not isinstance(members, list) or any(
        type(identity) is not int or identity < 0 for identity in members
    ):
        raise ManifestMismatch(
            f"{name} membership must be a list of non-negative source rows"
        )
    return members


def verify_manifest(manifest):
    """Check a loaded manifest against itself; return its `(train, val, test)`.

    Everything here is cheap and needs no data file, so it runs before the
    source sheet is read: a manifest that is internally inconsistent cannot be
    made consistent by what the sheet contains.

    `partition_sha256` is recomputed rather than trusted. The digest covers the
    membership dict, so it is what distinguishes "this is the NT partition"
    from "this is a file with the right shape" - and recomputing it is the only
    way the distinction survives a hand-edited manifest.
    """
    if not isinstance(manifest, dict):
        raise ManifestMismatch("partition manifest must be a JSON object")
    if manifest.get("serialization") != SERIALIZATION:
        raise ManifestMismatch(
            f"expected {SERIALIZATION} serialization, "
            f"got {manifest.get('serialization')!r}"
        )
    # The split this CNN reuses is NT's, so its provenance is part of its
    # identity: a manifest written by some other algorithm or seed is a
    # different population however well-formed it is.
    if manifest.get("algorithm") != ALGORITHM or manifest.get("seed") != SEED:
        raise ManifestMismatch(
            "partition manifest was not written by the pinned NT split "
            f"({ALGORITHM!r}, seed {SEED})"
        )
    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(PARTITION_NAMES):
        raise ManifestMismatch(
            f"partitions must be exactly {', '.join(PARTITION_NAMES)}"
        )

    counts = manifest_counts(manifest)
    members = {name: _identities(manifest, name) for name in PARTITION_NAMES}
    for name, count in zip(PARTITION_NAMES, counts):
        if len(members[name]) != count:
            raise ManifestMismatch(
                f"{name} membership holds {len(members[name])} rows, "
                f"counts says {count}"
            )
    run_kind = "production" if counts == PRODUCTION_COUNTS else "fixture"
    if manifest.get("run_kind") != run_kind:
        raise ManifestMismatch(
            f"counts {counts} describe a {run_kind} run, manifest says "
            f"{manifest.get('run_kind')!r}"
        )

    # Disjointness is checked across all three at once: a row shared between
    # train and test is the failure this whole module exists to prevent, and a
    # row repeated inside one partition would silently reweight it.
    flattened = [identity for name in PARTITION_NAMES for identity in members[name]]
    if len(set(flattened)) != len(flattened):
        raise ManifestMismatch("partitions overlap or repeat a source row")
    if digest(partitions) != manifest.get("partition_sha256"):
        raise ManifestMismatch(
            "partition_sha256 does not describe the recorded membership"
        )

    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != len(flattened):
        raise ManifestMismatch(
            f"row identities cover {len(rows) if isinstance(rows, list) else '?'} "
            f"rows, membership covers {len(flattened)}"
        )
    row_identities = [row.get("source_row") for row in rows]
    if row_identities != sorted(set(row_identities)):
        raise ManifestMismatch("row identities must ascend without repetition")
    if set(row_identities) != set(flattened):
        raise ManifestMismatch("membership and row identities cover different rows")

    baseline = manifest.get("baseline_train_mean")
    if (
        not isinstance(baseline, dict)
        or baseline.get("training_rows") != counts[0]
        or baseline.get("training_partition_sha256")
        != manifest.get("training_partition_sha256")
    ):
        raise ManifestMismatch(
            "frozen training baseline does not describe the train partition"
        )
    if run_kind == "production":
        verify_production_identity(manifest)
    return tuple(members[name] for name in PARTITION_NAMES)


def verify_production_identity(manifest):
    """Pin a production manifest to NT's frozen partition, field by field.

    The whole-document digest is checked last and is the one that cannot be
    satisfied by a re-digested forgery; the named fields are checked first only
    so that a failure says which part of the identity moved instead of
    reporting one opaque hash.
    """
    expected = dict(PRODUCTION_IDENTITY)
    document = expected.pop("manifest_sha256")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ManifestMismatch(
                f"production {key} is not NT's frozen partition: expected "
                f"{value}, got {manifest.get(key)!r}"
            )
    if digest(manifest) != document:
        raise ManifestMismatch(
            "production partition manifest digest is not NT's frozen "
            f"{document}; its membership has been altered"
        )


def verify_against_library(manifest, data):
    """Check a verified manifest against the library `read_library` returned.

    `data` has to come from `nt_data.read_library`, which is what makes this
    an identity check rather than a re-derivation: the 34 bp ACGT, TTTN PAM,
    duplicate-sequence, duplicate-protospacer and label-conversion rules are
    applied there, by the same code the NT run used, before anything here runs.

    Returns the per-row records in `source_row` order, keyed by identity.
    """
    train, validation, test = verify_manifest(manifest)
    for key in ("source_file_sha256", "table_sha256"):
        if manifest.get(key) != data[key]:
            raise ManifestMismatch(
                f"{key} differs: manifest {manifest.get(key)!r}, library {data[key]!r}"
            )

    rows = sorted(data["rows"], key=lambda row: row["source_row"])
    recorded = [
        {"source_row": row["source_row"], "sequence_sha256": row["sequence_sha256"]}
        for row in rows
    ]
    if recorded != manifest["rows"]:
        raise ManifestMismatch(
            "library row identities or sequence checksums differ from the manifest"
        )

    by_identity = {row["source_row"]: row for row in rows}
    if digest([by_identity[i] for i in train]) != manifest["training_partition_sha256"]:
        raise ManifestMismatch(
            "training_partition_sha256 does not describe these training rows"
        )
    baseline = float(np.mean([by_identity[i]["target"] for i in train]))
    if baseline != manifest["baseline_train_mean"]["value"]:
        raise ManifestMismatch(
            "frozen training baseline mean differs from these training rows: "
            f"manifest {manifest['baseline_train_mean']['value']!r}, "
            f"recomputed {baseline!r}"
        )

    # Protospacer disjointness across partitions, stated directly. `read_library`
    # already rejects a protospacer that repeats anywhere in the library, which
    # implies this - but the implication is exactly the kind that survives a
    # loader change unnoticed, and it is the property #115 leans on.
    seen = {}
    for name, members in zip(PARTITION_NAMES, (train, validation, test)):
        for identity in members:
            row = by_identity[identity]
            sequence = row["sequence"]
            if len(sequence) != SEQUENCE_LENGTH:
                raise ManifestMismatch(
                    f"source row {identity} is {len(sequence)} bp, not "
                    f"{SEQUENCE_LENGTH}"
                )
            target = row["target"]
            if not np.isfinite(target) or not SCORE_MIN <= target <= SCORE_MAX:
                raise ManifestMismatch(
                    f"source row {identity} target {target!r} is outside "
                    f"[{SCORE_MIN}, {SCORE_MAX}]"
                )
            protospacer = protospacer_of(sequence)
            if protospacer in seen:
                raise ManifestMismatch(
                    f"protospacer of source row {identity} ({name}) also "
                    f"appears in {seen[protospacer]}"
                )
            seen[protospacer] = name
    return by_identity


def load_library(source, manifest):
    """Read the source sheet under the manifest's own split configuration."""
    return read_library(source, split_config_for(manifest))


def partition_arrays(by_identity, identities):
    """`(X_seq, X_feat, y)` for exactly these source rows, in this order.

    Built with `cas12a.features`' canonical `one_hot_encode` and
    `extract_seq_features` - the same two functions `cas12a.data` uses - so the
    CNN sees the same encoding whether its rows came from the loader or from a
    manifest. The arrays carry no identity of their own, which is why
    `fitting_set` returns the identity list beside them and every artifact
    writes `source_row` rather than an array offset.
    """
    x_seq, x_feat, y = [], [], []
    for identity in identities:
        row = by_identity[identity]
        gc_feat, dinuc_feat = extract_seq_features(row["sequence"])
        if float(dinuc_feat.sum()) != CLEAN_DINUCLEOTIDE_SUM:
            raise ManifestMismatch(
                f"source row {identity} encodes {dinuc_feat.sum()} dinucleotide "
                f"pairs, not {CLEAN_DINUCLEOTIDE_SUM}"
            )
        x_seq.append(one_hot_encode(row["sequence"]))
        x_feat.append(np.concatenate([gc_feat, dinuc_feat]))
        y.append(row["target"])

    arrays = (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(x_feat, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )
    expected = (
        (len(identities), SEQUENCE_LENGTH, 4),
        (len(identities), FEATURE_DIM),
        (len(identities),),
    )
    if tuple(array.shape for array in arrays) != expected:
        raise ManifestMismatch(
            f"built arrays {tuple(a.shape for a in arrays)}, expected {expected}"
        )
    return arrays


def fitting_set(source, manifest):
    """Everything a single fit needs, with Test membership proved excluded.

    Returns a dict carrying, per fitting partition, the `source_row` identities
    and the three arrays - plus the held-out `test` identities, unexpanded, so
    that a caller can record what it was not allowed to see without ever
    holding its features or labels.
    """
    data = load_library(source, manifest)
    by_identity = verify_against_library(manifest, data)
    held_out = manifest["partitions"][HELD_OUT_NAME]

    partitions = {}
    for name in FITTING_NAMES:
        identities = manifest["partitions"][name]
        x_seq, x_feat, y = partition_arrays(by_identity, identities)
        partitions[name] = {
            "source_rows": list(identities),
            "X_seq": x_seq,
            "X_feat": x_feat,
            "y": y,
        }

    # A postcondition on what was built, not a restatement of the manifest
    # check: `verify_manifest` has already proved the three memberships
    # disjoint, so this can only fire if the loop above drew identities from
    # somewhere other than the two fitting partitions. That is the regression
    # worth a guard here - #115 calls back into this module for mutated
    # sequences, and a held-out row encoded by mistake would read as a result.
    built = {
        identity
        for partition in partitions.values()
        for identity in partition["source_rows"]
    }
    leaked = built.intersection(held_out)
    expected_rows = sum(len(partitions[name]["source_rows"]) for name in FITTING_NAMES)
    if leaked or len(built) != expected_rows:
        raise ManifestMismatch(
            f"the fitting set built {len(built)} of {expected_rows} rows, "
            f"including {len(leaked)} Test source row(s): {sorted(leaked)[:8]}"
        )
    return {
        "data": data,
        "by_identity": by_identity,
        "partitions": partitions,
        "held_out": {"name": HELD_OUT_NAME, "source_rows": list(held_out)},
    }
