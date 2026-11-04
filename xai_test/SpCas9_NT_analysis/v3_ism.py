"""In-silico saturation mutagenesis for 9-Hydra v3, at base resolution (#132).

ADR 0007 gives ISM the per-base PAM question: NT tokenises DNA as
non-overlapping 6-mers, so IG's last token merges the PAM (24-26) with the
right flank (27-29), while a single-base substitution is indifferent to token
boundaries. This module is therefore the only place in the v3 analysis that may
report a number about one PAM base.

**The layout is `(rows, 30, 4, 9)`, ACGT in fixed order, reference slot 0.**
The fourth slot costs 33% more and buys an axis that means the same thing in
every row. SaCas9's alt-only `(n, 36, 3)` makes "slot 0" a different base at
each position, which makes indexing a PAM base impossible without carrying the
wild-type sequence alongside. The zero in the reference slot is **true, not a
sentinel**: substituting a base for itself changes nothing. `reference_base`
is in the archive so that zero stays distinguishable from "no effect" for a
reader holding the artifact alone.

**All nine heads are stored, three are analysed.** One forward returns all
nine, so the extra six cost nothing and avoid a re-run; the sidecar marks them
*stored but not analysed* so they are not read as validated results.

**NaN is forbidden.** A head with no measurement on a row still has a
prediction, so an ISM delta is defined everywhere and a NaN here can only be a
bug — unlike SaCas9's sweep, which NaN-masked a relative delta wherever the
wild-type score approached zero. This module takes an **absolute** delta on
the fraction scale, so no such guard is needed and none is offered.

**The sweep is resumable.** 3,851 rows x 91 variants is ~350K forward passes
through a 500M-parameter encoder; `xai_test/outOfMemory.md` records SaCas9's
ISM sweep dying of CUDA OOM inside a forward pass, and a Colab disconnect is
at least as likely. Rows are swept in shards, each shard is written to
`checkpoint_dir` as it completes, and a later run reuses the shards it finds.
A shard is reused only if its rows *and* its sequences match, because an
edited split keeps its `row_index` column and row ids alone would let a stale
shard contaminate the sweep silently.

The model is reached only through `forward_fn` (`v3_predict.ForwardFn`), so
everything here is testable with no torch, no checkpoint and no GPU.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from v3_predict import (
    HEADS,
    SEQUENCE_LENGTH,
    ForwardFn,
    checkpoint_digests,
    destandardise,
    library_versions,
    population_digest,
    sha256_of,
    validate_sequences,
    write_array_artifact,
)
from v3_strata import ANALYSED_HEADS

SUBTREE_ROOT = Path(__file__).resolve().parent

#: Fixed order. The base axis means the same thing in every row *because* this
#: never varies with the reference base.
BASES: tuple[str, ...] = ("A", "C", "G", "T")
BASE_INDEX: dict[str, int] = {base: index for index, base in enumerate(BASES)}

POSITIONS = SEQUENCE_LENGTH

#: One wild type plus 30 positions x 3 alternatives.
VARIANTS_PER_ROW = 1 + POSITIONS * (len(BASES) - 1)

#: Matches `ism_sweep.py`'s post-OOM default for SaCas9. This sweep is the
#: heavier of the two in **total work** — ~350K forward passes through a 500M
#: encoder — but peak memory is set by one batch of 32-token sequences, which
#: is what this bounds; the length of the run is what `shard_rows` addresses.
#: Because the two are separate problems, the batch size is an argument
#: recorded in the sidecar rather than a constant tuned in place: if a run does
#: hit OOM, lowering it and re-running costs only the unfinished shards.
DEFAULT_ISM_BATCH_SIZE = 256

#: ~23K forward passes per shard, so a disconnect loses minutes rather than the
#: run, while the sweep still writes only ~16 checkpoint files for 3,851 rows.
DEFAULT_SHARD_ROWS = 256

#: ISM runs on the whole analysis intersection, not per stratum: the strata cut
#: the *reporting* population, and aggregation (#134/#137) applies them to this
#: array by `row_index`.
POPULATION = "analysis_intersection"

DELTA_SCALE = "fraction"

#: A `code_ref` has to address an immutable commit. A branch name moves, which
#: is precisely the drift AC11 exists to stop.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

ProgressFn = Callable[[dict[str, Any]], None]


# --- layout ----------------------------------------------------------------


def substitution_jobs(sequence: str) -> tuple[tuple[int, int], ...]:
    """Every (position, base index) substitution that is not the reference.

    Position-major, base index ascending — the order :func:`variants` emits and
    :func:`scatter` reads back.
    """
    return tuple(
        (position, index)
        for position, reference in enumerate(sequence)
        for index, base in enumerate(BASES)
        if base != reference
    )


def variants(sequence: str) -> list[str]:
    """The wild type followed by its 90 single-base substitutions."""
    validate_sequences([sequence])
    return [sequence] + [
        sequence[:position] + BASES[index] + sequence[position + 1 :]
        for position, index in substitution_jobs(sequence)
    ]


def reference_indices(sequence: str) -> np.ndarray:
    """The wild-type base's slot at each position, as an int8 vector."""
    return np.array([BASE_INDEX[base] for base in sequence], dtype=np.int8)


def delta_block(sequence: str, predictions: np.ndarray) -> np.ndarray:
    """Turn one row's 91 predictions into a `(30, 4, 9)` delta block.

    `predictions[0]` is the wild type and the rest follow
    :func:`substitution_jobs`. The reference slot is left at exactly 0.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (VARIANTS_PER_ROW, len(HEADS)):
        raise ValueError(
            f"expected ({VARIANTS_PER_ROW}, {len(HEADS)}) predictions for one "
            f"row's sweep, got {predictions.shape}"
        )
    delta = np.zeros((POSITIONS, len(BASES), len(HEADS)), dtype=np.float64)
    wild_type = predictions[0]
    for order, (position, index) in enumerate(substitution_jobs(sequence)):
        delta[position, index] = predictions[1 + order] - wild_type
    return delta


def sequence_digest(sequences: Sequence[str]) -> str:
    """SHA-256 over the swept sequences, in order. Pins a reused shard."""
    return hashlib.sha256("\n".join(sequences).encode("utf-8")).hexdigest()


# --- the sweep -------------------------------------------------------------


@dataclass(frozen=True)
class ISMResult:
    """One sweep's arrays, aligned on axis 0 by ascending `row_index`."""

    row_index: np.ndarray
    heads: tuple[str, ...]
    #: `(rows, 30, 4, 9)` float32, fraction scale, reference slot exactly 0.
    delta: np.ndarray
    #: `(rows, 9)` float32 — the wild-type prediction on the same scale as
    #: `predictions.csv`, which is what lets a consumer catch a mis-join
    #: instead of discovering it as a result.
    wt_prediction: np.ndarray
    #: `(rows, 30)` int8 — which slot holds the reference base.
    reference_base: np.ndarray
    batch_size: int
    shard_rows: int
    #: Indices into :func:`shard_bounds` that were read back from disk rather
    #: than computed in this run.
    reused_shards: tuple[int, ...] = ()

    replace = replace


def shard_bounds(rows: int, shard_rows: int) -> list[tuple[int, int]]:
    """Contiguous `[start, stop)` row blocks covering `rows`."""
    if shard_rows < 1:
        raise ValueError(f"shard_rows must be >= 1, got {shard_rows}")
    return [
        (start, min(start + shard_rows, rows)) for start in range(0, rows, shard_rows)
    ]


def predict_in_batches(
    forward_fn: ForwardFn, sequences: Sequence[str], batch_size: int
) -> np.ndarray:
    """Dispatch `forward_fn` in bounded chunks, as `(n, 9)` standardised.

    Non-finite output fails here rather than at the write: a 350K-forward run
    should stop at the batch that went wrong, with its earlier shards already
    on disk, not after the last one.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    chunks = []
    for start in range(0, len(sequences), batch_size):
        chunk = np.asarray(
            forward_fn(list(sequences[start : start + batch_size])), dtype=np.float64
        )
        if chunk.ndim != 2 or chunk.shape[1] != len(HEADS):
            raise ValueError(
                f"forward_fn returned {chunk.shape} for the batch at {start}; "
                f"expected (batch, {len(HEADS)})"
            )
        if not np.isfinite(chunk).all():
            raise ValueError(
                f"forward_fn returned a non-finite value in the batch at "
                f"sequence {start}; NaN is forbidden in an ISM delta"
            )
        chunks.append(chunk)
    return (
        np.concatenate(chunks)
        if chunks
        else np.empty((0, len(HEADS)), dtype=np.float64)
    )


def sweep_block(
    forward_fn: ForwardFn,
    sequences: Sequence[str],
    *,
    mu: np.ndarray,
    sd: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a block of rows, returning `(delta, wt_prediction)`.

    `mu` cancels in the delta — it is a difference of two predictions — but not
    in `wt_prediction`, so both are needed and de-standardisation is applied
    once, to the predictions, rather than twice to two derived quantities.
    """
    flattened = [variant for sequence in sequences for variant in variants(sequence)]
    predictions = destandardise(
        predict_in_batches(forward_fn, flattened, batch_size), mu, sd
    ).reshape(len(sequences), VARIANTS_PER_ROW, len(HEADS))

    delta = np.stack(
        [
            delta_block(sequence, predictions[row])
            for row, sequence in enumerate(sequences)
        ]
    ).astype(np.float32)
    return delta, predictions[:, 0].astype(np.float32)


def _shard_path(checkpoint_dir: Path, start: int) -> Path:
    return checkpoint_dir / f"ism_rows_{start:06d}.npz"


def _write_shard(
    path: Path,
    *,
    delta: np.ndarray,
    wt_prediction: np.ndarray,
    row_index: np.ndarray,
    sequences: Sequence[str],
) -> None:
    """Write one shard, then move it into place.

    The rename is what makes the checkpoint safe to resume onto: a session
    killed mid-write would otherwise leave a truncated `.npz` at the name the
    next run looks for, and the sweep would fail on a corrupt archive instead
    of recomputing the shard.
    """
    staging = path.with_suffix(".npz.partial")
    with open(staging, "wb") as handle:
        np.savez(
            handle,
            delta=delta,
            wt_prediction=wt_prediction,
            row_index=row_index,
            sequence_digest=np.array(sequence_digest(sequences)),
        )
    staging.replace(path)


def _load_shard(
    path: Path, row_index: np.ndarray, sequences: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Read a shard back, refusing one that is not this population's."""
    with np.load(path, allow_pickle=False) as archive:
        stored = archive["row_index"]
        if stored.tolist() != row_index.tolist():
            raise ValueError(
                f"{path.name} holds rows {stored[:3].tolist()}... but this run "
                f"expects {row_index[:3].tolist()}...: the checkpoint directory "
                f"belongs to a different population"
            )
        if str(archive["sequence_digest"]) != sequence_digest(sequences):
            raise ValueError(
                f"{path.name} was computed from different sequences for the "
                f"same rows; delete the checkpoint directory rather than "
                f"resuming onto an edited split"
            )
        return archive["delta"], archive["wt_prediction"]


def run_ism(
    forward_fn: ForwardFn,
    sequences: Sequence[str],
    row_index: Sequence[int],
    *,
    mu: Sequence[float] | np.ndarray,
    sd: Sequence[float] | np.ndarray,
    batch_size: int = DEFAULT_ISM_BATCH_SIZE,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    checkpoint_dir: str | Path | None = None,
    progress: ProgressFn | None = None,
) -> ISMResult:
    """Sweep every position x base over `sequences`, in `row_index` order.

    `row_index` must already be strictly ascending: implicit positional
    identity is how a 3,851-row artifact and a 400-row one silently mis-join,
    so the ordering is a precondition rather than something this fixes up.

    With `checkpoint_dir` set, each shard is written as it completes and a
    later call reuses what it finds — see the module docstring.

    `batch_size` bounds the chunk handed to `forward_fn`, and the sidecar
    records it as *the* batch size of the run. A `forward_fn` that batches
    internally — `v3_predict.model_forward_fn` does — must therefore be built
    with the same number, or the recorded figure is not the one the GPU saw and
    an OOM report cannot be acted on.
    """
    sequences = list(sequences)
    indices = np.asarray(row_index, dtype=np.int64)
    if indices.shape != (len(sequences),):
        raise ValueError(
            f"row_index has {indices.shape} entries for {len(sequences)} sequences"
        )
    if np.any(np.diff(indices) <= 0):
        raise ValueError("row_index must be strictly ascending and unique")
    validate_sequences(sequences)

    mu = np.asarray(mu, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    if mu.shape != (len(HEADS),):
        raise ValueError(f"mu must be ({len(HEADS)},), got {mu.shape}")
    if sd.shape != (len(HEADS),):
        raise ValueError(f"sd must be ({len(HEADS)},), got {sd.shape}")
    if not np.all(sd > 0):
        raise ValueError("sd holds a non-positive standard deviation")

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    deltas, wild_types, reused = [], [], []
    for shard, (start, stop) in enumerate(shard_bounds(len(sequences), shard_rows)):
        block = sequences[start:stop]
        block_rows = indices[start:stop]
        path = None if checkpoint_dir is None else _shard_path(checkpoint_dir, start)

        from_disk = path is not None and path.exists()
        if from_disk:
            delta, wild_type = _load_shard(path, block_rows, block)
            reused.append(shard)
        else:
            delta, wild_type = sweep_block(
                forward_fn, block, mu=mu, sd=sd, batch_size=batch_size
            )
            if path is not None:
                _write_shard(
                    path,
                    delta=delta,
                    wt_prediction=wild_type,
                    row_index=block_rows,
                    sequences=block,
                )

        deltas.append(np.asarray(delta, dtype=np.float32))
        wild_types.append(np.asarray(wild_type, dtype=np.float32))
        if progress is not None:
            progress(
                {
                    "shard": shard,
                    "rows_done": stop,
                    "rows_total": len(sequences),
                    "reused": from_disk,
                }
            )

    return ISMResult(
        row_index=indices,
        heads=tuple(HEADS),
        delta=(
            np.concatenate(deltas)
            if deltas
            else np.empty((0, POSITIONS, len(BASES), len(HEADS)), dtype=np.float32)
        ),
        wt_prediction=(
            np.concatenate(wild_types)
            if wild_types
            else np.empty((0, len(HEADS)), dtype=np.float32)
        ),
        reference_base=np.stack(
            [reference_indices(sequence) for sequence in sequences]
        ).astype(np.int8)
        if sequences
        else np.empty((0, POSITIONS), dtype=np.int8),
        batch_size=batch_size,
        shard_rows=shard_rows,
        reused_shards=tuple(reused),
    )


# --- artifact --------------------------------------------------------------


def assert_reference_slots_are_zero(result: ISMResult) -> None:
    """The reference base's delta must be exactly 0, in every row."""
    taken = np.take_along_axis(
        result.delta, result.reference_base.astype(np.int64)[:, :, None, None], axis=2
    )
    if np.any(taken != 0.0):
        row, position = (int(index[0]) for index in np.nonzero(taken.any(axis=(2, 3))))
        raise ValueError(
            f"the reference base's delta is non-zero at row "
            f"{result.row_index[row]} position {position}: substituting a base "
            f"for itself changes nothing, so this slot is 0 by definition"
        )


def write_ism_artifact(
    out_dir: str | Path,
    result: ISMResult,
    *,
    code_ref: str | None,
    split_csv: str | Path,
    checkpoint_dir: str | Path,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write `ism.npz` and `ism.sidecar.json`; return the archive's path.

    AC11 asks the sidecar to carry what re-derives the run, so everything it
    names is a required argument and three of them are refused rather than
    recorded as absent: `code_ref` must be a 40-character hex commit sha (a
    branch name moves and cannot address a run), and the `transformers` and
    `torch` versions must be present. Version drift between what produced an
    artifact and what reads it back is the exact mechanism that silently
    zeroed Model B's IG in the SaCas9 pipeline. `split_csv` and
    `checkpoint_dir` are required for the same reason: a sidecar that omits
    the consumed split's checksum cannot answer which split the GPU time was
    spent on.
    """
    if not code_ref or not _COMMIT_SHA.fullmatch(code_ref):
        raise ValueError(
            f"CODE_REF must be a 40-character hex commit sha (AC11), got "
            f"{code_ref!r}; a branch name moves and cannot pin a run"
        )

    versions = library_versions()
    unpinned = [name for name in ("torch", "transformers") if not versions.get(name)]
    if unpinned:
        raise ValueError(
            f"the sidecar cannot record {unpinned} (AC11): an ISM sweep runs "
            f"the real checkpoint, so these versions must be present"
        )

    arrays = {
        "delta": result.delta.astype(np.float32),
        "wt_prediction": result.wt_prediction.astype(np.float32),
        "reference_base": result.reference_base.astype(np.int8),
        "row_index": result.row_index.astype(np.int64),
    }
    assert_reference_slots_are_zero(result)

    rows = int(result.row_index.size)
    sidecar: dict[str, Any] = {
        "model": "9-Hydra v3",
        "analysis": "in_silico_saturation_mutagenesis",
        "population": POPULATION,
        "population_digest": population_digest(result.row_index),
        "rows": rows,
        "positions": POSITIONS,
        "bases": list(BASES),
        "heads": list(result.heads),
        "analysed_heads": list(ANALYSED_HEADS),
        "stored_not_analysed": [
            head for head in result.heads if head not in ANALYSED_HEADS
        ],
        "stored_not_analysed_note": (
            "One forward pass returns all nine heads, so all nine are stored "
            "to avoid a re-run. Only `analysed_heads` are analysed and "
            "reported; the rest are stored but not analysed, and no figure, "
            "claim or validation covers them."
        ),
        "delta_definition": (
            "prediction(single-base mutant) - prediction(wild type), per head, "
            "at delta[row, position, base, head]"
        ),
        "delta_scale": DELTA_SCALE,
        "scale_note": (
            "Deltas and `wt_prediction` are fractions of 1, the scale v3 was "
            "trained on and `predictions.csv` reports; the split's label "
            "columns are raw percentages."
        ),
        "reference_slot_note": (
            "The reference base's slot is exactly 0 and that zero is true, not "
            "a sentinel: substituting a base for itself changes nothing. "
            "`reference_base` gives the reference slot per row and position, so "
            "the zero stays distinguishable from a measured no-effect."
        ),
        "base_axis_note": (
            "ACGT in fixed order at every position, so the base axis means the "
            "same thing in every row and a PAM base can be indexed directly "
            "(ADR 0007 assigns base-resolution PAM work to ISM)."
        ),
        "wt_prediction_note": (
            "The wild-type prediction per row and head, on the same scale as "
            "`predictions.csv`. It joins by `row_index` and is the check that "
            "catches a mis-join, instead of a mis-join surfacing as a result."
        ),
        "row_index_note": (
            "int64, aligned to axis 0 and strictly ascending. The only identity "
            "between this artifact and the 400-row IG one."
        ),
        "variants_per_row": VARIANTS_PER_ROW,
        "forwards": rows * VARIANTS_PER_ROW,
        "batch_size": result.batch_size,
        "shard_rows": result.shard_rows,
        "reused_shards": list(result.reused_shards),
        "resume_note": (
            "The sweep runs in row shards, each written to a checkpoint "
            "directory as it completes; `reused_shards` are the shards this run "
            "read back rather than recomputed."
        ),
        "code_ref": code_ref,
        "versions": versions,
        "nan_policy": "forbidden",
    }
    split_csv = Path(split_csv)
    sidecar["split"] = {
        "path": split_csv.as_posix(),
        "sha256": sha256_of(split_csv),
    }
    sidecar["checkpoint"] = {
        "path": Path(checkpoint_dir).as_posix(),
        "files": checkpoint_digests(
            checkpoint_dir, ("config.json", "mu_sd.npy", "vocab.txt")
        ),
    }
    if extra:
        sidecar.update(extra)
    return write_array_artifact(
        out_dir, "ism.npz", arrays, sidecar, digest_key="delta_sha256"
    )
