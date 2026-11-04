"""Attention rollout for 9-Hydra v3, at five-token resolution (#136).

Rolls v3's 24 encoder attention layers into **one** matrix per row: each layer
is averaged over its attention heads, mixed with the residual connection's
implicit identity as ``0.5*A + 0.5*I``, row-normalised, and left-multiplied
onto the running product (Abnar & Zuidema, 2020, "Quantifying Attention Flow
in Transformers"). The operation order is fixed and tested — normalising
before adding the identity gives a numerically different matrix.

**There is no head axis** (ADR 0008). v3's nine regression heads sit on one
shared encoder trunk and the rollout is a function of that trunk's attention
alone, so the quantity has no head to vary over: one row yields one ``(5, 5)``
matrix, not nine. A per-head variant is constructible (gradient-weighted,
grad-CAM style) and deliberately out of scope. The sidecar says so outright,
because nine identical matrices read naturally as a bug or as "the heads attend
identically", and a reviewer expecting a per-head figure will look for one.
Analysis question 2 — whether ``SpCas9`` and ``SpCas9-NG`` use the PAM
differently — is answered by ISM and IG, not here.

**The matrix is not unfolded to base coordinates** (ADR 0007). Attention is a
quantity *between tokens*, and the last of the five 6-mer tokens binds the PAM
(24-26) to the right flank (27-29), so projecting onto 30 positions would
re-mix exactly what that ADR separates. The SaCas9 pipeline unfolded its
rollout to 36x36; that choice is not inherited, and there is no projection
function here to call by accident.

**Only the final rollout is stored.** The 24 per-layer matrices are cheap
(~9 MB) but no analysis reads them, and an artifact nothing interprets is a
liability rather than an option.

What the matrix means is **information dependency** — the connectivity the
encoder constructs between token positions — not a physical DNA/protein
interaction claim.

The rollout arithmetic is pure numpy and tested with no torch, no checkpoint
and no GPU; only :func:`run_rollout` touches the model, and the notebook is a
driver over it (#129 AC12).
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from v3_ig import (
    CLS_TOKEN_COUNT,
    PAM_TOKEN_NAME,
    SEQUENCE_TOKEN_SLICE,
    TOKEN_BOUNDARIES,
    TOKEN_COUNT,
    TOKEN_NAMES,
    encode,
)
from v3_predict import (
    HEADS,
    MAX_LENGTH,
    checkpoint_digests,
    device_of,
    library_versions,
    population_digest,
    sha256_of,
    validate_sequences,
    write_array_artifact,
)

SUBTREE_ROOT = Path(__file__).resolve().parent

# The token contract above is IG's (`v3_ig`), imported rather than restated:
# both analyses index the same five 6-mer tokens of the same 30bp window under
# the same ADR, and two copies of `SEQUENCE_TOKEN_SLICE` are two things to keep
# in step.

#: Abnar & Zuidema's residual mix: half the layer's attention, half the
#: identity the residual connection carries unchanged.
RESIDUAL_WEIGHT = 0.5

LAYER_OPERATION = "0.5*A + 0.5*I, then row-normalise"

#: ADR 0008, in a form a consumer can assert on. One matrix per row.
HEAD_AXIS = False

#: ADR 0007. There is deliberately no `project_rollout_to_nucleotides` here —
#: the refusal is the absence of the function, and this flag is what a sidecar
#: and a test can read.
UNFOLD_TO_BASES = False

#: Rollout runs on the whole analysis intersection, as ISM does: the strata cut
#: the *reporting* population, and aggregation applies them by `row_index`.
POPULATION = "analysis_intersection"

QUANTITY = "information_dependency"
QUANTITY_NOTE = (
    "The connectivity v3's encoder constructs between token positions, not a "
    "physical DNA/protein interaction claim."
)

HEAD_INVARIANCE_NOTE = (
    "There is no head axis (ADR 0008). v3's nine regression heads share one "
    "encoder trunk and the rollout is a function of that trunk's attention "
    "alone, so one row yields one 5x5 matrix rather than nine. `heads` is "
    "recorded only to name the trunk's consumers: no entry of this array "
    "belongs to a head, and no per-head rollout figure is derivable from it. "
    "The per-head PAM question is answered by ISM (base resolution) and IG "
    "(token resolution)."
)

TOKEN_RESOLUTION_NOTE = (
    "Stored at five-token resolution and not unfolded to the 30 base "
    "coordinates (ADR 0007): attention is a quantity between tokens, and the "
    f"last token is {PAM_TOKEN_NAME!r} — it binds the PAM (24-26) to the right "
    "flank (27-29), so an unfold would re-mix what that ADR separates. The "
    "SaCas9 pipeline's 36x36 unfold is not inherited."
)

CLS_REMOVAL_NOTE = (
    "`<cls>` participates in the rollout and its row and column are dropped "
    "afterwards, so each stored row sums to 1 minus the share that reached "
    "`<cls>` (`cls_mass` records that share's distribution). The block is "
    "**not** renormalised: rescaling it would state a dependency between "
    "sequence tokens that the encoder routed through `<cls>`."
)

#: One forward pass per row. 3,851 of them, which is why this is its own ticket
#: rather than a rider on the ~350K-forward ISM sweep: a plain forward batch
#: with `output_attentions=True` costs the attention tensors' memory and
#: nothing else.
DEFAULT_ROLLOUT_BATCH_SIZE = 64

#: A `code_ref` has to address an immutable commit. A branch name moves, which
#: is precisely the drift AC11 exists to stop.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

ProgressFn = Callable[[dict[str, int]], None]


# --- rollout arithmetic ----------------------------------------------------


def layer_rollout(head_averaged: np.ndarray) -> np.ndarray:
    """One layer's `0.5*A + 0.5*I`, row-normalised. `(batch, n, n)`.

    The order is fixed: the identity is added **first** and the result
    normalised **second**. Normalising first and adding the identity after
    gives a different matrix — on an already row-stochastic `A` the two happen
    to coincide, which is why the test pins the order on an unnormalised one.
    """
    attention = np.asarray(head_averaged, dtype=np.float64)
    if attention.ndim != 3 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError(
            f"expected a square (batch, n, n) attention matrix, got {attention.shape}"
        )
    mixed = RESIDUAL_WEIGHT * attention + RESIDUAL_WEIGHT * np.eye(attention.shape[-1])
    return mixed / mixed.sum(axis=-1, keepdims=True)


def rollout_matrix(attentions: Sequence[np.ndarray]) -> np.ndarray:
    """Roll every layer's attention into one `(batch, n, n)` matrix.

    `attentions` is one `(batch, attention_heads, n, n)` array per encoder
    layer, **ordered first layer to last**, with `<cls>` and the padding still
    present — they take part in the flow the rollout mixes through the encoder,
    so they are dropped afterwards (:func:`sequence_block`), not before.

    Each layer is averaged over its attention heads, passed through
    :func:`layer_rollout`, and left-multiplied onto the running product, so the
    result reads as flow from the input tokens through to the final layer.
    """
    rolled: np.ndarray | None = None
    for layer, attention in enumerate(attentions):
        attention = np.asarray(attention, dtype=np.float64)
        if attention.ndim != 4:
            raise ValueError(
                f"layer {layer}: expected (batch, attention_heads, n, n), got "
                f"{attention.shape}"
            )
        this_layer = layer_rollout(attention.mean(axis=1))
        rolled = this_layer if rolled is None else this_layer @ rolled
    if rolled is None:
        raise ValueError("a rollout needs at least one layer of attention")
    return rolled


def sequence_block(rollout: np.ndarray) -> np.ndarray:
    """The five sequence tokens' block of a full rollout. `(batch, 5, 5)`.

    Drops `<cls>` (index 0) and the padding (everything after the five tokens)
    *after* the cross-layer rollout. No renormalisation — see
    :data:`CLS_REMOVAL_NOTE`.
    """
    rollout = np.asarray(rollout)
    if rollout.ndim != 3:
        raise ValueError(f"expected a (batch, n, n) rollout, got {rollout.shape}")
    return rollout[:, SEQUENCE_TOKEN_SLICE, SEQUENCE_TOKEN_SLICE]


def cls_mass(rollout: np.ndarray) -> np.ndarray:
    """Each sequence token's share of dependency on `<cls>`. `(batch, 5)`.

    This is what :func:`sequence_block` discards, and it is the reason a stored
    row sums to less than 1. Summarised in the sidecar so a reader meeting a
    non-stochastic matrix finds the missing mass accounted for.
    """
    return np.asarray(rollout)[:, SEQUENCE_TOKEN_SLICE, 0]


def assert_padding_is_unattended(
    rollout: np.ndarray, *, tolerance: float = 1e-6
) -> None:
    """No sequence token may route dependency onto a padding position.

    With the attention mask in force the padded keys get zero probability, so
    the five sequence tokens' rows distribute over `<cls>` and the five tokens
    and nothing else. If the mask ever stopped reaching the encoder this is the
    only symptom: the rows would spread over 26 padding positions and the
    stored 5x5 block would silently lose that mass instead of failing.
    """
    rollout = np.asarray(rollout, dtype=np.float64)
    padding = np.abs(rollout[:, SEQUENCE_TOKEN_SLICE, CLS_TOKEN_COUNT + TOKEN_COUNT :])
    if padding.size and padding.max() > tolerance:
        # The worst offender, so the reported magnitude belongs to the
        # coordinates reported beside it.
        row, token, column = (
            int(index) for index in np.unravel_index(padding.argmax(), padding.shape)
        )
        raise ValueError(
            f"row {row} token {token} puts {padding[row, token, column]:.3g} of "
            f"its dependency on padding position "
            f"{CLS_TOKEN_COUNT + TOKEN_COUNT + column} (the worst of "
            f"{int((padding > tolerance).sum())} such positions): the attention "
            f"mask is not reaching the encoder, so the 5x5 block would silently "
            f"drop that mass"
        )


# --- the run ---------------------------------------------------------------


EAGER_ATTENTION = "eager"

#: Recorded when the model cannot say which implementation it is on, rather
#: than claiming `eager`: a sidecar that names an implementation the run did
#: not verify is worse than one that admits it does not know.
UNKNOWN_ATTENTION = "unknown"

EAGER_ATTENTION_NOTE = (
    "`attention_implementation` is the implementation that was actually in "
    "force while the probabilities were read, not the one the run asked for. "
    "The run switches the model to `eager` wherever it can be switched, "
    "because the fused kernels (`sdpa`, flash attention) never materialise "
    "the attention probability matrix: under transformers 5.x "
    "`output_attentions=True` on an sdpa model returns an empty tuple instead "
    "of raising, so a rollout run would have nothing to roll. On a model that "
    "cannot report its implementation this field reads `unknown` — older "
    "transformers fall back to eager by themselves, so the probabilities are "
    "real, but the run has not verified which path produced them."
)


@contextlib.contextmanager
def eager_attention(model: Any):
    """Put `model` on the eager attention path, then restore it.

    Yields the implementation actually in force inside the block, which is
    what the sidecar records: `eager` when this switched the model, and
    :data:`UNKNOWN_ATTENTION` when the model offers no way to switch or to
    report. The second case is transformers < 5, which warns and falls back to
    eager by itself — the numbers are real, but this has not verified the path,
    and a sidecar must not claim a pin it did not set.

    The original implementation is put back on the way out, including on an
    exception, so a notebook that runs the prediction path after this cell
    gets the fast kernel again.
    """
    setter = getattr(model, "set_attn_implementation", None)
    original = getattr(model.config, "_attn_implementation", None)
    if setter is None or original is None:
        yield UNKNOWN_ATTENTION
        return
    setter(EAGER_ATTENTION)
    try:
        yield EAGER_ATTENTION
    finally:
        setter(original)


@dataclass(frozen=True)
class RolloutResult:
    """One run's rollout, aligned on axis 0 by ascending `row_index`.

    `rollout` is `(rows, 5, 5)` and **3-dimensional by contract**: a head axis
    here would be nine copies of one matrix (ADR 0008).
    """

    row_index: np.ndarray
    #: `(rows, 5, 5)` float32, token-to-token, `<cls>` and padding removed.
    rollout: np.ndarray
    #: `(rows, 5)` float64 — the share each token sent to `<cls>`, which is
    #: what the block discards. Summarised in the sidecar, not archived: it is
    #: derivable from the row sums, and the schema is two keys.
    cls_mass: np.ndarray
    attention_layers: int
    attention_heads: int
    batch_size: int
    #: The attention implementation actually in force when the probabilities
    #: were read — `eager` wherever the model could be switched, otherwise
    #: :data:`UNKNOWN_ATTENTION`. See :data:`EAGER_ATTENTION_NOTE`.
    attention_implementation: str = EAGER_ATTENTION

    replace = replace


def run_rollout(
    model: Any,
    tokenizer: Any,
    sequences: Sequence[str],
    row_index: Sequence[int],
    *,
    batch_size: int = DEFAULT_ROLLOUT_BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    progress: ProgressFn | None = None,
) -> RolloutResult:
    """Attention rollout over `sequences`, in ascending `row_index` order.

    `row_index` must already be ascending: implicit positional identity is how
    a 3,851-row artifact and a 400-row one silently mis-join, so the ordering
    is a precondition rather than something this fixes up.

    One forward pass per row with `output_attentions=True`, batched. The
    attention tensors are reduced to one `(batch, 5, 5)` block per batch and
    the rest is released, so peak memory is set by one batch rather than by the
    run: keeping 3,851 x 24 x 20 x 32 x 32 probabilities would be ~75 GB.
    """
    import torch

    sequences = list(sequences)
    indices = np.asarray(row_index, dtype=np.int64)
    if indices.shape != (len(sequences),):
        raise ValueError(
            f"row_index has {indices.shape} entries for {len(sequences)} sequences"
        )
    if np.any(np.diff(indices) <= 0):
        raise ValueError("row_index must be strictly ascending and unique")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    validate_sequences(sequences)

    blocks, discarded = [], []
    layers = heads_averaged = 0
    device = device_of(model)
    with eager_attention(model) as implementation:
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = encode(tokenizer, batch, max_length=max_length, device=device)
            with torch.no_grad():
                attentions = model(**encoded, output_attentions=True).attentions
            if not attentions:
                raise ValueError(
                    "the model returned no attentions: a fused attention "
                    "kernel never forms the probability matrix, so there is "
                    "nothing to roll. Load the checkpoint with "
                    f"attn_implementation={EAGER_ATTENTION!r}"
                )
            layers = len(attentions)
            heads_averaged = int(attentions[0].shape[1])

            full = rollout_matrix([layer.float().cpu().numpy() for layer in attentions])
            assert_padding_is_unattended(full)
            blocks.append(sequence_block(full))
            discarded.append(cls_mass(full))
            if progress is not None:
                progress(
                    {"rows_done": start + len(batch), "rows_total": len(sequences)}
                )

    rollout = (
        np.concatenate(blocks)
        if blocks
        else np.empty((0, TOKEN_COUNT, TOKEN_COUNT), dtype=np.float64)
    )
    return RolloutResult(
        row_index=indices,
        rollout=rollout.astype(np.float32),
        cls_mass=(
            np.concatenate(discarded)
            if discarded
            else np.empty((0, TOKEN_COUNT), dtype=np.float64)
        ),
        attention_layers=layers,
        attention_heads=heads_averaged,
        batch_size=batch_size,
        attention_implementation=implementation,
    )


# --- artifact --------------------------------------------------------------


def assert_no_head_axis(rollout: np.ndarray) -> None:
    """Refuse an array shaped as if the rollout varied by regression head.

    ADR 0008's failure in artifact form is `(rows, 9, 5, 5)`: nine copies of
    one matrix, which a later reader has no way to tell from nine measurements
    that happen to agree.
    """
    rollout = np.asarray(rollout)
    if rollout.ndim != 3:
        raise ValueError(
            f"the rollout is {rollout.shape}, which has a head axis: the "
            f"quantity is head-invariant (ADR 0008), so it is (rows, "
            f"{TOKEN_COUNT}, {TOKEN_COUNT}) with no head axis"
        )
    if rollout.shape[1:] != (TOKEN_COUNT, TOKEN_COUNT):
        raise ValueError(
            f"the rollout is {rollout.shape}, not (rows, {TOKEN_COUNT}, "
            f"{TOKEN_COUNT}): it is stored at five-token resolution and is "
            f"never unfolded to base coordinates (ADR 0007)"
        )


def write_rollout_artifact(
    out_dir: str | Path,
    result: RolloutResult,
    *,
    code_ref: str | None,
    split_csv: str | Path,
    checkpoint_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `rollout.npz` and `rollout.sidecar.json`; return the archive path.

    The archive holds exactly two keys, `rollout` and `row_index`: only the
    final rollout, no per-layer matrices. The head axis and the
    base-coordinate unfold are **refused** here (:func:`assert_no_head_axis`)
    rather than documented against; that only the final rollout is stored is a
    property of what this writes, since there is nowhere to put a per-layer
    array.

    `code_ref`, `split_csv` and `checkpoint_dir` are refused rather than
    recorded as absent, as in :func:`v3_ism.write_ism_artifact` (AC11): a
    40-character commit sha because a branch name moves and cannot address a
    run, and the split and checkpoint so the sidecar can say which bytes the
    GPU time was spent on. `transformers` and `torch` must be importable for
    the same reason. This is a **different run** from the ISM sweep, so its
    versions may legitimately differ from that artifact's — which is why the
    sidecars are per analysis rather than shared.
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
            f"the sidecar cannot record {unpinned} (AC11): a rollout run reads "
            f"the real checkpoint's attention, so these versions must be present"
        )

    if result.attention_layers < 1:
        raise ValueError(
            "the result rolled no attention layers, so there is no rollout to "
            "write: a sidecar recording `layers_combined: 0` would describe a "
            "quantity that was never computed"
        )

    assert_no_head_axis(result.rollout)
    arrays = {
        "rollout": np.asarray(result.rollout, dtype=np.float32),
        "row_index": np.asarray(result.row_index, dtype=np.int64),
    }

    discarded = np.asarray(result.cls_mass, dtype=np.float64)
    rows = int(arrays["row_index"].size)
    sidecar: dict[str, Any] = {
        "model": "9-Hydra v3",
        "analysis": "attention_rollout",
        "adr": ["0007", "0008"],
        "population": POPULATION,
        "population_digest": population_digest(arrays["row_index"]),
        "rows": rows,
        "forwards": rows,
        "quantity": QUANTITY,
        "quantity_note": QUANTITY_NOTE,
        "head_axis": HEAD_AXIS,
        "heads": list(HEADS),
        "head_invariance_note": HEAD_INVARIANCE_NOTE,
        "token_resolution": TOKEN_COUNT,
        "token_names": list(TOKEN_NAMES),
        "token_boundaries": [list(bounds) for bounds in TOKEN_BOUNDARIES],
        "unfolded_to_bases": UNFOLD_TO_BASES,
        "token_resolution_note": TOKEN_RESOLUTION_NOTE,
        "layer_operation": LAYER_OPERATION,
        "layer_operation_note": (
            "Per layer, averaged over the encoder's attention heads: the "
            "identity is added first (the residual connection's implicit "
            "contribution) and the row normalised second. The reverse order "
            "gives a different matrix. Layers are combined by left-"
            "multiplying each later layer onto the running product."
        ),
        "layers_combined": result.attention_layers,
        "attention_heads_averaged": result.attention_heads,
        "attention_heads_note": (
            "Encoder attention heads, averaged per layer (Abnar & Zuidema, "
            "2020). Unrelated to v3's nine regression heads, which this "
            "quantity does not vary over."
        ),
        "attention_implementation": result.attention_implementation,
        "attention_implementation_note": EAGER_ATTENTION_NOTE,
        "stored": "final_rollout_only",
        "per_layer_stored": False,
        "per_layer_note": (
            "The 24 per-layer matrices are ~9 MB and are deliberately not "
            "stored: no analysis reads them."
        ),
        "renormalised_after_cls_removal": False,
        "cls_removal_note": CLS_REMOVAL_NOTE,
        "cls_mass": {
            "min": float(discarded.min()) if discarded.size else 0.0,
            "median": float(np.median(discarded)) if discarded.size else 0.0,
            "max": float(discarded.max()) if discarded.size else 0.0,
        },
        "batch_size": result.batch_size,
        "row_index_note": (
            "int64, aligned to axis 0 and strictly ascending — the same "
            "ordering contract as `ism.npz` and `ig.npz`, so a join is one "
            "operation rather than a per-artifact convention."
        ),
        "code_ref": code_ref,
        "split": {
            "path": Path(split_csv).as_posix(),
            "sha256": sha256_of(split_csv),
        },
        "versions": versions,
        "nan_policy": "forbidden",
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
        out_dir, "rollout.npz", arrays, sidecar, digest_key="rollout_sha256"
    )
