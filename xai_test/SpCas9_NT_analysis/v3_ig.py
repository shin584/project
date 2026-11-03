"""Integrated Gradients for 9-Hydra v3, at five-token resolution (#130).

Two contracts live here, and they constrain each other.

**The tokenise contract (ADR 0007).** NT tokenises DNA as non-overlapping
6-mers, so the 30bp SpCas9 window is exactly five tokens with boundaries
``0-5 | 6-11 | 12-17 | 18-23 | 24-29``. The PAM (24-26) and the right flank
(27-29) therefore share **one** token. IG yields one number per token, and
dividing it across the token's six bases cannot recover which half the signal
came from, so this module reports at token resolution only, names the last
token ``pam_plus_right_flank``, and refuses to write an artifact key called
``pam``. The base-level PAM question belongs to ISM, whose single-base
substitutions are indifferent to token boundaries.

**The attribution point (ADR 0006).** Gradients are taken on the **encoder
input embedding** with a **zero-embedding baseline**, replaying the whole
encoder and classification head. Attaching IG to any post-encoder tensor gives
every sequence position a gradient of exactly zero, because
``EsmClassificationHead`` pools ``<cls>`` alone — the SaCas9 pipeline shipped
that defect and its Model B attribution came back identically zero. The
attachment is done by **replacing the word-embedding module's output through a
forward hook**, not by passing ``inputs_embeds``: ESM's ``token_dropout``
rescale is guarded on ``input_ids is not None``, so the ``inputs_embeds`` path
would quietly explain a function the deployed model never computes.

Two completeness tolerances, for two different jobs:

- :func:`assert_completeness` is **strict** (1e-2 relative) and is what the
  local unit tests assert against the small in-test model. Loosened, it stops
  catching the all-zero-attribution bug ADR 0006 exists to prevent.
- :func:`completeness_report` is the **real run's** policy:
  ``|residual| <= max(0.01, 0.05 * |f(x) - f(0)|)``, returned as a pass rate.
  It never raises, because a 400-row run must not abort on the rows where
  ``f(x) - f(0)`` sits near zero and a relative bound means nothing — which is
  also why the absolute floor is there.

The hidden axis is summed away: ``(400, 3, 5, 1280)`` is 7.7M values with no
interpretation path. ``torch`` and ``transformers`` are imported lazily, so
this module imports, and its token-contract and tolerance tests run, without
them and without a checkpoint.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import v3_predict
from v3_predict import HEADS, SEQUENCE_LENGTH, validate_sequences
from v3_strata import ANALYSED_HEADS

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

SUBTREE_ROOT = Path(__file__).resolve().parent

#: Non-overlapping 6-mer tokenisation. 30 = 5 x 6 exactly.
TOKEN_SPAN = 6
TOKEN_COUNT = SEQUENCE_LENGTH // TOKEN_SPAN

#: Inclusive 0-indexed base boundaries of each token in the 30bp window.
TOKEN_BOUNDARIES: tuple[tuple[int, int], ...] = tuple(
    (index * TOKEN_SPAN, index * TOKEN_SPAN + TOKEN_SPAN - 1)
    for index in range(TOKEN_COUNT)
)

#: Token names, coordinate-explicit on purpose. The window's regions are left
#: flank 0-3, protospacer 4-23, PAM 24-26, right flank 27-29 — none of which
#: lines up with a token edge except at 24, so a name like `protospacer` would
#: be a claim the boundaries do not support.
TOKEN_NAMES: tuple[str, ...] = (
    "left_flank_plus_protospacer_5prime",
    "protospacer_6_11",
    "protospacer_12_17",
    "protospacer_18_23",
    "pam_plus_right_flank",
)

PAM_TOKEN_INDEX = TOKEN_COUNT - 1
PAM_TOKEN_NAME = TOKEN_NAMES[PAM_TOKEN_INDEX]

#: `<cls>` occupies index 0 and the five sequence tokens follow; everything
#: after them is padding. Verified against the checkpoint's own tokenizer.
CLS_TOKEN_COUNT = 1
SEQUENCE_TOKEN_SLICE = slice(CLS_TOKEN_COUNT, CLS_TOKEN_COUNT + TOKEN_COUNT)

MAX_LENGTH = 32

#: Matches the SaCas9 pipeline's `DEFAULT_IG_STEPS`, and the design's budget
#: of 400 x 50 x 3 forward+backward passes.
DEFAULT_IG_STEPS = 50

#: Far below a plain forward pass's batch size: IG holds an autograd graph per
#: step and runs one backward per analysed head on top of it.
DEFAULT_IG_BATCH_SIZE = 8

#: Strict, for the unit tests against the toy model.
UNIT_RELATIVE_TOLERANCE = 1e-2

#: Reported, for the real run.
RUN_ABSOLUTE_FLOOR = 0.01
RUN_RELATIVE_TOLERANCE = 0.05

ATTRIBUTION_POINT = "encoder_input_embedding"
BASELINE = "zero_embedding"

#: An artifact key naming the PAM at base resolution is the ADR 0007 mistake
#: in artifact form, so it is rejected rather than documented against.
_ALLOWED_PAM_KEYS = frozenset({PAM_TOKEN_NAME})


# --- token contract --------------------------------------------------------


def token_slice(index: int) -> slice:
    """The base positions one token covers, as a slice into the 30bp window."""
    if not 0 <= index < TOKEN_COUNT:
        raise ValueError(f"token index {index} outside 0..{TOKEN_COUNT - 1}")
    start, end = TOKEN_BOUNDARIES[index]
    return slice(start, end + 1)


def tokenize_window(sequence: str) -> tuple[str, ...]:
    """Split a 30bp window into its five 6-mers.

    Mirrors what the checkpoint's own tokenizer does, so the boundaries can be
    asserted without loading it. :func:`encode` uses the real tokenizer.
    """
    validate_sequences([sequence])
    return tuple(sequence[token_slice(index)] for index in range(TOKEN_COUNT))


def token_bases(sequence: str, index: int) -> str:
    """The bases of one token, for reporting a token's content alongside it."""
    return sequence[token_slice(index)]


def validate_artifact_keys(keys: Iterable[str]) -> None:
    """Refuse any key that names the PAM at base resolution (ADR 0007)."""
    offending = sorted(
        key for key in keys if key not in _ALLOWED_PAM_KEYS and "pam" in key.lower()
    )
    if offending:
        raise ValueError(
            f"ADR 0007: IG reports at token resolution, so {offending} may not "
            f"name the PAM at base level; the token is {PAM_TOKEN_NAME!r}"
        )


# --- result ----------------------------------------------------------------


@dataclass(frozen=True)
class IGResult:
    """One IG run's arrays, aligned on axis 0 by ascending ``row_index``.

    ``attribution`` is ``(rows, heads, 5)`` with the hidden axis summed away.
    ``special_attribution`` holds the share that landed on ``<cls>`` and the
    padding; it is not part of the reported attribution but it is what makes
    ``completeness_residual`` an integration-error measure rather than a
    silent accounting gap.
    """

    row_index: np.ndarray
    heads: tuple[str, ...]
    token_names: tuple[str, ...]
    attribution: np.ndarray
    completeness_residual: np.ndarray
    special_attribution: np.ndarray
    f_x: np.ndarray
    f_baseline: np.ndarray
    steps: int
    #: ``f(0)`` for all **nine** heads, not just the three analysed ones. One
    #: forward returns all nine, and the sidecar records them so the zero-
    #: embedding baseline can be re-derived for a head IG did not attribute.
    f_baseline_all_heads: dict[str, float] = field(default_factory=dict)

    replace = replace

    @property
    def delta(self) -> np.ndarray:
        """``f(x) - f(0)``, the quantity completeness is measured against."""
        return self.f_x - self.f_baseline


def assert_completeness(
    result: IGResult, *, relative_tolerance: float = UNIT_RELATIVE_TOLERANCE
) -> None:
    """Strict completeness, for unit tests against the in-test toy model."""
    delta = np.abs(result.delta)
    residual = np.abs(result.completeness_residual)
    allowed = relative_tolerance * delta
    bad = residual > allowed
    if bad.any():
        row, head = (int(index[0]) for index in np.nonzero(bad))
        raise AssertionError(
            f"IG completeness fails at row {result.row_index[row]} head "
            f"{result.heads[head]}: |residual| {residual[row, head]:.6g} > "
            f"{relative_tolerance:g} x |f(x) - f(0)| {delta[row, head]:.6g}"
        )


def assert_nonzero_attribution(result: IGResult) -> None:
    """No head may come back with an identically zero profile (ADR 0006)."""
    per_head = np.abs(result.attribution).sum(axis=2)
    bad = per_head == 0
    if bad.any():
        row, head = (int(index[0]) for index in np.nonzero(bad))
        raise AssertionError(
            f"attribution is identically zero at row {result.row_index[row]} "
            f"head {result.heads[head]}: this is the ADR 0006 failure, not a "
            f"finding about the model"
        )


def completeness_report(
    completeness_residual: np.ndarray,
    f_x: np.ndarray,
    f_baseline: np.ndarray,
    *,
    absolute_floor: float = RUN_ABSOLUTE_FLOOR,
    relative_tolerance: float = RUN_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """The real run's tolerance, as a pass rate. Never raises.

    ``|residual| <= max(absolute_floor, relative_tolerance * |f(x) - f(0)|)``.
    The floor matters because rows where ``f(x) - f(0)`` is near zero make a
    relative bound meaningless, and a 400-row run must not stop on them.
    """
    residual = np.abs(np.asarray(completeness_residual, dtype=np.float64))
    delta = np.abs(np.asarray(f_x, dtype=np.float64) - np.asarray(f_baseline))
    allowed = np.maximum(absolute_floor, relative_tolerance * delta)
    passing = residual <= allowed
    return {
        "tolerance": {
            "absolute_floor": absolute_floor,
            "relative": relative_tolerance,
        },
        "rule": "|residual| <= max(absolute_floor, relative * |f(x) - f(0)|)",
        "rows": int(residual.shape[0]),
        "checks": int(passing.size),
        "passing": int(passing.sum()),
        "failing": int((~passing).sum()),
        "pass_rate": float(passing.mean()) if passing.size else 1.0,
        "max_abs_residual": float(residual.max()) if residual.size else 0.0,
        "median_abs_residual": float(np.median(residual)) if residual.size else 0.0,
        "per_head": {
            "pass_rate": [float(column.mean()) for column in passing.T],
            "max_abs_residual": [float(column.max()) for column in residual.T],
        },
        "enforced": False,
        "enforcement_note": (
            "Reported, not enforced: a failing row is recorded and the run "
            "continues. The strict 1e-2 relative check lives in the unit "
            "tests against the toy model."
        ),
    }


# --- the forward pass ------------------------------------------------------


def head_indices(heads: Sequence[str]) -> list[int]:
    """Positions of ``heads`` in v3's nine output columns."""
    unknown = [head for head in heads if head not in HEADS]
    if unknown:
        raise ValueError(f"unknown heads {unknown}; v3's heads are {list(HEADS)}")
    return [HEADS.index(head) for head in heads]


def encode(tokenizer: Any, sequences: Sequence[str], *, max_length: int = MAX_LENGTH):
    """Tokenise a batch exactly as the prediction path does, and check it.

    The assertion is what lets every later step index the five sequence tokens
    positionally: a tokenizer that put ``<cls>`` elsewhere, or added an ``eos``,
    would otherwise shift the whole profile by one token.
    """
    validate_sequences(sequences)
    encoded = tokenizer(
        list(sequences),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    expected = list(tokenize_window(sequences[0]))
    if tokens[SEQUENCE_TOKEN_SLICE] != expected:
        raise ValueError(
            f"the tokenizer does not place the five sequence tokens at "
            f"{SEQUENCE_TOKEN_SLICE}: got {tokens[: TOKEN_COUNT + 1]}, "
            f"expected ['<cls>', *{expected}]"
        )
    return encoded


def _frozen_parameters(module: Any):
    """Disable ``requires_grad`` on a module's weights for the duration.

    IG needs a gradient only with respect to the interpolated input. Leaving
    v3's 500M parameters requiring grad makes autograd keep a same-sized graph
    for weights nothing reads — the SaCas9 pipeline exhausted host RAM that way
    once IG started replaying the full encoder.
    """
    import contextlib

    @contextlib.contextmanager
    def context():
        original = [p.requires_grad for p in module.parameters()]
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, required in zip(module.parameters(), original):
                parameter.requires_grad_(required)

    return context()


def _logits(model: Any, encoded: Any, indices: Sequence[int]) -> torch.Tensor:
    return model(**encoded).logits[:, list(indices)]


def _forward_with_embedding(
    model: Any, encoded: Any, indices: Sequence[int], embedding: torch.Tensor
) -> torch.Tensor:
    """Run the model with the word-embedding output replaced by ``embedding``.

    A forward hook rather than ``inputs_embeds``: ``input_ids`` stays in the
    call, so ESM's ``token_dropout`` rescale and its position-id derivation run
    exactly as they do in the deployed forward pass.
    """
    handle = model.get_input_embeddings().register_forward_hook(
        lambda module, inputs, output: embedding
    )
    try:
        return _logits(model, encoded, indices)
    finally:
        handle.remove()


def _input_embedding(model: Any, encoded: Any) -> torch.Tensor:
    """The word-embedding output for a batch — IG's ``x``."""
    import torch

    captured: dict[str, torch.Tensor] = {}
    handle = model.get_input_embeddings().register_forward_hook(
        lambda module, inputs, output: captured.setdefault("x", output.detach())
    )
    try:
        with torch.no_grad():
            model(**encoded)
    finally:
        handle.remove()
    return captured["x"]


def baseline_outputs(model: Any, encoded: Any) -> dict[str, float]:
    """``f(0)`` at the zero embedding, for all nine heads.

    One forward returns all nine, so the six unanalysed heads are free. They go
    in the sidecar because ``f(0)`` is half of the completeness arithmetic and
    cannot be recovered once the Colab session ends.
    """
    import torch

    with torch.no_grad():
        logits = _forward_with_embedding(
            model,
            encoded,
            range(len(HEADS)),
            torch.zeros_like(_input_embedding(model, encoded)),
        )
    return {head: float(logits[0, index]) for index, head in enumerate(HEADS)}


def integrated_gradients_batch(
    model: Any,
    encoded: Any,
    indices: Sequence[int],
    *,
    steps: int = DEFAULT_IG_STEPS,
) -> dict[str, np.ndarray]:
    """IG for one batch, summed over the hidden axis.

    Returns ``attribution`` ``(batch, heads, positions)`` over *all* token
    positions, plus ``f_x`` and ``f_baseline``. The midpoint Riemann rule is
    used rather than the right-endpoint sum: it costs the same 50 steps and
    leaves a markedly smaller completeness residual, which matters because the
    residual is the audit signal, not a diagnostic to be explained away.
    """
    import torch

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    x = _input_embedding(model, encoded)
    baseline = torch.zeros_like(x)
    diff = x - baseline

    accumulated = torch.zeros((len(indices), *x.shape), dtype=torch.float64)
    with _frozen_parameters(model):
        for step in range(steps):
            alpha = (step + 0.5) / steps
            interpolated = (
                (baseline + alpha * diff).detach().clone().requires_grad_(True)
            )
            logits = _forward_with_embedding(model, encoded, indices, interpolated)
            for position, _ in enumerate(indices):
                gradient = torch.autograd.grad(
                    logits[:, position].sum(),
                    interpolated,
                    retain_graph=position < len(indices) - 1,
                )[0]
                accumulated[position] += gradient.double()

        with torch.no_grad():
            f_x = _logits(model, encoded, indices).double()
            f_baseline = _forward_with_embedding(
                model, encoded, indices, baseline
            ).double()

    average_gradient = accumulated / steps
    attribution = (average_gradient * diff.double().unsqueeze(0)).sum(dim=-1)
    return {
        # (batch, heads, positions)
        "attribution": attribution.permute(1, 0, 2).numpy(),
        "f_x": f_x.numpy(),
        "f_baseline": f_baseline.numpy(),
    }


def run_ig(
    model: Any,
    tokenizer: Any,
    sequences: Sequence[str],
    row_index: Sequence[int],
    *,
    heads: Sequence[str] = ANALYSED_HEADS,
    steps: int = DEFAULT_IG_STEPS,
    batch_size: int = DEFAULT_IG_BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> IGResult:
    """Token-level IG over ``sequences``, in ascending ``row_index`` order.

    ``row_index`` must already be ascending. Implicit positional identity is
    how a 400-row IG artifact silently mis-joins against a 3,851-row ISM one,
    so the ordering is a precondition rather than something this fixes up.
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
    columns = head_indices(heads)

    attribution_batches, f_x_batches, baselines = [], [], []
    all_head_baseline: dict[str, float] = {}
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoded = encode(tokenizer, batch, max_length=max_length)
        if not all_head_baseline:
            all_head_baseline = baseline_outputs(model, encoded)
        computed = integrated_gradients_batch(model, encoded, columns, steps=steps)
        attribution_batches.append(computed["attribution"])
        f_x_batches.append(computed["f_x"])
        baselines.append(computed["f_baseline"])

    full = np.concatenate(attribution_batches, axis=0)
    f_x = np.concatenate(f_x_batches, axis=0)
    # Every sequence is 30bp and every batch pads to the same length, so the
    # zero-embedding baseline output is one vector, not one per row.
    f_baseline = np.concatenate(baselines, axis=0)
    if not np.allclose(f_baseline, f_baseline[0], rtol=1e-6, atol=1e-9):
        raise ValueError(
            "the zero-embedding baseline differs across rows, which it cannot "
            "do at fixed sequence length and fixed padding"
        )
    f_baseline = f_baseline[0]

    sequence_attribution = full[:, :, SEQUENCE_TOKEN_SLICE]
    residual = (f_x - f_baseline) - full.sum(axis=2)
    special = full.sum(axis=2) - sequence_attribution.sum(axis=2)

    result = IGResult(
        row_index=indices,
        heads=tuple(heads),
        token_names=TOKEN_NAMES,
        attribution=sequence_attribution.astype(np.float32),
        completeness_residual=residual.astype(np.float64),
        special_attribution=special.astype(np.float64),
        f_x=f_x,
        f_baseline=f_baseline,
        steps=steps,
        f_baseline_all_heads=all_head_baseline,
    )
    if result.attribution.shape != (len(sequences), len(heads), TOKEN_COUNT):
        raise ValueError(
            f"attribution shape {result.attribution.shape} is not "
            f"({len(sequences)}, {len(heads)}, {TOKEN_COUNT})"
        )
    return result


# --- artifact --------------------------------------------------------------


def write_ig_artifact(
    out_dir: str | Path,
    result: IGResult,
    *,
    stratum: str,
    checkpoint_dir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``ig.npz`` and ``ig.sidecar.json``; return the archive's path.

    ``f_baseline`` goes in the sidecar and ``completeness_residual`` into the
    archive because they are the audit trail for the all-zero-attribution
    failure: without them the check cannot be re-run once the Colab session is
    gone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "attribution": result.attribution.astype(np.float32),
        "completeness_residual": result.completeness_residual.astype(np.float64),
        "row_index": result.row_index.astype(np.int64),
    }
    validate_artifact_keys(arrays)
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} holds a non-finite value; NaN is forbidden")
    if np.any(np.diff(arrays["row_index"]) <= 0):
        raise ValueError("row_index must be written strictly ascending")

    path = out_dir / "ig.npz"
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)

    report = completeness_report(
        result.completeness_residual, result.f_x, result.f_baseline
    )
    sidecar: dict[str, Any] = {
        "model": "9-Hydra v3",
        "analysis": "integrated_gradients",
        "stratum": stratum,
        "rows": int(result.row_index.size),
        "heads": list(result.heads),
        "steps": result.steps,
        "attribution_point": ATTRIBUTION_POINT,
        "baseline": BASELINE,
        "attribution_note": (
            "Gradients are taken on the encoder input embedding with a "
            "zero-embedding baseline (ADR 0006). The hidden axis is summed "
            "away, so one value per token per head."
        ),
        "token_resolution": TOKEN_COUNT,
        "token_names": list(TOKEN_NAMES),
        "token_boundaries": [list(bounds) for bounds in TOKEN_BOUNDARIES],
        "token_resolution_note": (
            "ADR 0007: the last token carries PAM (24-26) and the right flank "
            f"(27-29) together, so it is named {PAM_TOKEN_NAME!r}. No "
            "base-level PAM attribution is derivable from this artifact; ISM "
            "carries that result."
        ),
        "f_baseline": dict(result.f_baseline_all_heads),
        "f_baseline_note": (
            "f(0) at the zero embedding, for all nine heads. The three "
            "analysed heads' values are the ones the completeness residual is "
            "measured against; the other six are recorded, not analysed."
        ),
        "special_attribution_mean": [
            float(value) for value in result.special_attribution.mean(axis=0)
        ],
        "special_attribution_note": (
            "Mean attribution landing on <cls> and padding, per head. Not part "
            "of the reported profile; recorded so the completeness residual "
            "reads as integration error rather than an accounting gap."
        ),
        "completeness": report,
        "versions": v3_predict._library_versions(),
    }
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        sidecar["checkpoint"] = {
            "path": checkpoint_dir.as_posix(),
            "files": {
                name: v3_predict.sha256_of(checkpoint_dir / name)
                for name in ("config.json", "mu_sd.npy", "vocab.txt")
                if (checkpoint_dir / name).exists()
            },
        }
    if extra:
        sidecar.update(extra)
    sidecar["attribution_sha256"] = v3_predict.sha256_of(path)

    (out_dir / "ig.sidecar.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path
