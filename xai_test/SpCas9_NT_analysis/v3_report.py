"""v3's aggregation and reporting step (#134, #129's AC10 and AC13).

Three artifacts, no model, no GPU. Everything here reads the `.npz` arrays,
the prediction export, the labels and #135's measured noise floor.

**AC10 — IG against ISM.** The two techniques do not work at the same
resolution: ISM substitutes one base at a time, IG attributes to NT's
non-overlapping 6-mer tokens, and the last of those tokens binds the PAM
(24-26) to the right flank (27-29). So the comparison is made on **IG's five
tokens**, with the ISM sweep folded onto the same boundaries, and
`token_bounds` is derived from `v3_ig` rather than restated — a report that
aggregated to its own idea of a token would compare two different windows and
read the difference as disagreement. **A mismatch is reported, not raised.**
ADR 0007 is explicit that base-resolution ISM and token-resolution IG have no
reason to coincide where one token merges two regions, so a low correlation
here is the limit of what token-level attribution supports; it is not a bug
and nothing is retried, thresholded or dropped until it rises.

Two profile variants, because one number cannot tell the two failures apart.
`magnitude` asks "do the two techniques rank the tokens the same way?" and
`signed` asks "do they agree about which way the token pushes the
prediction?". A head can agree on where the model reads and disagree about the
direction, and a single magnitude correlation would hide that.

**AC13 — head-order reversal, as a result.** A row is a reversal when the
analysed heads' predicted ordering disagrees with their measured ordering
(Kendall tau < 0 over the three head pairs). Reversal was planned as a
case-selection criterion and **is not one**: requiring every inverted pair to
be separated by more than the **replicate noise floor** leaves a single row of
203, so lowering the margin to manufacture a pool would report measurement
noise as model failure. `selection_pool` says so in the artifact and the
report carries no row list to pick from — counts and distributions only.

The margin is **per head pair and never the pooled constant**. The three
heads' floors span 7.1x (`SpCas9` 0.064, `SpCas9-NG` 0.070, `evoCas9`
0.0099), so the pooled median is about five times `evoCas9`'s own and would
wave through that head's noise as a result. A pair's floor is the **larger**
of its two heads' floors: the pair is ordered within noise unless its label
gap clears both. The pooled rule is still counted, under its own key, for
continuity with the figure the design doc first quoted — reported beside the
per-head rule, never in place of it.

The floor is **read from #135's artifact**, with its limit counts, and
`read_noise_floor` refuses a payload that cannot state them or whose bytes do
not match the checksum beside it. AC13 has to report that the floor rests on
79 repeated 30-mers (379 rows) of the 3,851-row intersection, and a number
copied into prose by hand is the one that goes stale.

**Ties have no ordering to reverse.** Where two heads hold the same measured
label, tau is deflated rather than negative, so such a row is counted under
`tied_rows` and left out of the reversal count — which is the difference
between the 204 rows with tau < 0 and the 203 reversals the design doc
quotes. The counts are the artifact's and are deliberately not repeated here:
a figure written into prose is the one that goes stale.

**The annotation points back at the selection and cannot rewrite it.**
`cases_annotated.json` addresses `cases.json` by `case_id`, in its order, and
carries no selection score — no absolute error, no head gap, no error
quartile — so there is nothing in it to re-rank on. Attribution outliers are
described and never selected on: selecting cases by the attribution the case
study then explains is circular, which is why `cases.json` is written from
predictions and labels alone and this file depends on it rather than the other
way round.

Attention rollout is not an input to any of the three. It has no head axis
(ADR 0008), so it cannot describe a per-head reading, and the two reports here
are per head throughout.

Everything describes the model's internal information use. Nothing here is a
claim about physical DNA/protein interaction.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import v3_agreement
import v3_ig
import v3_noise
import v3_predict
import v3_selection
import v3_strata
from v3_agreement import head_pairs, pair_key, summarise_values
from v3_strata import ANALYSED_HEADS, GENERALISATION, PAM_LIBRARY, STRATUM_NAMES

SUBTREE_ROOT = Path(__file__).resolve().parent

# --- the token contract, borrowed rather than restated ---------------------

#: IG's five 6-mer tokens. Taken from `v3_ig` so that one module owns the
#: tokenise contract (ADR 0007) and this one cannot drift from it.
TOKEN_COUNT = v3_ig.TOKEN_COUNT
TOKEN_NAMES: tuple[str, ...] = v3_ig.TOKEN_NAMES
PAM_TOKEN_NAME = v3_ig.PAM_TOKEN_NAME

POSITIONS = v3_predict.SEQUENCE_LENGTH

#: The PAM's position range, for the share a case's annotation reports. The
#: base-resolution figure stays ISM's (ADR 0007); this is a share of the
#: window's sensitivity, not a per-base attribution.
PAM_RANGE = v3_agreement.PAM_RANGE

MAGNITUDE = "magnitude"
SIGNED = "signed"

#: Two variants, because one correlation cannot separate "the techniques rank
#: the tokens differently" from "they disagree about the direction".
PROFILE_VARIANTS: tuple[str, ...] = (MAGNITUDE, SIGNED)

VARIANT_DEFINITIONS: dict[str, str] = {
    MAGNITUDE: (
        "|attribution| against the ISM position profile (mean |delta| over a "
        "position's three alternative bases), both aggregated to the five "
        "tokens. Answers whether the two techniques rank the tokens alike."
    ),
    SIGNED: (
        "the signed attribution against the signed ISM delta, mean over a "
        "position's three alternative bases, both aggregated to the five "
        "tokens. Answers whether they agree about which way a token pushes "
        "the prediction. A head can rank alike and sign oppositely, so the "
        "two variants are reported apart and neither replaces the other."
    ),
}

VALIDATION_STATISTIC = "spearman_rank_correlation"

#: Which way round agreement reads, per variant. The two techniques' signs
#: refer to **opposite operations**: IG's attribution is what the base that is
#: there contributes to the prediction, while an ISM delta is what replacing
#: it does. A token the model leans on therefore has a positive attribution
#: and a negative delta, so in the signed variant the two agree when they
#: *anti*-correlate. Recorded as data rather than left to prose, because a
#: median of -0.7 reads as flat disagreement to anyone who has not been told.
AGREEMENT_SIGN: dict[str, int] = {MAGNITUDE: +1, SIGNED: -1}

SIGNED_CONVENTION_NOTE = (
    "In the `signed` variant, agreement is a **negative** correlation and "
    "`per_row.share_negative` is the share of rows that agree. IG's sign is "
    "the contribution of the base that is present; an ISM delta's sign is the "
    "effect of substituting it away, so the two run opposite by construction "
    "on a token the model relies on. `agreement_sign` carries the expected "
    "direction per variant; multiply by it to read a correlation as "
    "agreement. The `magnitude` variant has no such convention - there, "
    "agreement is positive."
)

ISM_AGGREGATION = (
    "the ISM profile's mean over each token's six positions. IG attributes to "
    "tokens and ISM substitutes bases, so the comparison is made at the "
    "coarser of the two resolutions; folding IG's one value onto six bases "
    "instead would invent a per-base number ADR 0007 forbids."
)

MISMATCH_NOTE = (
    "A low correlation is a limit of token-level attribution, not a bug. The "
    "last token binds the PAM (24-26) to the right flank (27-29), so "
    "base-resolution ISM and token-resolution IG have no reason to coincide "
    "there (docs/adr/0007-ism-carries-pam-resolution-not-ig.md). Disagreement "
    "is reported as measured; nothing is retried, re-thresholded or dropped "
    "to raise it, and neither technique is treated as the other's ground "
    "truth."
)

CLAIM_LIMIT = (
    "Every figure here describes the model's internal information use - which "
    "positions or tokens its prediction depends on, and how its heads order "
    "one row. None of it is a claim about physical DNA/protein interaction."
)

ROLLOUT_EXCLUDED = (
    "Attention rollout is not an input. It has no head axis at all "
    "(docs/adr/0008-attention-rollout-is-head-invariant.md), so it cannot "
    "describe a per-head reading, and every figure here is per head."
)

#: Written beside a library-stratum **aggregate**, and deliberately not the
#: case table's `"observed in one of 5 targets"`. That wording is a claim
#: about one row, and stamped on a 1,287-row stratum spanning all five
#: targets it would read as "this figure came from one target". The two
#: spellings are two different limits: the per-case string is copied out of
#: `cases.json` verbatim by the annotation, and this one labels the pools.
LIBRARY_QUALIFIER = v3_agreement.LIBRARY_QUALIFIER

LIBRARY_QUALIFIER_NOTE = (
    "This stratum's rows are PAM variants of 5 protospacers, so its n is 5 "
    "targets rather than its row count, and that count is never added to the "
    "other stratum's. Every pooled figure in this entry is a figure over "
    "variants of those 5 targets, which is why `per_target` repeats it target "
    "by target: a rate that holds on the pool because one target carries it "
    "is a different result from one that holds on all five."
)

POOLING = "forbidden"

POOLING_NOTE = (
    "One entry per stratum and no combined one. The two strata answer "
    f"different questions over differently-shaped populations: "
    f"{GENERALISATION} is sequence generalisation on unseen protospacers, "
    f"{PAM_LIBRARY} is a PAM experiment on 5 targets, so adding their row "
    "counts would add a sample size to a replication factor. `population` is "
    "not their sum either: the intersection also holds rows that are in "
    "**neither** stratum (non-library, non-NGG), which is why it is reported "
    "under its own name rather than as a total."
)

#: The sweep's population, named as `v3_agreement` names it.
POPULATION = v3_agreement.POPULATION

IG_POPULATION = "ig_subset"

# --- AC13's vocabulary -----------------------------------------------------

KENDALL_STATISTIC = "kendall_tau_a"

REVERSAL_DEFINITION = (
    "A row whose predicted ordering of the analysed heads disagrees with "
    "their measured ordering: Kendall tau (tau-a over the three head pairs, "
    "concordant minus discordant over 3) below zero, with no tied pair on "
    "either side. The tau is taken per row across heads - it says nothing "
    "about how the heads rank rows."
)

TIE_POLICY = (
    "A pair whose two measured labels are equal has no ordering to reverse, "
    "and tau is deflated rather than negative on such a row, so a row with a "
    "tied pair (on either side) is counted under `tied_rows` and left out of "
    "`reversal_rows`. `tau_negative_rows` counts every negative tau, so the "
    "difference between the two is visible rather than silent."
)

SELECTION_POOL = "forbidden"

SELECTION_POOL_NOTE = (
    "Head-order reversal is a reported result and not a case-selection pool "
    "(#129 AC9). Requiring every inverted pair to clear the replicate noise "
    "floor leaves a single row, so lowering the margin to fill a pool would "
    "report measurement noise as model failure. Case selection uses "
    "cross-head error disagreement instead, from predictions and labels "
    "alone. This report therefore carries counts and distributions, and no "
    "list of rows to pick from."
)

PAIR_FLOOR_RULE = (
    "every inverted pair's measured label gap exceeds that pair's floor, "
    "which is the larger of its two heads' own floors"
)

PAIR_FLOOR_NOTE = (
    "The larger of the two, not the mean and never the pooled constant: the "
    "pair is ordered within noise unless its gap clears both heads, and the "
    "three per-head floors span head_floor_ratio. The pooled rule is counted "
    "separately as `pooled_rule_rows` for continuity with the figure the "
    "design doc first quoted, and is not what `rows` applies."
)

GAP_SOURCE = "measured_labels"

GAP_NOTE = (
    "The gap is between the two heads' **measured labels**, on the fraction "
    "scale: whether the true ordering is resolvable by the assay at all is "
    "what the noise floor answers. The predicted gap is summarised beside it "
    "and is not what the margin tests - a confident model separating two "
    "measurements that the assay cannot separate is the failure mode this "
    "margin exists to catch."
)

#: Which of the floor artifact's statistics the margin uses. The median is the
#: floor's own headline; p90 is available in the artifact for a stricter read.
DEFAULT_FLOOR_STATISTIC = "median"

# --- the annotation's vocabulary -------------------------------------------

NON_CIRCULARITY_NOTE = (
    "This file depends on cases.json and cases.json depends on nothing "
    "downstream, so a reviewer can confirm non-circularity from the two "
    "files' dependency direction. The selection is addressed by `case_id` in "
    "the order the case table holds it, and no selection score (absolute "
    "error, head gap, error quartile, selection rank) is copied here - there "
    "is nothing in this file to re-rank on."
)

OUTLIER_USE = "description only, never selection"

OUTLIER_PERCENTILE = 90

OUTLIER_DEFINITION = (
    "A case whose PAM share of the window's ISM sensitivity sits at or above "
    f"the {OUTLIER_PERCENTILE}th percentile of **its own stratum's** rows, "
    "per head. Reported to describe a case the selection already made: "
    "selecting cases by the attribution the case study then explains is "
    "circular."
)

PAM_SHARE_DEFINITION = (
    "the case's mean ISM sensitivity over PAM positions 24-26 divided by its "
    "mean over the whole 30bp window, per head. Scale-free, because the "
    "heads' dynamic ranges differ about sixfold and a raw magnitude would "
    "report that difference as a difference in where the model reads. It is a "
    "share of the window, not a per-base PAM attribution."
)

# --- artifact names --------------------------------------------------------

REPORT_DIR = "v3_report"
VALIDATION_NAME = "ig_ism_validation.json"
REVERSAL_NAME = "head_order_reversal.json"

CASES_DIR = v3_selection.CASES_DIR
CASES_NAME = v3_selection.CASES_NAME
CASES_CHECKSUM_NAME = v3_selection.CASES_CHECKSUM_NAME
ANNOTATED_NAME = "cases_annotated.json"

NOISE_NAME = v3_noise.NOISE_NAME
NOISE_CHECKSUM_NAME = v3_noise.NOISE_CHECKSUM_NAME

DEFAULT_ARTIFACTS_ROOT = SUBTREE_ROOT / "artifacts"
DEFAULT_ISM_NPZ = DEFAULT_ARTIFACTS_ROOT / "v3_ism" / "ism.npz"
DEFAULT_IG_NPZ = DEFAULT_ARTIFACTS_ROOT / "v3_ig" / "ig.npz"
DEFAULT_CASES_JSON = DEFAULT_ARTIFACTS_ROOT / CASES_DIR / CASES_NAME
DEFAULT_NOISE_FLOOR = DEFAULT_ARTIFACTS_ROOT / NOISE_NAME
DEFAULT_PREDICTIONS_CSV = v3_selection.DEFAULT_PREDICTIONS_CSV
DEFAULT_SPLIT_CSV = v3_selection.DEFAULT_SPLIT_CSV
DEFAULT_SPLIT_CHECKSUM = v3_selection.DEFAULT_SPLIT_CHECKSUM


# --- token aggregation -----------------------------------------------------


def _finite_or_none(value: float) -> float | None:
    """JSON holds no NaN, and an undefined correlation is not a zero.

    A copy of `v3_agreement`'s two-liner rather than an import of it: that one
    is private, and widening a sibling's API to share two lines would couple
    the two modules where duplicating the rule costs nothing.
    """
    return float(value) if np.isfinite(value) else None


def _verify_checksum(path: Path, checksum_name: str) -> None:
    """Refuse an artifact whose bytes do not match the checksum beside it.

    Shared by the two tracked inputs this module reads back (the measured
    floor and the case table), because "the file was edited after it was
    written" is the same failure in both and a report computed off either is
    addressed to bytes that no longer exist.
    """
    checksum = path.with_name(checksum_name)
    if not checksum.exists():
        return
    pinned = json.loads(checksum.read_text(encoding="utf-8"))["files"][path.name]
    digest = v3_predict.sha256_of(path)
    if pinned != digest:
        raise ValueError(
            f"{path.name} does not match its checksum file: pinned {pinned}, "
            f"read {digest}"
        )


def token_bounds() -> tuple[tuple[int, int], ...]:
    """IG's five tokens as `[start, stop)` position ranges.

    Taken from `v3_ig.token_slice`, which already owns the conversion from the
    inclusive boundaries a sidecar prints to the half-open ones an array is
    sliced by. Re-deriving it here would put the one off-by-one that matters
    in two modules.
    """
    slices = [v3_ig.token_slice(index) for index in range(TOKEN_COUNT)]
    return tuple((one.start, one.stop) for one in slices)


def _require_window(array: np.ndarray, axis: int = 1) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.shape[axis] != POSITIONS:
        raise ValueError(
            f"expected {POSITIONS} positions on axis {axis}, got "
            f"{array.shape[axis]}: the token fold is defined on the 30bp "
            f"SpCas9 window"
        )
    return array


def ism_token_profile(profile: np.ndarray) -> np.ndarray:
    """`(rows, 30, heads)` -> `(rows, 5, heads)`, the mean over each token.

    The mean rather than the sum: the two read the same under a rank
    correlation while the mean stays comparable to the per-position figures
    the ISM report quotes.
    """
    profile = _require_window(profile)
    return np.stack(
        [profile[:, start:stop, :].mean(axis=1) for start, stop in token_bounds()],
        axis=1,
    )


def _require_variant(variant: str) -> str:
    if variant not in PROFILE_VARIANTS:
        raise ValueError(
            f"unknown profile variant {variant!r}; expected one of "
            f"{list(PROFILE_VARIANTS)}"
        )
    return variant


def ism_token_tensor(delta: np.ndarray, variant: str = MAGNITUDE) -> np.ndarray:
    """`(rows, 30, 4, heads)` ISM delta -> `(rows, 5, heads)` token profile.

    Both variants divide a position by its **three** alternative bases: the
    reference slot is exactly 0 by construction, so dividing by four would
    scale every position by 3/4 and average over a base that was never
    substituted.
    """
    _require_variant(variant)
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 4 or delta.shape[2] != len(v3_agreement.BASES):
        raise ValueError(
            f"expected a (rows, positions, 4, heads) delta, got {delta.shape}"
        )
    if variant == MAGNITUDE:
        positions = v3_agreement.position_profile(delta)
    else:
        positions = delta.sum(axis=2) / v3_agreement.ALTERNATIVES_PER_POSITION
    return ism_token_profile(positions)


def ig_token_tensor(attribution: np.ndarray, variant: str = MAGNITUDE) -> np.ndarray:
    """`(rows, heads, 5)` IG attribution -> `(rows, 5, heads)`.

    Re-laid onto the ISM convention (tokens in the middle, heads last) rather
    than correlating the two archives' own axis orders against each other.
    """
    _require_variant(variant)
    attribution = np.asarray(attribution, dtype=np.float64)
    if attribution.ndim != 3 or attribution.shape[2] != TOKEN_COUNT:
        raise ValueError(
            f"expected a (rows, heads, {TOKEN_COUNT}) attribution, got "
            f"{attribution.shape}"
        )
    if variant == MAGNITUDE:
        attribution = np.abs(attribution)
    return np.moveaxis(attribution, 1, 2)


def _as_head_rows(tokens: np.ndarray) -> np.ndarray:
    """`(rows, tokens, heads)` -> `(rows, heads, tokens)` for `spearman_rows`."""
    return np.moveaxis(np.asarray(tokens, dtype=np.float64), -1, 1)


def profile_correlation(ism_tokens: np.ndarray, ig_tokens: np.ndarray) -> np.ndarray:
    """`(rows, heads)` per-row Spearman correlation of the two token profiles.

    NaN where a row's profile is constant over the five tokens: the
    correlation is undefined there, not zero, and `summarise_values` counts
    those rows apart rather than letting a zero into the distribution.
    """
    ism_tokens, ig_tokens = np.asarray(ism_tokens), np.asarray(ig_tokens)
    if ism_tokens.shape != ig_tokens.shape:
        raise ValueError(
            f"the two token profiles have different shapes: {ism_tokens.shape} "
            f"and {ig_tokens.shape}"
        )
    return v3_agreement.spearman_rows(
        _as_head_rows(ism_tokens), _as_head_rows(ig_tokens)
    )


def mean_profile_correlation(
    ism_tokens: np.ndarray, ig_tokens: np.ndarray
) -> np.ndarray:
    """`(heads,)`: the correlation of the two **mean** token profiles.

    A different question from `profile_correlation`'s distribution, not a
    summary of it: a population whose rows disagree one by one can still have
    mean profiles that line up, and the two are reported side by side.
    """
    ism_mean = np.asarray(ism_tokens, dtype=np.float64).mean(axis=0)
    ig_mean = np.asarray(ig_tokens, dtype=np.float64).mean(axis=0)
    return v3_agreement.spearman_rows(ism_mean.T, ig_mean.T)


# --- head-order reversal ---------------------------------------------------


def _index_pairs(count: int) -> tuple[tuple[int, int], ...]:
    """Column index pairs in `head_pairs` order, so keys and columns align."""
    return tuple(itertools.combinations(range(count), 2))


def _pair_signs(values: np.ndarray) -> np.ndarray:
    """`(rows, heads)` -> `(rows, pairs)` sign of the within-row difference."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a (rows, heads) array, got {values.shape}")
    pairs = _index_pairs(values.shape[1])
    return np.stack(
        [np.sign(values[:, left] - values[:, right]) for left, right in pairs], axis=1
    )


def _concordance(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape:
        raise ValueError(
            f"labels are {labels.shape} and predictions {predictions.shape}; "
            "both are (rows, heads) on the same rows in the same head order"
        )
    return _pair_signs(labels) * _pair_signs(predictions)


def kendall_tau(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """`(rows,)` tau-a across the heads of one row.

    Concordant minus discordant over the number of pairs, so three heads give
    a tau in {-1, -1/3, 1/3, 1} when nothing is tied. tau-a rather than tau-b
    because the tie correction only rescales a statistic whose tied rows are
    excluded anyway (see `TIE_POLICY`), and the uncorrected form keeps "two of
    three pairs are concordant" readable as 1/3.
    """
    concordance = _concordance(labels, predictions)
    return concordance.sum(axis=1) / concordance.shape[1]


def tied_pairs(values: np.ndarray) -> np.ndarray:
    """`(rows, pairs)`: the pairs whose two values are equal."""
    return _pair_signs(values) == 0


def tied_rows(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """`(rows,)`: rows holding a tie on either side.

    Both sides, symmetrically. The measured labels are where ties actually
    occur, but a tied *prediction* is as undefined an ordering as a tied
    label, and a rule that applied to one side only would read as an oversight
    the first time the other produced one.
    """
    return tied_pairs(labels).any(axis=1) | tied_pairs(predictions).any(axis=1)


def inverted_pairs(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """`(rows, pairs)`: the discordant pairs, in `head_pairs` order."""
    return _concordance(labels, predictions) < 0


def reversal_rows(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """`(rows,)`: negative tau and no tie — the reported reversals."""
    return (kendall_tau(labels, predictions) < 0) & ~tied_rows(labels, predictions)


def pair_gaps(values: np.ndarray) -> np.ndarray:
    """`(rows, pairs)`: `|left - right|`, in `head_pairs` order."""
    values = np.asarray(values, dtype=np.float64)
    pairs = _index_pairs(values.shape[1])
    return np.stack(
        [np.abs(values[:, left] - values[:, right]) for left, right in pairs], axis=1
    )


def floor_vector(
    noise_floor: Mapping[str, Any],
    heads: Sequence[str] = ANALYSED_HEADS,
    *,
    statistic: str = DEFAULT_FLOOR_STATISTIC,
) -> np.ndarray:
    """`(pairs,)`: each pair's floor, the larger of its two heads' own.

    Every lookup goes through `v3_noise.floor_for_head`, which refuses
    `"pooled"` — so a caller cannot reach for the single constant that is
    several times `evoCas9`'s floor by passing it as a head name.
    """
    if "by_head" not in noise_floor:
        raise ValueError(
            "the noise floor artifact has no `by_head`: the margin is per "
            "head pair, and the pooled median is several times evoCas9's own "
            "floor, so there is no single constant to fall back to"
        )
    return np.array(
        [
            max(
                v3_noise.floor_for_head(noise_floor, left, statistic=statistic),
                v3_noise.floor_for_head(noise_floor, right, statistic=statistic),
            )
            for left, right in head_pairs(heads)
        ],
        dtype=np.float64,
    )


def pair_floors(
    noise_floor: Mapping[str, Any],
    heads: Sequence[str] = ANALYSED_HEADS,
    *,
    statistic: str = DEFAULT_FLOOR_STATISTIC,
) -> dict[str, float]:
    """The same floors, keyed as the artifact reports them."""
    floors = floor_vector(noise_floor, heads, statistic=statistic)
    return {
        pair_key(left, right): float(floors[index])
        for index, (left, right) in enumerate(head_pairs(heads))
    }


def clears_noise_floor(
    gaps: np.ndarray, inverted: np.ndarray, floors: Sequence[float]
) -> np.ndarray:
    """`(rows,)`: every inverted pair is separated by more than its floor.

    A row with no inverted pair does not clear the floor — it has nothing to
    clear — so the result is a mask over reversals rather than a mask over
    rows that happen to be wide apart.
    """
    gaps = np.asarray(gaps, dtype=np.float64)
    inverted = np.asarray(inverted, dtype=bool)
    floors = np.asarray(floors, dtype=np.float64)
    if gaps.shape != inverted.shape or gaps.shape[1] != floors.size:
        raise ValueError(
            f"gaps {gaps.shape}, inverted {inverted.shape} and floors "
            f"{floors.shape} must agree on the pair axis"
        )
    clear = gaps > floors
    return inverted.any(axis=1) & np.all(clear | ~inverted, axis=1)


# --- reading #135's artifact -----------------------------------------------

#: What a report leaning on the floor has to be able to state. `groups` and
#: `rows` are AC13's limit, `pooled` is the continuity figure, and
#: `repeated_*_in_split` are what the uncut split holds — the sharper edge of
#: the limit, since most repeated 30-mers do not survive the intersection.
NOISE_FLOOR_REQUIRED: tuple[str, ...] = (
    "by_head",
    "pooled",
    "groups",
    "rows",
    "limit_note",
    "head_floor_ratio",
    "intersection_rows",
    "repeated_groups_in_split",
    "repeated_rows_in_split",
)


def read_noise_floor(path: str | Path = DEFAULT_NOISE_FLOOR) -> dict[str, Any]:
    """#135's measured floor, refused rather than defaulted.

    AC13 says the floor is read from the artifact and not written by hand, so
    the read is where that is enforced: a payload that cannot state the limit
    it rests on is not a floor this report can lean on, and bytes that do not
    match the checksum beside them are not the floor that was measured.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"no noise floor artifact at {path.name}: AC13's margin is "
            f"measured by v3_noise.py, not written by hand"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))

    missing = [key for key in NOISE_FLOOR_REQUIRED if key not in payload]
    if missing:
        raise ValueError(
            f"the noise floor artifact {path.name} records no {missing}: a "
            f"report that leans on the floor has to state the limit it rests "
            f"on, so the counts are read from here rather than written by hand"
        )
    _verify_checksum(path, NOISE_CHECKSUM_NAME)
    return payload


def floor_block(
    noise_floor: Mapping[str, Any],
    heads: Sequence[str] = ANALYSED_HEADS,
    *,
    statistic: str = DEFAULT_FLOOR_STATISTIC,
) -> dict[str, Any]:
    """The margin, with its per-head floors and the limit it rests on."""
    return {
        "source": NOISE_NAME,
        "statistic": statistic,
        "by_head": {
            head: v3_noise.floor_for_head(noise_floor, head, statistic=statistic)
            for head in heads
        },
        "pair_floors": pair_floors(noise_floor, heads, statistic=statistic),
        "pair_floor_rule": PAIR_FLOOR_RULE,
        "pair_floor_note": PAIR_FLOOR_NOTE,
        "head_floor_ratio": noise_floor["head_floor_ratio"],
        "pooled_median": noise_floor["pooled"][statistic],
        "pooled_applied": False,
        "gap_source": GAP_SOURCE,
        "gap_note": GAP_NOTE,
        "limit": {
            "source": NOISE_NAME,
            "groups": noise_floor["groups"],
            "rows": noise_floor["rows"],
            "intersection_rows": noise_floor["intersection_rows"],
            "repeated_groups_in_split": noise_floor["repeated_groups_in_split"],
            "repeated_rows_in_split": noise_floor["repeated_rows_in_split"],
            "note": noise_floor["limit_note"],
        },
    }


# --- loading the two archives ----------------------------------------------


def read_ig_sidecar(ig_npz: str | Path) -> dict[str, Any]:
    """The IG run's sidecar, refused rather than defaulted when it cannot pin.

    The token names are checked here, where the archive is opened: a run that
    named its tokens differently is a run this comparison cannot line up, and
    discovering that after the correlation is computed means discovering it as
    a surprising result rather than as a failure.
    """
    ig_npz = Path(ig_npz)
    sidecar = ig_npz.with_name(f"{ig_npz.stem}.sidecar.json")
    if not sidecar.exists():
        raise ValueError(
            f"{ig_npz.name} has no sidecar at {sidecar.name}: without it the "
            f"report cannot say which run produced its numbers"
        )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    missing = [
        key
        for key in ("heads", "code_ref", "token_names", "subset")
        if not payload.get(key)
    ]
    if missing:
        raise ValueError(
            f"{sidecar.name} records no {missing}: an IG run is addressed by "
            f"its code_ref and the subset it ran on, so a report cannot be "
            f"computed off a sidecar that omits them"
        )
    if tuple(payload["token_names"]) != TOKEN_NAMES:
        raise ValueError(
            f"{sidecar.name} names its tokens {payload['token_names']}, not "
            f"{list(TOKEN_NAMES)}: the two profiles are compared token by "
            f"token, so a differently-tokenised run cannot be lined up "
            f"against the ISM sweep"
        )
    return payload


class IGAttribution:
    """One IG run's attribution, cut to the analysed heads."""

    def __init__(
        self,
        row_index: np.ndarray,
        heads: tuple[str, ...],
        attribution: np.ndarray,
        sidecar: Mapping[str, Any],
    ) -> None:
        self.row_index = row_index
        self.heads = heads
        #: `(rows, heads, 5)` float64, signed, in `heads` order.
        self.attribution = attribution
        self.sidecar = dict(sidecar)

    def select(self, rows: np.ndarray) -> IGAttribution:
        rows = np.asarray(rows, dtype=np.int64)
        return IGAttribution(
            row_index=self.row_index[rows],
            heads=self.heads,
            attribution=self.attribution[rows],
            sidecar=self.sidecar,
        )


def load_ig_attribution(
    ig_npz: str | Path = DEFAULT_IG_NPZ, heads: Sequence[str] = ANALYSED_HEADS
) -> IGAttribution:
    """Read `ig.npz`, keep the analysed heads in the order asked for.

    The stored head order comes from the sidecar rather than from
    `ANALYSED_HEADS`, for the reason `v3_agreement.load_ism_profiles` reads
    it: a re-ordered archive would otherwise be sliced into the wrong heads
    and the error would surface as a result.
    """
    ig_npz = Path(ig_npz)
    sidecar = read_ig_sidecar(ig_npz)
    stored = tuple(sidecar["heads"])

    with np.load(ig_npz) as archive:
        attribution = np.asarray(archive["attribution"], dtype=np.float64)
        row_index = np.asarray(archive["row_index"], dtype=np.int64)

    if attribution.ndim != 3 or attribution.shape[1:] != (len(stored), TOKEN_COUNT):
        raise ValueError(
            f"the attribution is {attribution.shape}; expected (rows, "
            f"{len(stored)}, {TOKEN_COUNT}) for the sidecar's heads"
        )
    if np.any(np.diff(row_index) <= 0):
        raise ValueError("row_index must be strictly ascending")
    if not np.isfinite(attribution).all():
        raise ValueError("the attribution holds a non-finite value")

    missing = [head for head in heads if head not in stored]
    if missing:
        raise ValueError(f"the IG archive does not store head(s) {missing}")
    taken = attribution[:, [stored.index(head) for head in heads], :]
    return IGAttribution(
        row_index=row_index,
        heads=tuple(heads),
        attribution=taken,
        sidecar=sidecar,
    )


def _archive_rows(row_index: np.ndarray) -> dict[int, int]:
    """`row_index` -> position on axis 0, which is the only join key."""
    return {int(value): index for index, value in enumerate(row_index)}


def _positions_of(
    wanted: Sequence[int], archive_row_of: Mapping[int, int]
) -> np.ndarray:
    missing = [int(value) for value in wanted if int(value) not in archive_row_of]
    if missing:
        raise ValueError(
            f"{len(missing)} row(s) were never swept by ISM (first: "
            f"{missing[:3]}): the sweep covers the analysis intersection, and "
            f"a row outside it has no profile to compare"
        )
    return np.array([archive_row_of[int(value)] for value in wanted], dtype=np.int64)


# --- AC10: the IG <-> ISM validation report --------------------------------


def head_validation(
    ism_tokens: np.ndarray,
    ig_tokens: np.ndarray,
    head_index: int,
    variant: str = MAGNITUDE,
) -> dict[str, Any]:
    """One head's agreement between the two techniques, for one variant."""
    _require_variant(variant)
    per_row = profile_correlation(ism_tokens, ig_tokens)[:, head_index]
    mean = mean_profile_correlation(ism_tokens, ig_tokens)[head_index]
    return {
        "agreement_sign": AGREEMENT_SIGN[variant],
        "per_row": summarise_values(per_row),
        "mean_profile": _finite_or_none(mean),
        "mean_token_profile": {
            "ism": [
                float(value) for value in ism_tokens[:, :, head_index].mean(axis=0)
            ],
            "ig": [float(value) for value in ig_tokens[:, :, head_index].mean(axis=0)],
        },
    }


def stratum_validation(
    stratum: str,
    frame: pd.DataFrame,
    ism_variants: Mapping[str, np.ndarray],
    ig_variants: Mapping[str, np.ndarray],
    rows: np.ndarray,
    heads: Sequence[str],
) -> dict[str, Any]:
    """One stratum's IG-vs-ISM report. There is no combined counterpart.

    `rows` indexes the two token tensors, which are already on the IG subset's
    axis and joined by `row_index`; the frame is this stratum's slice of the
    split and carries the sequences the counts are taken from.
    """
    by_head: dict[str, Any] = {}
    for position, head in enumerate(heads):
        by_head[head] = {
            variant: head_validation(
                ism_variants[variant][rows],
                ig_variants[variant][rows],
                position,
                variant,
            )
            for variant in PROFILE_VARIANTS
        }

    targets = v3_strata.protospacers(frame["sequence"])
    report: dict[str, Any] = {
        "rows": int(rows.size),
        "distinct_protospacers": int(targets.nunique()),
        "by_head": by_head,
    }
    if stratum == PAM_LIBRARY:
        report["qualifier"] = LIBRARY_QUALIFIER
        report["qualifier_note"] = LIBRARY_QUALIFIER_NOTE
        report["per_target"] = per_target_validation(
            targets.to_numpy(), ism_variants, ig_variants, rows, heads
        )
    return report


def per_target_validation(
    targets: np.ndarray,
    ism_variants: Mapping[str, np.ndarray],
    ig_variants: Mapping[str, np.ndarray],
    rows: np.ndarray,
    heads: Sequence[str],
) -> list[dict[str, Any]]:
    """The library stratum's agreement repeated target by target.

    Pooling 1,287 PAM variants of 5 protospacers reads as 1,287 independent
    observations, so the pooled median is reported with the five it is made
    of. `targets` is positional over the same rows as `rows`.
    """
    entries: list[dict[str, Any]] = []
    for target in pd.unique(targets):
        inside = np.flatnonzero(targets == target)
        local = rows[inside]
        entries.append(
            {
                "protospacer": str(target),
                "rows": int(inside.size),
                "per_row_median": {
                    variant: {
                        head: _median_or_none(
                            profile_correlation(
                                ism_variants[variant][local],
                                ig_variants[variant][local],
                            )[:, position]
                        )
                        for position, head in enumerate(heads)
                    }
                    for variant in PROFILE_VARIANTS
                },
            }
        )
    return sorted(entries, key=lambda entry: (-entry["rows"], entry["protospacer"]))


def build_validation_report(
    *,
    ism_npz: str | Path = DEFAULT_ISM_NPZ,
    ig_npz: str | Path = DEFAULT_IG_NPZ,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    split_checksum: str | Path | None = DEFAULT_SPLIT_CHECKSUM,
    heads: Sequence[str] = ANALYSED_HEADS,
) -> dict[str, Any]:
    """AC10: the IG-vs-ISM rank correlation, per stratum and per head."""
    split = pd.read_csv(split_csv)
    intersection = v3_strata.analysis_intersection(split)
    strata = v3_strata.stratify(intersection)

    profiles = v3_agreement.load_ism_profiles(ism_npz, heads)
    ig = load_ig_attribution(ig_npz, heads)

    # The IG subset is 400 of the sweep's 3,851 rows, so the join is by
    # `row_index` in one direction only: every IG row must have been swept.
    swept = _archive_rows(profiles.row_index)
    cut = profiles.select(_positions_of(ig.row_index, swept))

    ism_variants = {
        variant: ism_token_tensor(cut.delta, variant) for variant in PROFILE_VARIANTS
    }
    ig_variants = {
        variant: ig_token_tensor(ig.attribution, variant)
        for variant in PROFILE_VARIANTS
    }

    subset_row_of = _archive_rows(ig.row_index)
    stratum_reports: dict[str, Any] = {}
    for name in STRATUM_NAMES:
        frame = strata[name]
        inside = frame.loc[frame["row_index"].isin(set(subset_row_of))]
        stratum_reports[name] = stratum_validation(
            name,
            inside,
            ism_variants,
            ig_variants,
            np.array(
                [subset_row_of[int(value)] for value in inside["row_index"]],
                dtype=np.int64,
            ),
            heads,
        )

    consumed = v3_agreement.consumed_inputs(
        ism_npz, split_csv, split_checksum=split_checksum
    )
    consumed["ig"] = {
        "name": Path(ig_npz).name,
        "sha256": v3_predict.sha256_of(ig_npz),
        "sidecar": {
            key: ig.sidecar[key]
            for key in (
                "code_ref",
                "rows",
                "steps",
                "token_names",
                "subset",
                "attribution_sha256",
                "versions",
            )
            if key in ig.sidecar
        },
    }

    return {
        "model": "9-Hydra v3",
        "analysis": "ig_ism_rank_correlation",
        "inputs": ["ig", "ism"],
        "statistic": VALIDATION_STATISTIC,
        "heads": list(heads),
        "token_resolution": TOKEN_COUNT,
        "token_names": list(TOKEN_NAMES),
        "token_boundaries": [list(bounds) for bounds in v3_ig.TOKEN_BOUNDARIES],
        "ism_aggregation": ISM_AGGREGATION,
        "variants": dict(VARIANT_DEFINITIONS),
        "agreement_sign": dict(AGREEMENT_SIGN),
        "signed_convention_note": SIGNED_CONVENTION_NOTE,
        "profile_definition": v3_agreement.PROFILE_DEFINITION,
        "mismatch_note": MISMATCH_NOTE,
        "population": {
            "name": IG_POPULATION,
            "rows": int(ig.row_index.size),
            "population_digest": v3_predict.population_digest(ig.row_index),
            "swept_rows": int(profiles.row_index.size),
            "note": (
                "The correlation is reported on the frozen IG subset, which is "
                "the smaller of the two populations: ISM swept the whole "
                "analysis intersection and IG ran on the subset. `swept_rows` "
                "is the sweep's population and is not this report's "
                "denominator."
            ),
        },
        "pooling": POOLING,
        "pooling_note": POOLING_NOTE,
        "nan_policy_note": (
            "A correlation over five tokens is undefined, not zero, where a "
            "profile is constant; such rows are counted as `undefined_rows` "
            "and a single undefined number is written as null."
        ),
        "rollout_excluded": ROLLOUT_EXCLUDED,
        "claim_limit": CLAIM_LIMIT,
        "strata": stratum_reports,
        "versions": v3_predict.library_versions(),
        "consumed": consumed,
    }


def write_validation_report(out_dir: str | Path, report: Mapping[str, Any]) -> Path:
    """Write `ig_ism_validation.json`; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return v3_predict.write_sidecar(out_dir, VALIDATION_NAME, dict(report))


# --- AC13: the head-order reversal report ----------------------------------


def _head_matrix(frame: pd.DataFrame, prefix: str, heads: Sequence[str]) -> np.ndarray:
    columns = [f"{prefix}{head}" for head in heads]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"the frame is missing {missing}; join predictions first")
    return frame.loc[:, columns].to_numpy(dtype=float)


def _median_or_none(values: np.ndarray) -> float | None:
    """The median of the defined values, or None when none are defined."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _share_or_none(count: int, total: int) -> float | None:
    """A share of a population that may be empty; JSON holds no division."""
    return float(count / total) if total else None


def reversal_summary(
    frame: pd.DataFrame,
    floors: np.ndarray,
    pooled_floor: float,
    heads: Sequence[str] = ANALYSED_HEADS,
) -> dict[str, Any]:
    """One population's reversal figures: counts, taus and gap distributions.

    Every pair statistic is taken over the **reversal rows only**, because the
    question AC13 asks is how far apart the pairs a reversal misorders were
    measured to be — a concordant pair's gap is not evidence about ordering.
    """
    labels = _head_matrix(frame, "label_", heads)
    predictions = _head_matrix(frame, "pred_", heads)

    tau = kendall_tau(labels, predictions)
    tied = tied_rows(labels, predictions)
    negative = tau < 0
    reversal = negative & ~tied
    inverted = inverted_pairs(labels, predictions)
    label_gaps = pair_gaps(labels)
    prediction_gaps = pair_gaps(predictions)

    clears = clears_noise_floor(label_gaps, inverted, floors) & reversal
    pooled = (
        clears_noise_floor(
            label_gaps, inverted, np.full(floors.size, float(pooled_floor))
        )
        & reversal
    )

    # Only the pairs a reversal row actually misorders.
    counted = inverted & reversal[:, None]
    ratio = np.divide(
        label_gaps,
        floors,
        out=np.full(label_gaps.shape, np.nan),
        where=floors > 0,
    )

    pairs: dict[str, Any] = {}
    for index, (left, right) in enumerate(head_pairs(heads)):
        column = counted[:, index]
        pairs[pair_key(left, right)] = {
            "rows": int(column.sum()),
            "floor": float(floors[index]),
            "label_gap": summarise_values(label_gaps[column, index]),
            "prediction_gap": summarise_values(prediction_gaps[column, index]),
            "rows_clearing_floor": int(
                (column & (label_gaps[:, index] > floors[index])).sum()
            ),
        }

    targets = v3_strata.protospacers(frame["sequence"])
    return {
        "rows": len(frame),
        "distinct_protospacers": int(targets.nunique()),
        "tau": summarise_values(tau),
        "tau_negative_rows": int(negative.sum()),
        "tied_rows": int(tied.sum()),
        "tied_rows_with_negative_tau": int((negative & tied).sum()),
        "reversal_rows": int(reversal.sum()),
        "reversal_rate": _share_or_none(int(reversal.sum()), len(frame)),
        "inverted_pairs": pairs,
        "gap_over_floor": summarise_values(ratio[counted]),
        "clearing_floor": {
            "rows": int(clears.sum()),
            "share_of_reversals": _share_or_none(
                int(clears.sum()), int(reversal.sum())
            ),
            "rule": PAIR_FLOOR_RULE,
            "pooled_rule_rows": int(pooled.sum()),
            "pooled_rule_note": (
                "The same count under the single pooled median, for "
                "continuity with the figure the design doc first quoted. It "
                "is reported beside the per-head rule, not in place of it."
            ),
        },
    }


def per_target_reversal(
    frame: pd.DataFrame,
    floors: np.ndarray,
    pooled_floor: float,
    heads: Sequence[str] = ANALYSED_HEADS,
) -> list[dict[str, Any]]:
    """The library stratum's reversal rate repeated target by target.

    Computed by running `reversal_summary` on each target's own rows rather
    than by re-deriving the masks, so a per-target rate cannot come to mean
    something slightly different from the pooled one it sits beside.
    """
    targets = v3_strata.protospacers(frame["sequence"]).to_numpy()
    entries: list[dict[str, Any]] = []
    for target in pd.unique(targets):
        inside = frame.loc[targets == target]
        summary = reversal_summary(inside, floors, pooled_floor, heads)
        entries.append(
            {
                "protospacer": str(target),
                "rows": summary["rows"],
                "reversal_rows": summary["reversal_rows"],
                "reversal_rate": summary["reversal_rate"],
                "rows_clearing_floor": summary["clearing_floor"]["rows"],
                "gap_over_floor_median": summary["gap_over_floor"].get("median"),
            }
        )
    return sorted(entries, key=lambda entry: (-entry["rows"], entry["protospacer"]))


def build_reversal_report(
    *,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    predictions_csv: str | Path = DEFAULT_PREDICTIONS_CSV,
    noise_floor: str | Path = DEFAULT_NOISE_FLOOR,
    split_checksum: str | Path | None = DEFAULT_SPLIT_CHECKSUM,
    heads: Sequence[str] = ANALYSED_HEADS,
    statistic: str = DEFAULT_FLOOR_STATISTIC,
) -> dict[str, Any]:
    """AC13: reversal as a reported result, against the replicate noise floor."""
    split = pd.read_csv(split_csv)
    predictions = pd.read_csv(predictions_csv)
    intersection = v3_strata.join_predictions(
        v3_strata.analysis_intersection(split), predictions
    )
    strata = v3_strata.stratify(intersection)

    floor_path = Path(noise_floor)
    measured = read_noise_floor(floor_path)
    floors = floor_vector(measured, heads, statistic=statistic)
    pooled_floor = measured["pooled"][statistic]

    consumed = v3_selection.consumed_inputs(
        split_csv, predictions_csv, split_checksum=split_checksum
    )
    consumed["noise_floor"] = floor_path.name
    consumed["digests"][floor_path.name] = v3_predict.sha256_of(floor_path)

    population = reversal_summary(intersection, floors, pooled_floor, heads)
    population["name"] = POPULATION

    stratum_reports: dict[str, Any] = {}
    for name in STRATUM_NAMES:
        entry = reversal_summary(strata[name], floors, pooled_floor, heads)
        if name == PAM_LIBRARY:
            entry["qualifier"] = LIBRARY_QUALIFIER
            entry["qualifier_note"] = LIBRARY_QUALIFIER_NOTE
            entry["per_target"] = per_target_reversal(
                strata[name], floors, pooled_floor, heads
            )
        stratum_reports[name] = entry

    return {
        "model": "9-Hydra v3",
        "analysis": "head_order_reversal",
        "inputs": ["predictions", "labels", "noise_floor"],
        "statistic": KENDALL_STATISTIC,
        "heads": list(heads),
        "head_pairs": [list(pair) for pair in head_pairs(heads)],
        "reversal_definition": REVERSAL_DEFINITION,
        "tie_policy": TIE_POLICY,
        "selection_pool": SELECTION_POOL,
        "selection_pool_note": SELECTION_POOL_NOTE,
        "floor": floor_block(measured, heads, statistic=statistic),
        "label_scale": "fraction",
        "prediction_scale": v3_predict.PREDICTION_SCALE,
        "population": population,
        "pooling": POOLING,
        "pooling_note": POOLING_NOTE,
        "rollout_excluded": ROLLOUT_EXCLUDED,
        "claim_limit": CLAIM_LIMIT,
        "strata": stratum_reports,
        "versions": v3_predict.library_versions(),
        "consumed": consumed,
    }


def write_reversal_report(out_dir: str | Path, report: Mapping[str, Any]) -> Path:
    """Write `head_order_reversal.json`; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return v3_predict.write_sidecar(out_dir, REVERSAL_NAME, dict(report))


# --- the annotated case table ----------------------------------------------

#: The selection's own fields, which the annotation copies verbatim and
#: `assert_selection_unchanged` compares. `case_id` is the key; the other
#: three are what a reader needs to use it without opening both files.
SELECTION_FIELDS: tuple[str, ...] = ("case_id", "row_index", "stratum", "criterion")


def load_cases(path: str | Path = DEFAULT_CASES_JSON) -> dict[str, Any]:
    """The frozen case table, with its own checksum verified when present."""
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"no case table at {path.name}: the annotation describes a "
            f"selection that already exists and never makes one"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("analysis") != "case_selection" or not payload.get("cases"):
        raise ValueError(
            f"{path.name} is not a v3 case table: the annotation addresses "
            f"`cases` by `case_id` and has nothing to describe without them"
        )
    _verify_checksum(path, CASES_CHECKSUM_NAME)
    return payload


def assert_selection_unchanged(
    cases: Mapping[str, Any], annotated: Mapping[str, Any]
) -> None:
    """The annotation holds the case table's selection, in its order.

    Compared field by field rather than by count: an annotation that dropped a
    case, added one, re-ordered them or moved one to another row would read as
    a new selection made by the attribution step, which is the circularity the
    two files' dependency direction exists to rule out.
    """
    expected = [
        {field: case.get(field) for field in SELECTION_FIELDS}
        for case in cases["cases"]
    ]
    written = [
        {field: case.get(field) for field in SELECTION_FIELDS}
        for case in annotated["cases"]
    ]
    if expected != written:
        raise ValueError(
            f"the annotation does not hold {CASES_NAME}'s selection: "
            f"{len(expected)} case(s) selected against {len(written)} "
            f"annotated, in the order the case table holds them. The "
            f"annotation describes a selection and never rewrites one"
        )


def _top_position(profile: np.ndarray) -> int:
    return int(np.argmax(profile))


def annotate_cases(
    *,
    cases_json: str | Path = DEFAULT_CASES_JSON,
    ism_npz: str | Path = DEFAULT_ISM_NPZ,
    ig_npz: str | Path = DEFAULT_IG_NPZ,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    predictions_csv: str | Path = DEFAULT_PREDICTIONS_CSV,
    noise_floor: str | Path = DEFAULT_NOISE_FLOOR,
    split_checksum: str | Path | None = DEFAULT_SPLIT_CHECKSUM,
    heads: Sequence[str] = ANALYSED_HEADS,
    statistic: str = DEFAULT_FLOOR_STATISTIC,
) -> dict[str, Any]:
    """`cases_annotated.json`: attribution description for a frozen selection."""
    cases = load_cases(cases_json)
    split = pd.read_csv(split_csv)
    predictions = pd.read_csv(predictions_csv)
    intersection = v3_strata.join_predictions(
        v3_strata.analysis_intersection(split), predictions
    )
    strata = v3_strata.stratify(intersection)

    profiles = v3_agreement.load_ism_profiles(ism_npz, heads)
    ig = load_ig_attribution(ig_npz, heads)
    swept = _archive_rows(profiles.row_index)
    ig_row_of = _archive_rows(ig.row_index)

    floor_path = Path(noise_floor)
    measured = read_noise_floor(floor_path)
    floors = floor_vector(measured, heads, statistic=statistic)

    # Each stratum's own PAM-share distribution, which is the denominator an
    # outlier is called against. Computed once per stratum rather than per
    # case: the percentile is a property of the population, not of the case.
    stratum_shares: dict[str, np.ndarray] = {}
    for name, frame in strata.items():
        cut = profiles.select(_positions_of(frame["row_index"], swept))
        stratum_shares[name] = v3_agreement.position_share(cut.profile, PAM_RANGE)

    joined = intersection.set_index("row_index")
    records: list[dict[str, Any]] = []
    with_ig = 0
    for case in cases["cases"]:
        row_index = int(case["row_index"])
        position = _positions_of([row_index], swept)
        cut = profiles.select(position)
        tokens = ism_token_tensor(cut.delta, MAGNITUDE)[0]
        share = v3_agreement.position_share(cut.profile, PAM_RANGE)[0]
        stratum = case.get("stratum")
        population = stratum_shares.get(stratum)

        record: dict[str, Any] = {field: case.get(field) for field in SELECTION_FIELDS}
        record["reporting_qualifier"] = case.get("reporting_qualifier", "")
        record["ism"] = {
            "token_profile": {
                head: [float(value) for value in tokens[:, index]]
                for index, head in enumerate(heads)
            },
            "top_position": {
                head: _top_position(cut.profile[0, :, index])
                for index, head in enumerate(heads)
            },
            "top_token": {
                head: TOKEN_NAMES[int(np.argmax(tokens[:, index]))]
                for index, head in enumerate(heads)
            },
            "pam_share": {
                head: _finite_or_none(share[index]) for index, head in enumerate(heads)
            },
            "pam_share_percentile_in_stratum": {
                head: _percentile_in(population, index, share[index])
                for index, head in enumerate(heads)
            },
        }

        if row_index in ig_row_of:
            with_ig += 1
            attribution = ig.select(np.array([ig_row_of[row_index]])).attribution[0]
            record["ig"] = {
                "token_attribution": {
                    head: [float(value) for value in attribution[index]]
                    for index, head in enumerate(heads)
                },
                "top_token": {
                    head: TOKEN_NAMES[int(np.argmax(np.abs(attribution[index])))]
                    for index, head in enumerate(heads)
                },
            }
        else:
            record["ig"] = None

        labels = joined.loc[[row_index], [f"label_{head}" for head in heads]].to_numpy(
            dtype=float
        )
        predicted = joined.loc[
            [row_index], [f"pred_{head}" for head in heads]
        ].to_numpy(dtype=float)
        inverted = inverted_pairs(labels, predicted)[0]
        gaps = pair_gaps(labels)[0]
        reversal = bool(reversal_rows(labels, predicted)[0])
        record["head_order"] = {
            "tau": float(kendall_tau(labels, predicted)[0]),
            "reversal": reversal,
            "tied_pair": bool(tied_rows(labels, predicted)[0]),
            # The population report's definition, kept identical: a row
            # clears only if it is a reversal *and* every inverted pair is
            # separated by more than its floor. A row with one discordant
            # pair out of three is not a reversal, and reporting it as
            # clearing would read as a misordering that beat the noise.
            "clears_noise_floor": reversal
            and bool(clears_noise_floor(gaps[None, :], inverted[None, :], floors)[0]),
            "clears_noise_floor_rule": PAIR_FLOOR_RULE,
            "inverted_pairs": [
                {
                    "pair": pair_key(left, right),
                    "label_gap": float(gaps[index]),
                    "floor": float(floors[index]),
                    "clears_floor": bool(gaps[index] > floors[index]),
                }
                for index, (left, right) in enumerate(head_pairs(heads))
                if inverted[index]
            ],
        }
        record["attribution_outlier"] = {
            head: _is_outlier(record["ism"]["pam_share_percentile_in_stratum"][head])
            for head in heads
        }
        records.append(record)

    annotated = {
        "model": "9-Hydra v3",
        "analysis": "case_attribution_annotation",
        "inputs": ["ism", "ig", "predictions", "labels", "noise_floor"],
        "points_at": {
            "file": Path(cases_json).name,
            "key": "case_id",
            "sha256": v3_predict.sha256_of(cases_json),
        },
        "selection_authority": CASES_NAME,
        "rewrites_selection": False,
        "non_circularity": NON_CIRCULARITY_NOTE,
        "heads": list(heads),
        "token_names": list(TOKEN_NAMES),
        "profile_definition": v3_agreement.PROFILE_DEFINITION,
        "pam_share_definition": PAM_SHARE_DEFINITION,
        "attribution_outlier": {
            "definition": OUTLIER_DEFINITION,
            "threshold_percentile": OUTLIER_PERCENTILE,
            "use": OUTLIER_USE,
        },
        "ig_coverage": {
            "cases": len(records),
            "cases_with_ig": with_ig,
            "note": (
                "IG ran on the frozen 400-row subset, drawn under its own "
                "seed before the cases were selected, so a case carries IG "
                "only where the two lists happen to meet. A case without it "
                "is recorded as null rather than left out, and no case was "
                "chosen or dropped for being in the subset."
            ),
        },
        "head_order_note": (
            "The per-case ordering is description, not selection: "
            f"{REVERSAL_NAME} reports the population figures, and reversal is "
            "not a selection criterion (#129 AC9)."
        ),
        "rollout_excluded": ROLLOUT_EXCLUDED,
        "claim_limit": CLAIM_LIMIT,
        "cases": records,
        "versions": v3_predict.library_versions(),
        "consumed": {
            "cases": {
                "name": Path(cases_json).name,
                "sha256": v3_predict.sha256_of(cases_json),
            },
            "ism": {
                "name": Path(ism_npz).name,
                "sha256": v3_predict.sha256_of(ism_npz),
            },
            "ig": {"name": Path(ig_npz).name, "sha256": v3_predict.sha256_of(ig_npz)},
            "noise_floor": {
                "name": floor_path.name,
                "sha256": v3_predict.sha256_of(floor_path),
            },
            **v3_selection.consumed_inputs(
                split_csv, predictions_csv, split_checksum=split_checksum
            ),
        },
    }
    assert_selection_unchanged(cases, annotated)
    return annotated


def _percentile_in(
    population: np.ndarray | None, head_index: int, value: float
) -> float | None:
    """Where `value` sits in its stratum's own distribution, as a percentile."""
    if population is None or not np.isfinite(value):
        return None
    column = population[:, head_index]
    column = column[np.isfinite(column)]
    if column.size == 0:
        return None
    return float(100.0 * (column <= value).mean())


def _is_outlier(percentile: float | None) -> bool:
    return bool(percentile is not None and percentile >= OUTLIER_PERCENTILE)


def write_annotated_cases(out_dir: str | Path, annotated: Mapping[str, Any]) -> Path:
    """Write `cases_annotated.json`; return its path."""
    if annotated.get("rewrites_selection") is not False or not annotated.get(
        "points_at", {}
    ).get("sha256"):
        raise ValueError(
            "the annotation must point at the case table it describes by "
            "digest and declare that it rewrites no selection"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return v3_predict.write_sidecar(out_dir, ANNOTATED_NAME, dict(annotated))


# --- driver ----------------------------------------------------------------


def build_reports(
    *,
    ism_npz: str | Path = DEFAULT_ISM_NPZ,
    ig_npz: str | Path = DEFAULT_IG_NPZ,
    cases_json: str | Path = DEFAULT_CASES_JSON,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    predictions_csv: str | Path = DEFAULT_PREDICTIONS_CSV,
    noise_floor: str | Path = DEFAULT_NOISE_FLOOR,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
) -> dict[str, Path]:
    """Write all three artifacts; return their paths by name."""
    artifacts_root = Path(artifacts_root)
    validation = build_validation_report(
        ism_npz=ism_npz, ig_npz=ig_npz, split_csv=split_csv
    )
    reversal = build_reversal_report(
        split_csv=split_csv, predictions_csv=predictions_csv, noise_floor=noise_floor
    )
    annotated = annotate_cases(
        cases_json=cases_json,
        ism_npz=ism_npz,
        ig_npz=ig_npz,
        split_csv=split_csv,
        predictions_csv=predictions_csv,
        noise_floor=noise_floor,
    )
    return {
        VALIDATION_NAME: write_validation_report(
            artifacts_root / REPORT_DIR, validation
        ),
        REVERSAL_NAME: write_reversal_report(artifacts_root / REPORT_DIR, reversal),
        ANNOTATED_NAME: write_annotated_cases(artifacts_root / CASES_DIR, annotated),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ism-npz", default=DEFAULT_ISM_NPZ)
    parser.add_argument("--ig-npz", default=DEFAULT_IG_NPZ)
    parser.add_argument("--cases-json", default=DEFAULT_CASES_JSON)
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--predictions-csv", default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--noise-floor", default=DEFAULT_NOISE_FLOOR)
    parser.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    arguments = parser.parse_args(argv)

    written = build_reports(
        ism_npz=arguments.ism_npz,
        ig_npz=arguments.ig_npz,
        cases_json=arguments.cases_json,
        split_csv=arguments.split_csv,
        predictions_csv=arguments.predictions_csv,
        noise_floor=arguments.noise_floor,
        artifacts_root=arguments.artifacts_root,
    )
    for name, path in written.items():
        print(f"{name}: {path}")

    validation = json.loads(written[VALIDATION_NAME].read_text(encoding="utf-8"))
    for name, stratum in validation["strata"].items():
        medians = ", ".join(
            f"{head} "
            f"{stratum['by_head'][head][MAGNITUDE]['per_row'].get('median', float('nan')):.2f}"
            for head in validation["heads"]
        )
        print(f"  IG<->ISM {name}: {stratum['rows']} rows, median {medians}")

    reversal = json.loads(written[REVERSAL_NAME].read_text(encoding="utf-8"))
    for label, entry in (
        (reversal["population"]["name"], reversal["population"]),
        *((name, stratum) for name, stratum in reversal["strata"].items()),
    ):
        print(
            f"  reversal {label}: {entry['reversal_rows']} of {entry['rows']} rows, "
            f"{entry['clearing_floor']['rows']} clearing the per-head floor"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - driver
    raise SystemExit(main())
