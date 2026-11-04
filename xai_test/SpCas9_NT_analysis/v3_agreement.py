"""Cross-head attribution agreement for 9-Hydra v3 (#137, #129's AC8).

The analysis's one single-model-native axis. v3 has nine regression heads on a
shared encoder trunk, so "do two heads read the window the same way?" is a
question the model itself poses and no second model is needed to answer it.
This module answers it from `ism.npz` alone: the sweep's `(rows, 30, 4, 9)`
delta becomes a per-row **position profile** per head, and the agreement is
the **Spearman rank correlation between two heads' profiles**.

**Inputs: ISM only.** Attention rollout is deliberately not read. Rollout is a
function of the shared trunk's attention weights and has no head axis at all
(ADR 0008), so it cannot contribute to a question about two heads differing —
nine identical matrices would read as agreement where in fact the quantity has
no head to vary over.

**The denominator is written down.** Agreement is reported per head *pair* and
per *position range*, and both are in the artifact: a correlation over the
whole 30bp window and one over the PAM answer different questions, and a
reader who assumes the wrong range mis-reads every number. The pairs are the
three unordered pairs of `v3_strata.ANALYSED_HEADS`.

**Populations, kept apart.** Every cross-head figure is computed on the
3,851-row analysis intersection, because head agreement is undefined unless
the heads are compared on the same rows. A single head's own measured coverage
(`SpCas9` 5,021 / `SpCas9-NG` 3,982 / `evoCas9` 5,161) is a different number
from 3,851 and from every other head's, so coverage is recorded under its own
key with its own population name and never beside an intersection figure. The
two strata are reported separately and **never summed**: 895 + 1,287 adds a
generalisation population to a 5-target PAM experiment. Library-stratum
findings carry the "5 targets" qualifier and a per-target breakdown, because
reading 1,287 PAM variants of 5 protospacers as 1,287 independent
observations overstates the evidence by the replication factor.

**Analysis question 2 lands here.** `SpCas9` requires NGG where `SpCas9-NG`
accepts NGA, so their PAM sensitivity profiles should separate. If they do not,
that is itself the result — it says the shared trunk does not specialise its
PAM reading per head — and this module reports it as a finding rather than
raising, retrying or thresholding until separation appears.

Attribution describes the model's internal information use. Nothing here is a
statement about physical DNA/protein interaction, and no key names one.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import v3_predict
import v3_strata
from v3_strata import ANALYSED_HEADS, GENERALISATION, PAM_LIBRARY, STRATUM_NAMES

SUBTREE_ROOT = Path(__file__).resolve().parent

#: ACGT in fixed order, matching `v3_ism.BASES` and the delta's base axis.
BASES: tuple[str, ...] = ("A", "C", "G", "T")

POSITIONS = v3_predict.SEQUENCE_LENGTH

#: The reference base's slot is exactly 0, so a position's three alternatives
#: are the whole of its signal and the mean divides by 3, not 4.
ALTERNATIVES_PER_POSITION = len(BASES) - 1


def head_pairs(heads: Sequence[str] | None = None) -> tuple[tuple[str, str], ...]:
    """The unordered pairs of the heads given, in their order.

    Derived from the argument rather than read off a constant: every function
    here takes `heads`, and a pair list that ignored it would correlate a
    caller's heads against `ANALYSED_HEADS`' positions.
    """
    return tuple(itertools.combinations(ANALYSED_HEADS if heads is None else heads, 2))


#: The three unordered pairs of the analysed heads — the default case, and
#: what the report's `head_pairs` field names as the agreement's denominator.
HEAD_PAIRS: tuple[tuple[str, str], ...] = head_pairs(ANALYSED_HEADS)

#: `[start, stop)` position ranges the agreement is reported over. The window
#: answers "do the heads read the 30bp window alike?"; the PAM answers
#: analysis question 2 at the base resolution ADR 0007 assigns to ISM.
WINDOW_RANGE: tuple[int, int] = (0, POSITIONS)
PAM_RANGE: tuple[int, int] = (v3_strata.PAM.start, v3_strata.PAM.stop)
NGG_RANGE: tuple[int, int] = (
    v3_strata.NGG_POSITIONS.start,
    v3_strata.NGG_POSITIONS.stop,
)

#: `SpCas9` requires NGG and `SpCas9-NG` accepts NGA, so the two PAMs differ
#: at **position 26 alone** — both require G at 25. The 25-26 range answers
#: "how much does this head weight the NGG positions?", which is a profile
#: shape; position 26 on its own is the only place the two heads' *recognition
#: rules* disagree, so the contrast is reported there as well and that is
#: where the sharper evidence sits.
DISCRIMINATING_RANGE: tuple[int, int] = (26, 27)

POSITION_RANGES: dict[str, tuple[int, int]] = {
    "window": WINDOW_RANGE,
    "pam": PAM_RANGE,
}

#: The two heads whose PAM profiles analysis question 2 asks about.
SEPARATION_CONTRAST: tuple[str, str] = ("SpCas9", "SpCas9-NG")

#: A **reporting convention, not a significance test**: the two heads' NGG
#: sensitivity shares count as separated when their median symmetric relative
#: difference reaches 10%. It is recorded in the artifact so the verdict can be
#: re-read against a different margin instead of being taken on trust.
SEPARATION_RELATIVE_MARGIN = 0.10

#: `separated` says the two profiles differ in magnitude; it says nothing
#: about every row differing the same way. `direction_consistent` is the
#: second claim, and it is the stricter one: the contrast has to run in the
#: predicted direction on at least this share of rows. Reported apart because
#: a stratum can separate on the median while a quarter of its rows run the
#: other way, and one boolean covering both would hide that.
DIRECTION_CONSISTENT_SHARE = 0.90

PROFILE_DEFINITION = (
    "mean |ISM delta| over a position's three alternative bases, per row, "
    "position and head; the reference base's slot is a true zero and is "
    "excluded from the mean"
)

AGREEMENT_STATISTIC = "spearman_rank_correlation"

AGREEMENT_DEFINITION = (
    "Spearman rank correlation between two heads' position profiles, taken "
    "within the named position range. `per_row` correlates the two profiles "
    "of a single row and summarises the distribution over the stratum's rows; "
    "`mean_profile` correlates the stratum's two mean profiles, one number per "
    "pair. The two answer different questions and neither replaces the other: "
    "a stratum can hold rows that disagree pairwise while its means align."
)

#: A range narrower than this carries a `width_note`: over three positions a
#: rank correlation can only come back ±1, ±0.5 or 0, so the PAM range's
#: correlation is reported for completeness and is not what analysis question
#: 2 is answered from. `pam_separation` is, and it works at base resolution
#: over every row.
NARROW_RANGE_WIDTH = 5

NARROW_RANGE_NOTE = (
    "This range spans too few positions for a rank correlation to be "
    "informative — over three positions the statistic can only take ±1, ±0.5 "
    "or 0, and a tie decides it. It is reported for completeness. The PAM "
    "question is answered by `pam_separation`, which works at base resolution "
    "over every row instead of ranking three positions."
)

ROLLOUT_EXCLUDED = (
    "Attention rollout is not an input here. It is a function of the shared "
    "trunk's attention weights and carries no head axis (ADR 0008), so it "
    "cannot speak to whether two heads differ; nine identical matrices would "
    "read as agreement where the quantity has no head to vary over."
)

INTERPRETATION_LIMIT = (
    "Attribution describes the model's internal information use. No figure "
    "here is a claim about physical DNA/protein interaction, and a position "
    "the model is insensitive to may still matter biologically."
)

LIBRARY_QUALIFIER = "observed on 5 targets"

LIBRARY_QUALIFIER_NOTE = (
    "This stratum's rows are PAM variants of 5 protospacers, so its n is 5 "
    "targets rather than its row count, and the row count is never added to "
    "the other stratum's. Every figure in this entry is pooled over those "
    "variants: `rows` is the variant count, and the quantiles in `agreement` "
    "and `pam_separation` are quantiles over variants of 5 targets, not over "
    "5 independent observations. `per_target` repeats the window agreement "
    "and the PAM contrast target by target, which is the breakdown a "
    "library-stratum finding is reported with — a contrast that holds on the "
    "pool but on only two of the five targets is a different result from one "
    "that holds on all five."
)

POOLING = "forbidden"

POPULATION = "analysis_intersection"

COVERAGE_POPULATION = "own_measured_coverage"

COVERAGE_NOTE = (
    "A head's own measured coverage on the Testset, which is a different "
    "population from the analysis intersection and different for each head. "
    "It is named apart so an intersection figure is never read as a coverage "
    "figure; no coverage count equals the intersection's row count."
)

AGREEMENT_DIR = "v3_head_agreement"
AGREEMENT_NAME = "head_agreement.json"

DEFAULT_ISM_NPZ = SUBTREE_ROOT / "artifacts" / "v3_ism" / "ism.npz"
DEFAULT_SPLIT_CSV = SUBTREE_ROOT.parent / "SpCas9" / "published_split" / "test.csv"
DEFAULT_SPLIT_CHECKSUM = (
    SUBTREE_ROOT.parent / "SpCas9" / "published_split" / "split_checksum.json"
)
DEFAULT_ARTIFACTS_ROOT = SUBTREE_ROOT / "artifacts"


def pair_key(left: str, right: str) -> str:
    """The artifact's key for one head pair."""
    return f"{left}_vs_{right}"


# --- rank statistics -------------------------------------------------------
#
# Implemented here rather than looped through `scipy.stats` per row: the real
# run takes 3,851 rows x 3 pairs x 2 ranges, and a row-vectorised rank is one
# pass instead of 23,000 calls. Tests pin both against `scipy.stats`.


def rank_rows(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks along the last axis, 1-based."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("rank_rows needs a non-empty last axis")

    flat = values.reshape(-1, values.shape[-1])
    width = flat.shape[-1]
    order = np.argsort(flat, axis=-1, kind="stable")
    sorted_values = np.take_along_axis(flat, order, axis=-1)

    starts_run = np.ones_like(sorted_values, dtype=bool)
    starts_run[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group = np.cumsum(starts_run, axis=-1) - 1

    positions = np.broadcast_to(np.arange(1, width + 1, dtype=np.float64), flat.shape)
    rows = np.arange(flat.shape[0])[:, None]
    sums = np.zeros_like(flat, dtype=np.float64)
    counts = np.zeros_like(flat, dtype=np.float64)
    np.add.at(sums, (rows, group), positions)
    np.add.at(counts, (rows, group), 1.0)

    averaged = sums[rows, group] / counts[rows, group]
    ranks = np.empty_like(flat, dtype=np.float64)
    np.put_along_axis(ranks, order, averaged, axis=-1)
    return ranks.reshape(values.shape)


def pearson_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis; NaN where either is constant."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    centred_left = left - left.mean(axis=-1, keepdims=True)
    centred_right = right - right.mean(axis=-1, keepdims=True)
    numerator = (centred_left * centred_right).sum(axis=-1)
    denominator = np.sqrt(
        (centred_left**2).sum(axis=-1) * (centred_right**2).sum(axis=-1)
    )
    out = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def spearman_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Spearman correlation along the last axis.

    NaN where a profile is constant over the range: the correlation is
    undefined there, not zero, and `summarise_values` counts those rows rather
    than letting them land in the distribution as a value.
    """
    return pearson_rows(rank_rows(left), rank_rows(right))


def summarise_values(values: np.ndarray) -> dict[str, Any]:
    """A distribution's reported shape, with the undefined rows counted apart.

    There is deliberately no sum or total: the two strata are different
    populations and no figure here may be added across them.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    summary: dict[str, Any] = {
        "rows": int(finite.size),
        "undefined_rows": int(values.size - finite.size),
    }
    if finite.size == 0:
        return summary
    quantiles = np.quantile(finite, [0.1, 0.25, 0.5, 0.75, 0.9])
    summary.update(
        {
            "median": float(quantiles[2]),
            "mean": float(finite.mean()),
            "p10": float(quantiles[0]),
            "q1": float(quantiles[1]),
            "q3": float(quantiles[3]),
            "p90": float(quantiles[4]),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "share_negative": float((finite < 0).mean()),
        }
    )
    return summary


def _finite_or_none(value: float) -> float | None:
    """JSON holds no NaN, and an undefined correlation is not a zero."""
    return float(value) if np.isfinite(value) else None


def _median_or_none(values: np.ndarray) -> float | None:
    """The median of the defined values, or None when none are defined."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


# --- profiles --------------------------------------------------------------


def select_heads(
    array: np.ndarray, stored: Sequence[str], wanted: Sequence[str] = ANALYSED_HEADS
) -> np.ndarray:
    """Take the wanted heads off the last axis, in the order asked for."""
    stored = list(stored)
    missing = [head for head in wanted if head not in stored]
    if missing:
        raise ValueError(f"the archive does not store head(s) {missing}")
    return np.asarray(array)[..., [stored.index(head) for head in wanted]]


def position_profile(delta: np.ndarray) -> np.ndarray:
    """`(rows, 30, heads)` sensitivity from a `(rows, 30, 4, heads)` delta.

    The mean of `|delta|` over a position's three **alternative** bases. The
    reference base's slot is exactly 0 by construction (substituting a base
    for itself changes nothing), so dividing by 4 would scale every position by
    3/4 and report a mean over a base that was never substituted.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 4 or delta.shape[2] != len(BASES):
        raise ValueError(
            f"expected a (rows, positions, 4, heads) delta, got {delta.shape}"
        )
    return np.abs(delta).sum(axis=2) / ALTERNATIVES_PER_POSITION


def position_share(profile: np.ndarray, position_range: tuple[int, int]) -> np.ndarray:
    """`(rows, heads)`: a range's mean sensitivity over the window's.

    Scale-free on purpose. The heads' deltas live on the same fraction scale
    but in wildly different dynamic ranges — the replicate noise floor differs
    about sixfold between `SpCas9` and `evoCas9` — so comparing raw magnitudes
    across heads would report that difference in range as a difference in
    where the model reads.
    """
    start, stop = position_range
    inside = profile[:, start:stop, :].mean(axis=1)
    window = profile.mean(axis=1)
    out = np.full(inside.shape, np.nan, dtype=np.float64)
    np.divide(inside, window, out=out, where=window > 0)
    return out


def pair_agreement(
    profile: np.ndarray,
    heads: Sequence[str] = ANALYSED_HEADS,
    position_range: tuple[int, int] = WINDOW_RANGE,
) -> dict[str, np.ndarray]:
    """Per-row Spearman agreement for each head pair, within the range."""
    start, stop = position_range
    heads = list(heads)
    within = profile[:, start:stop, :]
    return {
        pair_key(left, right): spearman_rows(
            within[:, :, heads.index(left)], within[:, :, heads.index(right)]
        )
        for left, right in head_pairs(heads)
    }


def mean_profile_agreement(
    profile: np.ndarray,
    heads: Sequence[str] = ANALYSED_HEADS,
    position_range: tuple[int, int] = WINDOW_RANGE,
) -> dict[str, float | None]:
    """Spearman agreement between the heads' mean profiles over the rows."""
    start, stop = position_range
    heads = list(heads)
    mean = profile[:, start:stop, :].mean(axis=0)
    return {
        pair_key(left, right): _finite_or_none(
            spearman_rows(
                mean[None, :, heads.index(left)], mean[None, :, heads.index(right)]
            )[0]
        )
        for left, right in head_pairs(heads)
    }


def mean_position_profile(
    profile: np.ndarray, heads: Sequence[str] = ANALYSED_HEADS
) -> dict[str, list[float]]:
    """Each head's mean sensitivity per position, over the rows given."""
    mean = profile.mean(axis=0)
    return {
        head: [float(value) for value in mean[:, index]]
        for index, head in enumerate(heads)
    }


# --- analysis question 2: does the PAM reading separate per head? -----------


def pam_base_sensitivity(
    delta: np.ndarray, heads: Sequence[str] = ANALYSED_HEADS
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean `|delta|` at each PAM position and base, per head.

    The base-resolution PAM table ADR 0007 assigns to ISM rather than IG: a
    single-base substitution is indifferent to the 6-mer token boundary that
    merges the PAM with the right flank. A base's entry averages over rows
    where it is the reference as well, and there its delta is a true zero, so
    the reference base's cell reads low by construction rather than because
    the model ignores it.
    """
    magnitude = np.abs(np.asarray(delta, dtype=np.float64)).mean(axis=0)
    start, stop = PAM_RANGE
    return {
        head: {
            str(position): {
                base: float(magnitude[position, base_index, head_index])
                for base_index, base in enumerate(BASES)
            }
            for position in range(start, stop)
        }
        for head_index, head in enumerate(heads)
    }


def g_loss_share(
    delta: np.ndarray,
    profile: np.ndarray,
    heads: Sequence[str] = ANALYSED_HEADS,
) -> dict[str, np.ndarray]:
    """Per-head drop from substituting away from G at positions 25-26.

    Signed and directional, unlike `position_share`: positive means losing the
    GG lowers the prediction. Divided by the row's own window sensitivity so
    the three heads' different dynamic ranges do not decide the comparison.
    Call it on NGG rows only — elsewhere the reference base at 25-26 is not G
    and the quantity names nothing.
    """
    start, stop = NGG_RANGE
    g_index = BASES.index("G")
    alternatives = [index for index in range(len(BASES)) if index != g_index]
    loss = -np.asarray(delta, dtype=np.float64)[:, start:stop, alternatives, :].mean(
        axis=(1, 2)
    )
    window = profile.mean(axis=1)
    out = np.full(loss.shape, np.nan, dtype=np.float64)
    np.divide(loss, window, out=out, where=window > 0)
    return {head: out[:, index] for index, head in enumerate(heads)}


def _contrast(
    profile: np.ndarray, heads: Sequence[str], position_range: tuple[int, int]
) -> dict[str, Any]:
    """The paired `SEPARATION_CONTRAST` comparison over one position range.

    Extracted because the comparison is made at two ranges — the NGG
    positions and the one position where the two heads' recognition rules
    actually differ — and two copies would be free to drift apart.
    """
    left, right = SEPARATION_CONTRAST
    share = position_share(profile, position_range)
    left_share = share[:, list(heads).index(left)]
    right_share = share[:, list(heads).index(right)]

    difference = left_share - right_share
    scale = (np.abs(left_share) + np.abs(right_share)) / 2.0
    relative = np.full(difference.shape, np.nan, dtype=np.float64)
    np.divide(np.abs(difference), scale, out=relative, where=scale > 0)
    return {
        "paired_difference": summarise_values(difference),
        "paired_relative_difference": summarise_values(relative),
        "direction_share": (
            float((difference > 0).mean()) if difference.size else None
        ),
    }


def pam_separation(
    delta: np.ndarray,
    profile: np.ndarray,
    heads: Sequence[str] = ANALYSED_HEADS,
    is_ngg: np.ndarray | None = None,
) -> dict[str, Any]:
    """Whether `SpCas9` and `SpCas9-NG` read the NGG positions differently.

    `SpCas9` requires NGG where `SpCas9-NG` accepts NGA, so the two heads'
    PAM sensitivity should separate. **No separation is a result, not a bug**:
    it says the shared trunk does not specialise its PAM reading per head. So
    this returns a report either way and the verdict's margin is in it.

    Two statistics, because they answer different halves of the question:

    - `ngg_position_share` — how much of a head's window sensitivity sits at
      positions 25-26, defined on every row whatever its PAM. Its paired
      relative difference is what `separated` is read from.
    - `g_loss_share` — the signed drop from losing the GG, on NGG rows only,
      which is the direction the biology predicts for `SpCas9`.
    """
    heads = list(heads)
    left, right = SEPARATION_CONTRAST
    rows = int(profile.shape[0])
    contrast = _contrast(profile, heads, NGG_RANGE)
    relative_summary = contrast["paired_relative_difference"]
    median_relative = relative_summary.get("median")
    share = position_share(profile, NGG_RANGE)

    ngg = (
        np.ones(rows, dtype=bool) if is_ngg is None else np.asarray(is_ngg, dtype=bool)
    )
    losses = g_loss_share(delta[ngg], profile[ngg], heads) if ngg.any() else {}

    return {
        "contrast": [left, right],
        "positions": {"start": NGG_RANGE[0], "stop": NGG_RANGE[1]},
        "rows": rows,
        "ngg_rows": int(ngg.sum()),
        "statistic": "position_share_25_26, paired per row",
        "position_share_25_26": {
            head: summarise_values(share[:, index]) for index, head in enumerate(heads)
        },
        "position_share_definition": (
            "a head's mean sensitivity at positions 25-26 divided by its mean "
            "over the whole 30bp window, per row. A **profile shape**, "
            "defined on every row whatever its own PAM, and scale-free so the "
            "heads' sixfold different dynamic ranges do not decide the "
            "comparison. It is not an NGG-only quantity; `g_loss_share` is."
        ),
        "paired_difference": contrast["paired_difference"],
        "paired_relative_difference": relative_summary,
        "direction_share": contrast["direction_share"],
        "direction_share_definition": (
            f"share of rows where {left}'s sensitivity share strictly exceeds "
            f"{right}'s. 1.0 is full separation in the direction the PAM "
            f"requirement predicts; 0.0 means it never exceeds it, which "
            f"includes the heads reading the positions identically. "
            f"`paired_difference.share_negative` is the share running the "
            f"other way."
        ),
        "relative_margin": SEPARATION_RELATIVE_MARGIN,
        "separated": bool(
            median_relative is not None
            and median_relative >= SEPARATION_RELATIVE_MARGIN
        ),
        "separated_definition": (
            "the median paired symmetric relative difference of the two "
            "heads' 25-26 sensitivity shares reaches `relative_margin`. A "
            "reporting convention, not a significance test; the summaries are "
            "in this artifact so another margin can be applied to them. It is "
            "a claim about **magnitude only** — see `direction_consistent`."
        ),
        "direction_consistent_share": DIRECTION_CONSISTENT_SHARE,
        "direction_consistent": bool(
            contrast["direction_share"] is not None
            and contrast["direction_share"] >= DIRECTION_CONSISTENT_SHARE
        ),
        "direction_consistent_definition": (
            "the contrast runs in the predicted direction on at least "
            "`direction_consistent_share` of rows. Separate from `separated`, "
            "and the stricter claim: a stratum can separate on the median "
            "while a sizeable minority of its rows order the other way, and "
            "one boolean covering both would hide that."
        ),
        "discriminating_position": {
            "positions": {
                "start": DISCRIMINATING_RANGE[0],
                "stop": DISCRIMINATING_RANGE[1],
            },
            "why": (
                f"{left} requires NGG and {right} accepts NGA, so the two "
                f"recognition rules differ at position 26 alone — both "
                f"require G at 25. This is the only position where the heads' "
                f"rules disagree, so the contrast is reported here as well as "
                f"over 25-26."
            ),
            "position_share": {
                head: summarise_values(
                    position_share(profile, DISCRIMINATING_RANGE)[:, index]
                )
                for index, head in enumerate(heads)
            },
            **_contrast(profile, heads, DISCRIMINATING_RANGE),
        },
        "g_loss_share": {
            head: summarise_values(values) for head, values in losses.items()
        },
        "g_loss_definition": (
            "-(mean delta over the non-G substitutions at positions 25-26) "
            "divided by the row's mean window sensitivity, on NGG rows only. "
            "Positive means losing the GG lowers the prediction."
        ),
        "not_a_bug": True,
        "reading": (
            "Separation would show the shared trunk specialising its PAM "
            "reading per head. No separation is equally a result: it says the "
            "shared trunk reads the PAM positions alike for both heads, and "
            "it is reported as a finding rather than treated as a bug or "
            "resolved by lowering the margin."
        ),
    }


# --- loading ---------------------------------------------------------------


@dataclass(frozen=True)
class ISMProfiles:
    """The analysed heads' slice of one ISM sweep, with its profile."""

    row_index: np.ndarray
    heads: tuple[str, ...]
    #: `(rows, 30, 4, 3)` float64 — signed, fraction scale, analysed heads.
    delta: np.ndarray
    #: `(rows, 30, 3)` float64 — `position_profile(delta)`.
    profile: np.ndarray
    #: Every head the archive holds, in axis order. Nine against three
    #: analysed, which the report records so the six unanalysed ones are not
    #: read as validated results.
    stored_heads: tuple[str, ...]

    def select(self, rows: np.ndarray) -> ISMProfiles:
        """The same sweep cut to `rows`, positionally.

        One call rather than three parallel index expressions: `delta` and
        `profile` have to be cut by the same rows in the same order, and
        slicing them separately is how a stratum's profile comes to belong to
        another stratum's rows.
        """
        rows = np.asarray(rows, dtype=np.int64)
        return ISMProfiles(
            row_index=self.row_index[rows],
            heads=self.heads,
            delta=self.delta[rows],
            profile=self.profile[rows],
            stored_heads=self.stored_heads,
        )


def load_ism_profiles(
    ism_npz: str | Path, heads: Sequence[str] = ANALYSED_HEADS
) -> ISMProfiles:
    """Read `ism.npz`, keep the analysed heads, and build their profiles.

    **The sidecar is required, and the stored head order is read from it.**
    Falling back to `v3_predict.HEADS` would restore the assumption the read
    exists to remove: a re-ordered archive would be sliced into the wrong
    heads, and the error would surface as a result rather than as a failure.
    For the same reason a sidecar that names no `heads` is refused rather than
    defaulted — `v3_ism` always writes the key, so its absence means this is
    not an artifact this module can address.
    """
    ism_npz = Path(ism_npz)
    stored = tuple(read_ism_sidecar(ism_npz)["heads"])

    with np.load(ism_npz) as archive:
        delta = np.asarray(archive["delta"], dtype=np.float64)
        row_index = np.asarray(archive["row_index"], dtype=np.int64)

    if delta.shape[-1] != len(stored):
        raise ValueError(
            f"the delta stores {delta.shape[-1]} heads but the sidecar names "
            f"{len(stored)}"
        )
    if np.any(np.diff(row_index) <= 0):
        raise ValueError("row_index must be strictly ascending")

    taken = select_heads(delta, stored, heads)
    return ISMProfiles(
        row_index=row_index,
        heads=tuple(heads),
        delta=taken,
        profile=position_profile(taken),
        stored_heads=stored,
    )


def read_ism_sidecar(ism_npz: str | Path) -> dict[str, Any]:
    """The sweep's sidecar, refused rather than defaulted when it cannot pin.

    `v3_ism.write_ism_artifact` refuses to write a sidecar that does not name
    the commit the sweep ran at, so a report computed off one that does not is
    a report whose numbers cannot be addressed to a run. Checked here, where
    the archive is opened, rather than at the write: by then the numbers are
    already in hand and the temptation is to record the gap and move on.
    """
    ism_npz = Path(ism_npz)
    sidecar = ism_npz.with_name(f"{ism_npz.stem}.sidecar.json")
    if not sidecar.exists():
        raise ValueError(
            f"{ism_npz.name} has no sidecar at {sidecar.name}: without it the "
            f"report cannot say which sweep produced its numbers"
        )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    missing = [
        key
        for key in ("heads", "code_ref", "population", "population_digest")
        if not payload.get(key)
    ]
    if missing:
        raise ValueError(
            f"{sidecar.name} records no {missing}: an ISM sweep is addressed "
            f"by its code_ref and population digest, so a report cannot be "
            f"computed off a sidecar that omits them"
        )
    return payload


def head_coverage(split: pd.DataFrame, heads: Sequence[str] = ANALYSED_HEADS) -> dict:
    """Each head's own measured coverage, under its own population name."""
    rows = {head: int(split[head].notna().sum()) for head in heads}
    return {
        "population": COVERAGE_POPULATION,
        "rows": rows,
        "note": COVERAGE_NOTE,
    }


def consumed_inputs(
    ism_npz: str | Path,
    split_csv: str | Path,
    *,
    split_checksum: str | Path | None = DEFAULT_SPLIT_CHECKSUM,
) -> dict[str, Any]:
    """What this report was computed from, by digest.

    The ISM sidecar's own pins are copied in rather than re-derived: "which
    sweep produced the numbers in this report?" has to be answerable from the
    report alone, and a sweep is addressed by its `code_ref` and population
    digest, not by a file path that can be overwritten.

    File **names** and digests, never the paths they were read from: unlike
    the `.npz` sidecars this report is tracked, and an absolute path would
    make its bytes depend on whose machine wrote it.
    """
    ism_npz, split_csv = Path(ism_npz), Path(split_csv)
    consumed: dict[str, Any] = {
        "ism": {"name": ism_npz.name, "sha256": v3_predict.sha256_of(ism_npz)},
        "split": {
            "name": split_csv.name,
            "sha256": v3_predict.sha256_of(split_csv),
        },
    }
    payload = read_ism_sidecar(ism_npz)
    consumed["ism"]["sidecar"] = {
        key: payload[key]
        for key in (
            "code_ref",
            "population",
            "population_digest",
            "delta_sha256",
            "delta_scale",
            "versions",
        )
        if key in payload
    }

    pinned = None
    if split_checksum is not None and Path(split_checksum).exists():
        pinned = json.loads(Path(split_checksum).read_text(encoding="utf-8"))["files"][
            "test.csv"
        ]
        if pinned != consumed["split"]["sha256"]:
            raise ValueError(
                "test.csv does not match split_checksum.json: pinned "
                f"{pinned}, read {consumed['split']['sha256']}"
            )
    # Named as `v3_selection.consumed_inputs` names it: a mismatch raises
    # above, so False can only mean there was no pin file to check against.
    consumed["split"]["split_checksum_pin_checked"] = pinned is not None
    return consumed


# --- the report ------------------------------------------------------------


def stratum_agreement(
    stratum: str,
    frame: pd.DataFrame,
    profiles: ISMProfiles,
    archive_row_of: Mapping[int, int],
) -> dict[str, Any]:
    """One stratum's agreement report. There is no combined counterpart.

    `archive_row_of` maps a `row_index` to its position on the ISM archive's
    axis 0. The join is by `row_index` and never by row order: the strata cut
    the reporting population out of one array over all 3,851 rows, and
    positional identity is how a stratum's profile comes to belong to another
    stratum's rows.
    """
    cut = profiles.select(
        np.array(
            [archive_row_of[int(value)] for value in frame["row_index"]],
            dtype=np.int64,
        )
    )
    profile, delta = cut.profile, cut.delta
    sequences = frame["sequence"]
    targets = v3_strata.protospacers(sequences)
    ngg = v3_strata.is_ngg(sequences)

    agreement: dict[str, Any] = {}
    for name, position_range in POSITION_RANGES.items():
        per_row = pair_agreement(profile, profiles.heads, position_range)
        start, stop = position_range
        agreement[name] = {
            "positions": {"start": start, "stop": stop},
            "width": stop - start,
            "per_row": {
                key: summarise_values(values) for key, values in per_row.items()
            },
            "mean_profile": mean_profile_agreement(
                profile, profiles.heads, position_range
            ),
        }
        if stop - start < NARROW_RANGE_WIDTH:
            agreement[name]["width_note"] = NARROW_RANGE_NOTE

    report: dict[str, Any] = {
        "rows": len(frame),
        "distinct_protospacers": int(targets.nunique()),
        "ngg_rows": int(ngg.sum()),
        "non_ngg_rows": int((~ngg).sum()),
        "agreement": agreement,
        "pam_separation": pam_separation(delta, profile, profiles.heads, ngg),
        "pam_base_sensitivity": {
            "positions": {"start": PAM_RANGE[0], "stop": PAM_RANGE[1]},
            "definition": (
                "mean |ISM delta| over the stratum's rows at each PAM "
                "position and base, per head, on the fraction scale. The "
                "base-resolution PAM table ADR 0007 assigns to ISM rather "
                "than IG, since a single-base substitution is indifferent to "
                "the 6-mer token boundary that merges the PAM with the right "
                "flank."
            ),
            "reference_slot_note": (
                "A base's cell averages over the rows where it is the "
                "reference base too, and there its delta is a **true zero** "
                "- substituting a base for itself changes nothing. So a low "
                "or zero cell at a position whose reference is that base is "
                "an artifact of the layout, not a finding that the model "
                "ignores the base: in an all-NGG population G at 25 and 26 "
                "reads exactly 0 for that reason."
            ),
            "by_head": pam_base_sensitivity(delta, profiles.heads),
        },
        "single_head_profile": {
            "population": POPULATION,
            "definition": PROFILE_DEFINITION,
            "mean_by_position": mean_position_profile(profile, profiles.heads),
        },
    }

    if stratum == PAM_LIBRARY:
        report["qualifier"] = LIBRARY_QUALIFIER
        report["qualifier_note"] = LIBRARY_QUALIFIER_NOTE
        per_target = []
        target_values = targets.to_numpy()
        # Positional, not label-based: `inside` indexes this stratum's own
        # rows, which is the axis both `cut` and `ngg` are already on.
        for target in pd.unique(target_values):
            inside = np.flatnonzero(target_values == target)
            local = cut.select(inside)
            local_profile = local.profile
            per_target.append(
                {
                    "protospacer": str(target),
                    "rows": int(inside.size),
                    "ngg_rows": int(ngg[inside].sum()),
                    "window_agreement": {
                        key: _median_or_none(values)
                        for key, values in pair_agreement(
                            local_profile, profiles.heads, WINDOW_RANGE
                        ).items()
                    },
                    # The PAM contrast is the only reason this stratum
                    # exists, so it is reported per target and not only
                    # pooled over the targets' PAM variants.
                    "pam_separation": pam_separation(
                        local.delta, local.profile, profiles.heads, ngg[inside]
                    ),
                }
            )
        report["per_target"] = sorted(
            per_target, key=lambda entry: (-entry["rows"], entry["protospacer"])
        )
    return report


def build_report(
    *,
    ism_npz: str | Path = DEFAULT_ISM_NPZ,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    split_checksum: str | Path | None = DEFAULT_SPLIT_CHECKSUM,
    heads: Sequence[str] = ANALYSED_HEADS,
) -> dict[str, Any]:
    """The cross-head agreement report, per stratum, from ISM alone."""
    split = pd.read_csv(split_csv)
    intersection = v3_strata.analysis_intersection(split)
    profiles = load_ism_profiles(ism_npz, heads)

    expected = intersection["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(expected), profiles.row_index):
        raise ValueError(
            f"the ISM artifact's {profiles.row_index.size} rows are not the "
            f"analysis intersection's {expected.size}: cross-head agreement "
            f"is undefined unless the heads are compared on the same rows"
        )
    archive_row_of = {
        int(value): index for index, value in enumerate(profiles.row_index)
    }

    strata = v3_strata.stratify(intersection)
    return {
        "model": "9-Hydra v3",
        "analysis": "cross_head_attribution_agreement",
        "inputs": ["ism"],
        "rollout_excluded": ROLLOUT_EXCLUDED,
        "population": POPULATION,
        "population_digest": v3_predict.population_digest(profiles.row_index),
        "rows": int(profiles.row_index.size),
        "heads": list(profiles.heads),
        "stored_heads": list(profiles.stored_heads),
        "stored_not_analysed": [
            head for head in profiles.stored_heads if head not in profiles.heads
        ],
        "stored_not_analysed_note": (
            "The ISM archive stores all nine heads because one forward pass "
            "returns them all. Only `heads` are analysed here, and no "
            "agreement figure, pair or verdict covers the rest."
        ),
        "head_pairs": [list(pair) for pair in head_pairs(profiles.heads)],
        "head_coverage": head_coverage(split, heads),
        "position_ranges": {
            name: {"start": start, "stop": stop}
            for name, (start, stop) in POSITION_RANGES.items()
        },
        "statistic": AGREEMENT_STATISTIC,
        "agreement_definition": AGREEMENT_DEFINITION,
        "profile_definition": PROFILE_DEFINITION,
        "single_head_profile": {
            "population": POPULATION,
            "note": (
                "ISM was swept over the analysis intersection, so a per-head "
                "profile here is an intersection figure even though a single "
                "head's absolute numbers may use its own coverage elsewhere. "
                "`head_coverage` carries those counts under their own "
                "population name."
            ),
        },
        "pooling": POOLING,
        "pooling_note": (
            "One entry per stratum and no combined entry. The two strata "
            f"answer different questions over differently-shaped populations: "
            f"{GENERALISATION} is sequence generalisation on unseen "
            f"protospacers, {PAM_LIBRARY} is a PAM experiment on 5 targets. "
            "Adding their row counts would add a sample size to a "
            "replication factor."
        ),
        "delta_scale": "fraction",
        "nan_policy": "forbidden",
        "nan_policy_note": (
            "A correlation is undefined, not zero, where a profile is "
            "constant over the range; such rows are counted as "
            "`undefined_rows` and left out of the distribution, and a "
            "single undefined number is written as null. No field holds NaN."
        ),
        "interpretation_limit": INTERPRETATION_LIMIT,
        "strata": {
            name: stratum_agreement(name, strata[name], profiles, archive_row_of)
            for name in STRATUM_NAMES
        },
        "versions": v3_predict.library_versions(),
        "consumed": consumed_inputs(ism_npz, split_csv, split_checksum=split_checksum),
    }


def write_report(out_dir: str | Path, report: Mapping[str, Any]) -> Path:
    """Write `head_agreement.json`; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return v3_predict.write_sidecar(out_dir, AGREEMENT_NAME, dict(report))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ism-npz", default=DEFAULT_ISM_NPZ)
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    arguments = parser.parse_args(argv)

    report = build_report(ism_npz=arguments.ism_npz, split_csv=arguments.split_csv)
    path = write_report(Path(arguments.artifacts_root) / AGREEMENT_DIR, report)
    print(f"head_agreement: {path}")
    for name, stratum in report["strata"].items():
        separation = stratum["pam_separation"]
        share = separation["direction_share"]
        print(
            f"  {name}: {stratum['rows']} rows, "
            f"{stratum['distinct_protospacers']} targets, "
            f"PAM separated={separation['separated']} "
            f"(direction share {'n/a' if share is None else f'{share:.3f}'})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - driver
    raise SystemExit(main())
