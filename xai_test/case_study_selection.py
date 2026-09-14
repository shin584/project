"""Case Study selection (Step 5, issue #17; Reverse Discordant, issue #18).

Selects the 15 representative Testset samples used for deep per-sample
explainability bundling, following the residual-quantile rule documented in
CONTEXT.md / plan_realize.md Step 5:

- **Primary Discordant** (preferred): Model A predicts poorly, Model B
  predicts well (`e_A >= Q75(e_A) AND e_B <= Q25(e_B)`). Top 5 by
  discordance gap (`e_A - e_B`, descending) - the largest "A got this wrong,
  B got this right" gap first.
- **Secondary Discordant** (fallback only): used to fill remaining
  Discordant-family slots *only* when Primary yields fewer than 5
  (`(e_A - e_B) / e_A >= 0.5 AND e_A >= Median(e_A)`), ranked by relative
  gap descending. Never overlaps with the Primary selection.
- **Reverse Primary Discordant** (issue #18; preferred): the mirror image -
  Model B predicts poorly, Model A predicts well (`e_B >= Q75(e_B) AND
  e_A <= Q25(e_A)`). Top 5 by reverse gap (`e_B - e_A`, descending) - the
  largest "B regressed relative to A" gap first. This is a separate 5-slot
  budget from the forward Discordant family (not a subset of it), since the
  reverse population is comparably sized to the forward one (see issue #18)
  and would starve one direction if the two shared a single 5-slot budget.
- **Reverse Secondary Discordant** (fallback only): fills remaining
  Reverse-family slots *only* when Reverse Primary yields fewer than 5
  (`(e_B - e_A) / e_B >= 0.5 AND e_B >= Median(e_B)`), ranked by relative
  gap descending. Never overlaps with the Reverse Primary selection, nor
  with anything already selected by the forward Discordant family.
- **Concordant**: both models predict well (`e_A <= Q25(e_A) AND
  e_B <= Q25(e_B)`). Top 5 by combined error (`e_A + e_B`, ascending) - the
  best joint accuracy first.

Total Case Study count is capped at 15 (5 Discordant-family + 5
Reverse-Discordant-family + 5 Concordant); any group can come up short of
its target if too few samples qualify - this is not itself an error, just a
smaller-than-usual export.

Ties in every ranking break by ascending array index, for a fully
deterministic selection given the same `e_a`/`e_b` input.
"""

import numpy as np

PRIMARY_TARGET = 5
REVERSE_TARGET = 5
CONCORDANT_TARGET = 5
SECONDARY_RELATIVE_GAP_THRESHOLD = 0.5

PRIMARY_DISCORDANT = "Primary Discordant"
SECONDARY_DISCORDANT = "Secondary Discordant"
REVERSE_PRIMARY_DISCORDANT = "Reverse Primary Discordant"
REVERSE_SECONDARY_DISCORDANT = "Reverse Secondary Discordant"
CONCORDANT = "Concordant"


def _rank(indices, key) -> list[int]:
    """Stable-ish ranking: descending/ascending `key`, ties broken by ascending index."""
    return sorted(indices, key=key)


def _select_discordant_family(
    e_hi: np.ndarray, e_lo: np.ndarray, target: int, exclude: set
) -> tuple[list[int], list[int]]:
    """Top-`target` gap-descending selection with a relative-gap fallback.

    Shared by both directions of the Discordant family (Primary/Secondary
    Discordant calls this with `e_hi=e_a, e_lo=e_b`; Reverse Primary/
    Secondary Discordant calls it with `e_hi=e_b, e_lo=e_a`):

    - Primary condition: `e_hi >= Q75(e_hi) AND e_lo <= Q25(e_lo)`, ranked by
      `(e_hi - e_lo)` descending.
    - Secondary fallback (only fills what Primary leaves short):
      `(e_hi - e_lo) / e_hi >= SECONDARY_RELATIVE_GAP_THRESHOLD AND
      e_hi >= Median(e_hi)`, ranked by relative gap descending. Never
      overlaps with this family's own Primary selection.

    `exclude` is a set of indices this family must never select (already
    claimed by another family) - applied to both the Primary and Secondary
    candidate pools.

    Returns `(primary_selected, secondary_selected)` index lists.
    """
    q75_hi = np.quantile(e_hi, 0.75)
    q25_lo = np.quantile(e_lo, 0.25)
    median_hi = np.median(e_hi)

    primary_mask = (e_hi >= q75_hi) & (e_lo <= q25_lo)
    primary_candidates = [i for i in np.flatnonzero(primary_mask) if i not in exclude]
    gap = e_hi - e_lo
    primary_selected = _rank(primary_candidates, key=lambda i: (-gap[i], i))[:target]

    secondary_selected: list[int] = []
    remaining = target - len(primary_selected)
    if remaining > 0:
        primary_selected_set = set(primary_selected)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_gap = (e_hi - e_lo) / e_hi
        secondary_mask = (relative_gap >= SECONDARY_RELATIVE_GAP_THRESHOLD) & (
            e_hi >= median_hi
        )
        secondary_candidates = [
            i
            for i in np.flatnonzero(secondary_mask)
            if i not in primary_selected_set and i not in exclude
        ]
        secondary_selected = _rank(
            secondary_candidates, key=lambda i: (-relative_gap[i], i)
        )[:remaining]

    return primary_selected, secondary_selected


def _entries_for(selected: list[int], case_type: str, id_prefix: str) -> list[dict]:
    return [
        {"index": int(idx), "case_type": case_type, "case_id": f"{id_prefix}{rank:02d}"}
        for rank, idx in enumerate(selected, start=1)
    ]


def select_case_studies(e_a: np.ndarray, e_b: np.ndarray) -> list[dict]:
    """Select up to 15 Case Study indices per the documented residual-quantile rule.

    `e_a`/`e_b` are 1D arrays of per-sample absolute error for Model A / Model
    B, aligned by index (index `i` is Testset `sample_id` `i`, since
    `test_metadata.csv` assigns `sample_id` by contiguous row order).

    Returns a list of dicts (`index`, `case_type`, `case_id`), ordered
    Primary Discordant, then Secondary Discordant (if any), then Reverse
    Primary Discordant, then Reverse Secondary Discordant (if any), then
    Concordant - at most `PRIMARY_TARGET + REVERSE_TARGET + CONCORDANT_TARGET`
    (15) entries total.
    """
    e_a = np.asarray(e_a, dtype=np.float64)
    e_b = np.asarray(e_b, dtype=np.float64)
    if e_a.shape != e_b.shape:
        raise ValueError(
            f"e_a and e_b must have the same shape, got {e_a.shape} vs {e_b.shape}"
        )
    if e_a.ndim != 1:
        raise ValueError(f"e_a/e_b must be 1D, got ndim={e_a.ndim}")

    primary_selected, secondary_selected = _select_discordant_family(
        e_a, e_b, PRIMARY_TARGET, exclude=set()
    )
    forward_selected_set = set(primary_selected) | set(secondary_selected)

    reverse_primary_selected, reverse_secondary_selected = _select_discordant_family(
        e_b, e_a, REVERSE_TARGET, exclude=forward_selected_set
    )

    # --- Concordant ---
    q25_a = np.quantile(e_a, 0.25)
    q25_b = np.quantile(e_b, 0.25)
    concordant_mask = (e_a <= q25_a) & (e_b <= q25_b)
    concordant_candidates = np.flatnonzero(concordant_mask)
    combined_error = e_a + e_b
    concordant_selected = _rank(
        concordant_candidates.tolist(), key=lambda i: (combined_error[i], i)
    )[:CONCORDANT_TARGET]

    return (
        _entries_for(primary_selected, PRIMARY_DISCORDANT, "DISCORDANT_P")
        + _entries_for(secondary_selected, SECONDARY_DISCORDANT, "DISCORDANT_S")
        + _entries_for(
            reverse_primary_selected, REVERSE_PRIMARY_DISCORDANT, "DISCORDANT_R"
        )
        + _entries_for(
            reverse_secondary_selected, REVERSE_SECONDARY_DISCORDANT, "DISCORDANT_RS"
        )
        + _entries_for(concordant_selected, CONCORDANT, "CONCORDANT_C")
    )
