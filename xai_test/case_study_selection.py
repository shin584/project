"""Case Study selection (Step 5, issue #17).

Selects the 10 representative Testset samples used for deep per-sample
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
- **Concordant**: both models predict well (`e_A <= Q25(e_A) AND
  e_B <= Q25(e_B)`). Top 5 by combined error (`e_A + e_B`, ascending) - the
  best joint accuracy first.

Total Case Study count is capped at 10 (5 Discordant-family + 5
Concordant); either group can come up short of its target if too few
samples qualify - this is not itself an error, just a smaller-than-usual
export.

Ties in every ranking break by ascending array index, for a fully
deterministic selection given the same `e_a`/`e_b` input.
"""

import numpy as np

PRIMARY_TARGET = 5
CONCORDANT_TARGET = 5
SECONDARY_RELATIVE_GAP_THRESHOLD = 0.5

PRIMARY_DISCORDANT = "Primary Discordant"
SECONDARY_DISCORDANT = "Secondary Discordant"
CONCORDANT = "Concordant"


def _rank(indices, key) -> list[int]:
    """Stable-ish ranking: descending/ascending `key`, ties broken by ascending index."""
    return sorted(indices, key=key)


def select_case_studies(e_a: np.ndarray, e_b: np.ndarray) -> list[dict]:
    """Select up to 10 Case Study indices per the documented residual-quantile rule.

    `e_a`/`e_b` are 1D arrays of per-sample absolute error for Model A / Model
    B, aligned by index (index `i` is Testset `sample_id` `i`, since
    `test_metadata.csv` assigns `sample_id` by contiguous row order).

    Returns a list of dicts (`index`, `case_type`, `case_id`), ordered
    Primary Discordant, then Secondary Discordant (if any), then Concordant -
    at most `PRIMARY_TARGET + CONCORDANT_TARGET` (10) entries total.
    """
    e_a = np.asarray(e_a, dtype=np.float64)
    e_b = np.asarray(e_b, dtype=np.float64)
    if e_a.shape != e_b.shape:
        raise ValueError(
            f"e_a and e_b must have the same shape, got {e_a.shape} vs {e_b.shape}"
        )
    if e_a.ndim != 1:
        raise ValueError(f"e_a/e_b must be 1D, got ndim={e_a.ndim}")

    q75_a = np.quantile(e_a, 0.75)
    q25_a = np.quantile(e_a, 0.25)
    q25_b = np.quantile(e_b, 0.25)
    median_a = np.median(e_a)

    # --- Primary Discordant ---
    primary_mask = (e_a >= q75_a) & (e_b <= q25_b)
    primary_candidates = np.flatnonzero(primary_mask)
    primary_gap = e_a - e_b
    primary_selected = _rank(
        primary_candidates.tolist(), key=lambda i: (-primary_gap[i], i)
    )[:PRIMARY_TARGET]

    # --- Secondary Discordant (fallback, only fills what Primary left short) ---
    secondary_selected: list[int] = []
    remaining = PRIMARY_TARGET - len(primary_selected)
    if remaining > 0:
        primary_selected_set = set(primary_selected)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_gap = (e_a - e_b) / e_a
        secondary_mask = (relative_gap >= SECONDARY_RELATIVE_GAP_THRESHOLD) & (
            e_a >= median_a
        )
        secondary_candidates = [
            i for i in np.flatnonzero(secondary_mask) if i not in primary_selected_set
        ]
        secondary_selected = _rank(
            secondary_candidates, key=lambda i: (-relative_gap[i], i)
        )[:remaining]

    # --- Concordant ---
    concordant_mask = (e_a <= q25_a) & (e_b <= q25_b)
    concordant_candidates = np.flatnonzero(concordant_mask)
    combined_error = e_a + e_b
    concordant_selected = _rank(
        concordant_candidates.tolist(), key=lambda i: (combined_error[i], i)
    )[:CONCORDANT_TARGET]

    entries: list[dict] = []
    for rank, idx in enumerate(primary_selected, start=1):
        entries.append(
            {
                "index": int(idx),
                "case_type": PRIMARY_DISCORDANT,
                "case_id": f"DISCORDANT_P{rank:02d}",
            }
        )
    for rank, idx in enumerate(secondary_selected, start=1):
        entries.append(
            {
                "index": int(idx),
                "case_type": SECONDARY_DISCORDANT,
                "case_id": f"DISCORDANT_S{rank:02d}",
            }
        )
    for rank, idx in enumerate(concordant_selected, start=1):
        entries.append(
            {
                "index": int(idx),
                "case_type": CONCORDANT,
                "case_id": f"CONCORDANT_C{rank:02d}",
            }
        )
    return entries
