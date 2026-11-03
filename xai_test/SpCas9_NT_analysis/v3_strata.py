"""Analysis populations for the 9-Hydra v3 XAI analysis (#129).

Builds the two populations the analysis works on, each cut for the question it
answers, and refuses to hand back a pooled frame.

- **generalisation** — non-library rows restricted to NGG. "What does the model
  read from a protospacer it has never seen, once the PAM shortcut is gone?"
  The restriction is not cosmetic: `is_NGG` on its own reaches Spearman 0.80
  against the label, so a profile over the pooled population is dominated by
  positions 25-26 and its headline is true before the analysis starts.
- **pam_library** — library rows across *all* PAMs. "How does the model use the
  PAM?" Here the PAM contrast is the object of study, so non-NGG rows are kept
  deliberately; restricting this stratum to NGG would leave 134 rows.

Both are drawn from the **analysis intersection**: the rows where all three
analysed heads are measured, which is what makes cross-head comparison defined.

Nothing here imports torch or reads a checkpoint. Labels arrive as raw
percentages and predictions as fractions of 1 (see `v3_predict`); this module
converts the labels and never the predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

#: The three heads the analysis explains. A subset of `v3_predict.HEADS`, in
#: the same relative order.
ANALYSED_HEADS: tuple[str, ...] = ("SpCas9", "SpCas9-NG", "evoCas9")

#: The head whose error drives case selection and subset balancing.
PRIMARY_HEAD = "SpCas9"

#: 0-indexed slices into the 30bp SpCas9 target window.
LEFT_FLANK = slice(0, 4)
PROTOSPACER = slice(4, 24)
PAM = slice(24, 27)
RIGHT_FLANK = slice(27, 30)

#: NGG means G at 25 and G at 26; position 24 is the PAM's free N.
NGG_POSITIONS = slice(25, 27)

GENERALISATION = "generalisation"
PAM_LIBRARY = "pam_library"
STRATUM_NAMES: tuple[str, ...] = (GENERALISATION, PAM_LIBRARY)

#: The two strata answer different questions over differently-shaped
#: populations, so no figure is ever computed over their union. `summarise`
#: returns one entry per stratum and offers no combined total.
POOLING_FORBIDDEN = True

LABEL_DIVISOR = 100.0


def is_ngg(sequences: Sequence[str] | pd.Series) -> np.ndarray:
    """True where the PAM ends in GG."""
    series = pd.Series(list(sequences), dtype="string")
    return (series.str[NGG_POSITIONS] == "GG").to_numpy(dtype=bool)


def protospacers(sequences: Sequence[str] | pd.Series) -> pd.Series:
    """The 20nt protospacer at positions 4-23."""
    return pd.Series(list(sequences), dtype="string").str[PROTOSPACER]


def analysis_intersection(split: pd.DataFrame) -> pd.DataFrame:
    """The rows where every analysed head is measured."""
    missing = [head for head in ANALYSED_HEADS if head not in split.columns]
    if missing:
        raise ValueError(f"the split is missing head columns {missing}")
    measured = split.loc[:, list(ANALYSED_HEADS)].notna().all(axis=1)
    return split.loc[measured].reset_index(drop=True).copy()


def stratify(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a frame into the two analysis strata.

    The two do not cover the input: non-library non-NGG rows belong to neither,
    by design. Call this on the analysis intersection.
    """
    for column in ("sequence", "in_pam_library"):
        if column not in frame.columns:
            raise ValueError(f"the frame is missing {column!r}")

    library = frame["in_pam_library"].to_numpy(dtype=bool)
    ngg = is_ngg(frame["sequence"])
    return {
        GENERALISATION: frame.loc[~library & ngg].reset_index(drop=True).copy(),
        PAM_LIBRARY: frame.loc[library].reset_index(drop=True).copy(),
    }


def join_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach labels, predictions and absolute error for the analysed heads.

    Joins on `row_index`. Labels are divided by 100 so both sides are fractions;
    predictions are left untouched.
    """
    if "row_index" not in frame.columns or "row_index" not in predictions.columns:
        raise ValueError("both frames must carry row_index")

    prediction_columns = [f"pred_{head}" for head in ANALYSED_HEADS]
    missing = [c for c in prediction_columns if c not in predictions.columns]
    if missing:
        raise ValueError(f"the predictions are missing {missing}")

    unknown = set(frame["row_index"]) - set(predictions["row_index"])
    if unknown:
        raise ValueError(
            f"{len(unknown)} row_index values have no prediction "
            f"(first: {sorted(unknown)[:3]})"
        )

    joined = frame.merge(
        predictions.loc[:, ["row_index", *prediction_columns]],
        on="row_index",
        how="left",
        validate="one_to_one",
    )
    for head in ANALYSED_HEADS:
        label = joined[head].astype(float) / LABEL_DIVISOR
        joined[f"label_{head}"] = label
        joined[f"abs_error_{head}"] = (label - joined[f"pred_{head}"]).abs()
    return joined


def describe_stratum(frame: pd.DataFrame) -> dict[str, object]:
    """Counts a stratum report must carry, including its distinct-target count.

    `distinct_protospacers` is mandatory because the library stratum's rows are
    PAM variants of 5 targets: reading its n as a sample size overstates the
    evidence by the replication factor.
    """
    sequences = frame["sequence"]
    ngg = is_ngg(sequences)
    targets = protospacers(sequences)
    return {
        "rows": len(frame),
        "distinct_protospacers": int(targets.nunique()),
        "distinct_sequences": int(sequences.nunique()),
        "ngg_rows": int(ngg.sum()),
        "non_ngg_rows": int((~ngg).sum()),
        "rows_per_protospacer": targets.value_counts().to_dict(),
    }


def summarise(strata: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    """One report per stratum. There is deliberately no combined total."""
    unknown = set(strata) - set(STRATUM_NAMES)
    if unknown:
        raise ValueError(f"unknown strata {sorted(unknown)}")
    return {name: describe_stratum(frame) for name, frame in strata.items()}


def build(split: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The intersection, stratified, with labels and errors attached."""
    return {
        name: join_predictions(frame, predictions)
        for name, frame in stratify(analysis_intersection(split)).items()
    }
