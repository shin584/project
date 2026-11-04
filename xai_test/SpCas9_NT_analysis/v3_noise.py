"""Replicate noise floor for the 9-Hydra v3 XAI analysis (#135, #129's AC14).

How far apart two measurements of the *same* 30-mer are in the published
split. That number is what took head-ordering reversal off the selection
criteria — requiring every inverted pair to clear it leaves one row of 203 —
and it is the scale AC13 reports the surviving reversals against. A constant
that carries a conclusion needs a reproducible source, which is what this
module is: the figure was previously a one-off measurement quoted in
`CONTEXT.md` and the design doc.

**Labels only.** Nothing here reads a prediction, a checkpoint or an
attribution artifact. The spread of a repeated sequence's labels is a property
of the assay, not of the model, so a floor measured through the model would
confound the two.

Two things the module refuses to let a reader get wrong:

- **Scale.** Labels are raw percentages in the split; every number reported
  here is divided by 100, so it compares directly with the fraction-scale
  predictions and errors `v3_strata` hands out.
- **One constant for three heads.** The per-head floors are `SpCas9` 0.064,
  `SpCas9-NG` 0.070 and `evoCas9` 0.0099 — a 7.1x span, reported as the
  artifact's own `head_floor_ratio` rather than written into prose, which is
  how "about sixfold" (the `SpCas9`/`evoCas9` pair the design doc quotes) came
  to sit next to a measurement that says more. The pooled median is about five
  times `evoCas9`'s own floor, so using it everywhere would wave through
  `evoCas9` reversals far outside that head's noise. The artifact therefore
  always carries `by_head`, and `floor_for_head` is the only lookup — it takes
  a head and refuses `"pooled"`.

The floor rests on 79 repeated 30-mers over 379 rows, out of the 3,851-row
analysis intersection. Those counts are written into the artifact rather than
footnoted by hand, because every report that leans on the floor has to state
the limit it rests on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import v3_predict
import v3_strata
from v3_strata import ANALYSED_HEADS, LABEL_DIVISOR

SUBTREE_ROOT = Path(__file__).resolve().parent

#: A "replicate" is the same 30-mer measured again. Two is the minimum that
#: has a spread at all; groups of three and more are kept whole and contribute
#: one spread each, so a heavily repeated sequence does not outvote the rest.
REPEAT_MIN = 2

#: The upper tail the design quotes alongside the median. `numpy`'s linear
#: interpolation, named here so the artifact's p90 is reproducible from the
#: per-group spreads it also carries.
TAIL_PERCENTILE = 90

#: The key the tail is reported under, derived so that changing the percentile
#: cannot leave the artifact calling a p95 "p90".
TAIL_KEY = f"p{TAIL_PERCENTILE}"

#: The statistics a report may ask for as a floor. `mean`, `min` and `max` are
#: in the artifact for context but are not floors: a mean over a skewed
#: 79-group sample is not the number the margin test used.
FLOOR_STATISTICS: tuple[str, ...] = ("median", TAIL_KEY)

POOLED = "pooled"

#: The libraries this measurement actually runs on. `v3_predict` also reports
#: torch and transformers, which an artifact built from labels alone has no
#: business pinning: recording them would suggest a checkpoint was involved.
LIBRARIES_READ: tuple[str, ...] = ("numpy", "pandas")

#: Written at the artifacts root rather than in a per-artifact directory,
#: which is where the sibling artifacts live: #135's AC names the path
#: `artifacts/v3_noise_floor.json` outright, and AC13's consumer is written
#: against it. The checksum file sits beside it under the same stem.
NOISE_NAME = "v3_noise_floor.json"
NOISE_CHECKSUM_NAME = "v3_noise_floor_checksum.json"

DEFAULT_SPLIT_CSV = SUBTREE_ROOT.parent / "SpCas9" / "published_split" / "test.csv"
DEFAULT_SPLIT_CHECKSUM = (
    SUBTREE_ROOT.parent / "SpCas9" / "published_split" / "split_checksum.json"
)
DEFAULT_ARTIFACTS_ROOT = SUBTREE_ROOT / "artifacts"

SPREAD_DEFINITION = (
    "max - min of a repeated 30-mer's labels for one head, on the fraction "
    "scale. One spread per group per head; groups of three or more rows "
    "contribute one spread, not one per pair."
)

SINGLE_CONSTANT_NOTE = (
    "Use by_head, not pooled. The per-head floors span head_floor_ratio (see "
    "below), so the pooled median is several times evoCas9's own and would "
    "wave through evoCas9 gaps well outside that head's noise. The pooled "
    "figures are reported for continuity with the design doc, which quotes "
    "them as the single scale reversal was first tested against."
)

LIMIT_NOTE = (
    "The floor rests on the repeated 30-mers of one split, and on the subset "
    "of them that survives the analysis intersection: groups and rows against "
    "intersection_rows, and against repeated_groups_in_split and "
    "repeated_rows_in_split, which are what the uncut split holds. The cut is "
    "the sharpest edge of the limit - most repeated 30-mers are not measured "
    "on all three analysed heads - so a report that uses the floor states "
    "both pairs of counts."
)


@dataclass(frozen=True)
class NoiseFloor:
    """The per-group spreads and the summary computed from them.

    Both, because the summary's percentiles are only checkable against the
    spreads they came from, and the artifact carries the two together.
    """

    spreads: pd.DataFrame
    summary: dict[str, Any]


def _spread_columns() -> list[str]:
    return [f"spread_{head}" for head in ANALYSED_HEADS]


def repeated_sequence_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows whose 30-mer appears at least `REPEAT_MIN` times in `frame`.

    Repetition is counted over the frame handed in, so a caller that cut the
    frame first (to the analysis intersection, say) gets the groups that
    survive that cut rather than groups counted on the uncut split and then
    silently reduced to one row.
    """
    if "sequence" not in frame.columns:
        raise ValueError("the frame is missing 'sequence'")
    counts = frame["sequence"].value_counts()
    repeated = counts.index[counts >= REPEAT_MIN]
    kept = frame.loc[frame["sequence"].isin(repeated)]
    return kept.sort_values("sequence", kind="stable").reset_index(drop=True).copy()


def group_spreads(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per repeated 30-mer: its row count and its per-head spread.

    Ordered by `sequence`, so the artifact's table does not reorder with the
    input's row order. Labels are divided by 100 here and nowhere else.
    """
    missing = [head for head in ANALYSED_HEADS if head not in frame.columns]
    if missing:
        raise ValueError(f"the frame is missing head columns {missing}")

    repeated = repeated_sequence_rows(frame)
    if repeated.empty:
        return pd.DataFrame(
            {
                "sequence": pd.Series(dtype="string"),
                "rows": pd.Series(dtype="int64"),
                **{name: pd.Series(dtype="float64") for name in _spread_columns()},
            }
        )

    grouped = repeated.groupby("sequence", sort=True)
    records = {"rows": grouped.size().astype("int64")}
    for head in ANALYSED_HEADS:
        labels = grouped[head]
        records[f"spread_{head}"] = (
            labels.max().astype(float) - labels.min().astype(float)
        ) / LABEL_DIVISOR
    return pd.DataFrame(records).reset_index().rename(columns={"index": "sequence"})


def _statistics(values: np.ndarray) -> dict[str, float]:
    """The distribution figures, with no count: the callers name their own.

    A per-head entry counts groups and the pooled entry counts measurements —
    three times as many over the same groups — so a count returned from here
    would be right for one caller and misleading for the other.
    """
    return {
        "median": float(np.median(values)),
        TAIL_KEY: float(np.percentile(values, TAIL_PERCENTILE)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarise(spreads: pd.DataFrame) -> dict[str, Any]:
    """Per-head and pooled statistics over a `group_spreads` table.

    `by_head` is always present and `pooled` is never alone: the two are
    written by one function so an artifact cannot carry the pooled median
    without the per-head floors that qualify it.
    """
    if spreads.empty:
        raise ValueError(
            "no repeated 30-mers: the floor is undefined without at least one "
            f"sequence measured {REPEAT_MIN} times"
        )

    by_head = {
        head: {
            "groups": len(spreads),
            **_statistics(spreads[f"spread_{head}"].to_numpy(dtype=float)),
        }
        for head in ANALYSED_HEADS
    }
    pooled_values = np.concatenate(
        [spreads[name].to_numpy(dtype=float) for name in _spread_columns()]
    )
    pooled = {"measurements": int(pooled_values.size), **_statistics(pooled_values)}

    medians = [entry["median"] for entry in by_head.values()]
    lowest = min(medians)
    return {
        "repeat_minimum": REPEAT_MIN,
        "groups": len(spreads),
        "rows": int(spreads["rows"].sum()),
        "heads": list(ANALYSED_HEADS),
        "by_head": by_head,
        POOLED: pooled,
        # max / min of the per-head medians. The reason the artifact is
        # per-head at all, as one number a reader can check at a glance.
        "head_floor_ratio": float(max(medians) / lowest) if lowest else float("inf"),
        "label_scale": "fraction",
        "label_divisor": LABEL_DIVISOR,
        "percentile": TAIL_PERCENTILE,
    }


def measure(split: pd.DataFrame) -> NoiseFloor:
    """Measure the floor over a split's analysis intersection.

    The intersection, because a floor that mixed in rows where a head is
    unmeasured would be a different population per head, and AC13 compares
    the three heads' gaps against it on the same rows.
    """
    intersection = v3_strata.analysis_intersection(split)
    spreads = group_spreads(intersection)
    summary = summarise(spreads)
    summary["intersection_rows"] = len(intersection)
    summary["split_rows"] = len(split)

    # What the intersection cut costs the floor, measured rather than
    # asserted: most of the split's repeated 30-mers are not measured on all
    # three analysed heads, and `LIMIT_NOTE` sends a report to these counts.
    uncut = repeated_sequence_rows(split)
    summary["repeated_groups_in_split"] = int(uncut["sequence"].nunique())
    summary["repeated_rows_in_split"] = len(uncut)

    measured = repeated_sequence_rows(intersection)
    if "row_index" not in measured.columns:
        raise ValueError("the split is missing 'row_index'; the sidecar pins the rows")
    # `row_index` ascending, which is `population_digest`'s contract: the
    # digest is order-sensitive, so the rows are sorted rather than left in
    # the grouping's sequence order.
    summary["population_digest"] = v3_predict.population_digest(
        np.sort(measured["row_index"].to_numpy(dtype=np.int64))
    )
    return NoiseFloor(spreads=spreads, summary=summary)


def floor_for_head(
    summary: dict[str, Any], head: str, *, statistic: str = "median"
) -> float:
    """One head's floor. The only lookup, and it will not hand back `pooled`.

    Takes the summary mapping rather than a `NoiseFloor`, because the caller
    that matters (AC13's reporting step) reads the artifact back from JSON and
    has a payload, not an object. `"pooled"` is rejected rather than served:
    the pooled median is several times `evoCas9`'s floor, so a caller reaching
    for a single constant is about to report that head's noise as a result.
    """
    if head == POOLED:
        raise ValueError(
            "pooled is not a head's floor: it is several times evoCas9's own, "
            "so a margin test using it reports that head's measurement noise "
            f"as model behaviour. Pass one of {list(ANALYSED_HEADS)}."
        )
    if head not in summary["by_head"]:
        raise ValueError(
            f"unknown head {head!r}; expected one of {list(ANALYSED_HEADS)}"
        )
    if statistic not in FLOOR_STATISTICS:
        raise ValueError(
            f"{statistic!r} is not a floor statistic; expected one of "
            f"{list(FLOOR_STATISTICS)}"
        )
    return float(summary["by_head"][head][statistic])


def consumed_split(
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    *,
    split_checksum: str | Path | None = DEFAULT_SPLIT_CHECKSUM,
) -> dict[str, Any]:
    """The split's digest, with its own pin checked. One input, no predictions.

    A `test.csv` that does not match `published_split/split_checksum.json`
    stops the measurement here rather than publishing a floor measured on
    different bytes than the analysis it bounds.
    """
    split_csv = Path(split_csv)
    digest, pin_checked = v3_predict.verified_split_digest(split_csv, split_checksum)
    return {
        "split": split_csv.name,
        "digests": {"test.csv": digest},
        "split_checksum_pin_checked": pin_checked,
    }


def write_noise_floor(
    out_dir: str | Path,
    measured: NoiseFloor,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `v3_noise_floor.json` and its checksum file; return its path.

    The per-group spreads go in alongside the summary, so the median and the
    p90 are recomputable from the artifact without the git-ignored split.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": "9-Hydra v3",
        "analysis": "replicate_noise_floor",
        "inputs": ["labels"],
        "spread_definition": SPREAD_DEFINITION,
        "single_constant_note": SINGLE_CONSTANT_NOTE,
        "limit_note": LIMIT_NOTE,
        **measured.summary,
        # Only the libraries this measurement runs on. Pinning torch and
        # transformers in a labels-only artifact would suggest a checkpoint
        # was involved in a number that is a property of the assay.
        "versions": {
            name: version
            for name, version in v3_predict.library_versions().items()
            if name in LIBRARIES_READ
        },
    }
    if extra:
        payload.update(extra)
    payload["group_spreads"] = json.loads(
        measured.spreads.to_json(orient="records", double_precision=15)
    )

    path = v3_predict.write_sidecar(out_dir, NOISE_NAME, payload)
    v3_predict.write_sidecar(
        out_dir,
        NOISE_CHECKSUM_NAME,
        {"files": {NOISE_NAME: v3_predict.sha256_of(path)}},
    )
    return path


def build_noise_floor(
    *,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
    split_checksum: str | Path | None = DEFAULT_SPLIT_CHECKSUM,
) -> Path:
    """Measure the floor from a split CSV and write the artifact."""
    consumed = consumed_split(split_csv, split_checksum=split_checksum)
    measured = measure(pd.read_csv(split_csv))
    return write_noise_floor(artifacts_root, measured, extra={"consumed": consumed})


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    arguments = parser.parse_args(argv)

    path = build_noise_floor(
        split_csv=arguments.split_csv, artifacts_root=arguments.artifacts_root
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"noise_floor: {path}")
    print(
        f"  {payload['groups']} groups / {payload['rows']} rows "
        f"of {payload['intersection_rows']}"
    )
    for head in ANALYSED_HEADS:
        entry = payload["by_head"][head]
        print(f"  {head}: median {entry['median']:.4f} p90 {entry['p90']:.4f}")
    print(
        f"  pooled: median {payload[POOLED]['median']:.4f} "
        f"p90 {payload[POOLED]['p90']:.4f} (use by_head)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - driver
    raise SystemExit(main())
