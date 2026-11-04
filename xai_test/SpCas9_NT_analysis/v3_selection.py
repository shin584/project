"""Frozen row lists for the 9-Hydra v3 XAI analysis (#131, #129's AC7/AC9).

Two selections, one module, because they share the `SpCas9` error-quartile
computation and the manifest writer:

- **The IG subset** (AC7) — 200 rows per stratum, 400 in all, drawn under a
  recorded seed and written to disk with a membership digest, so every later
  IG artifact addresses the same rows rather than re-deriving a subset that
  drifts with the code that draws it.
- **The case table** (AC9) — six cells, `{GEN, LIB} x {ERRLO, ERRHI, HEADGAP}`,
  three cases each, at most 18.

Both read **predictions and labels only**. That is what lets them carry tests:
no checkpoint, no GPU, no network. It is also what keeps the case selection
non-circular — `cases.json` cannot be chosen by the attribution that the case
study then explains, because nothing here can read an attribution artifact.
`write_cases` enforces that at the artifact level: it refuses a payload whose
keys or values name one. The annotation step (#134) points back at `case_id`
and never rewrites the selection, so a reviewer can confirm non-circularity
from the dependency direction of the two files.

Storage order is **`row_index` ascending** everywhere, and selection order,
stratum and cell are recorded as *fields*. Overlaying the two axes is how a
re-drawn manifest silently reorders an array that was joined to it positionally.

Labels arrive from `v3_strata.join_predictions` already divided by 100, so
every error here is a fraction of 1 on both sides.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import v3_predict
import v3_strata
from v3_strata import (
    ANALYSED_HEADS,
    GENERALISATION,
    PAM_LIBRARY,
    PRIMARY_HEAD,
    STRATUM_NAMES,
)

SUBTREE_ROOT = Path(__file__).resolve().parent

#: The design's IG budget: 400 x 50 steps x 3 heads of forward+backward.
IG_SUBSET_ROWS_PER_STRATUM = 200
IG_SUBSET_ROWS = IG_SUBSET_ROWS_PER_STRATUM * len(STRATUM_NAMES)

#: Recorded in the manifest, so the draw is reproducible from the artifact
#: alone. It is the parent ticket's number rather than the split's seed: the
#: two are unrelated draws and a shared constant invites the reader to assume
#: they are not.
IG_SUBSET_SEED = 129

ERROR_QUARTILE_COUNT = 4

#: The cross-head gap pool is each stratum's **own** top decile. The two
#: strata differ about twofold in gap scale (generalisation median 0.075,
#: library 0.038), so a single absolute threshold would be p90 in one stratum
#: and p75 in the other.
HEADGAP_QUANTILE = 0.90

CASES_PER_CELL = 3

#: What each stratum's IG draw is balanced on, in **outer-to-inner order**.
#: One table, read by both the cell labeller and the manifest payload, so the
#: two cannot drift — and read as a nesting, so the library's NGG axis is
#: balanced first and the quartiles within it, rather than all eight cells
#: being evened as one flat axis. Flat evening starves the NGG axis: only 134
#: of the stratum's 1,287 rows are NGG and they cluster in two quartiles, so
#: it would return 80 NGG / 120 non-NGG while AC7 asks for *both* axes.
BALANCING: dict[str, tuple[str, ...]] = {
    GENERALISATION: ("error_quartile",),
    PAM_LIBRARY: ("is_ngg", "error_quartile"),
}

#: Strata whose case cells take at most one row per distinct protospacer. The
#: library's rows are PAM variants of five targets, so three PAM variants of
#: one target would be one case reported three times.
DISTINCT_PROTOSPACER_STRATA = frozenset({PAM_LIBRARY})

ERRLO = "ERRLO"
ERRHI = "ERRHI"
#: `HEADGAP`, not `DISAG`: the latter reads as the SaCas9 pipeline's
#: `Discordant`, which was definitionally two-model and does not carry over.
HEADGAP = "HEADGAP"
CRITERIA: tuple[str, ...] = (ERRLO, ERRHI, HEADGAP)

#: Carried by every library case record. The stratum's rows are PAM variants
#: of five targets, so a result there is "observed in one of 5 targets" and
#: reading its n as a sample size overstates the evidence.
LIBRARY_QUALIFIER = "observed in one of 5 targets"

IG_SUBSET_DIR = "v3_ig_subset"
IG_SUBSET_NAME = "ig_subset.json"
IG_SUBSET_CHECKSUM_NAME = "ig_subset_checksum.json"

CASES_DIR = "v3_cases"
CASES_NAME = "cases.json"
CASES_CHECKSUM_NAME = "cases_checksum.json"

#: Words whose presence in `cases.json` would mean the selection had seen an
#: attribution artifact. Matched on word boundaries over keys and string
#: values, so "mechanism" is not an `ism` reference.
#: `ig` alone is deliberately absent: `ig_subset.json` is a selection *input*
#: and naming it is not circularity, so banning the word would stop the
#: payload from recording what it was drawn against.
_ATTRIBUTION_WORDS = (
    "attribution",
    "attributions",
    "ism",
    "rollout",
    "npz",
    "integrated",
    "gradient",
    "gradients",
    "saliency",
    "shap",
)
_ATTRIBUTION_PATTERN = re.compile(
    r"(?<![a-z])(" + "|".join(_ATTRIBUTION_WORDS) + r")(?![a-z])"
)


#: The two inputs, and nothing else. Both are git-ignored (`*.csv`), so the
#: frozen JSON artifacts this module writes are the tracked record.
DEFAULT_SPLIT_CSV = SUBTREE_ROOT.parent / "SpCas9" / "published_split" / "test.csv"
DEFAULT_SPLIT_CHECKSUM = (
    SUBTREE_ROOT.parent / "SpCas9" / "published_split" / "split_checksum.json"
)
DEFAULT_PREDICTIONS_CSV = (
    SUBTREE_ROOT / "artifacts" / "v3_testset_predictions" / "predictions.csv"
)
DEFAULT_ARTIFACTS_ROOT = SUBTREE_ROOT / "artifacts"


def _require_known_strata(names: Sequence[str]) -> None:
    unknown = [name for name in names if name not in STRATUM_NAMES]
    if unknown:
        raise ValueError(
            f"unknown strata {unknown}; expected a subset of {list(STRATUM_NAMES)}"
        )


def stratum_prefix(stratum: str) -> str:
    """The `case_id` prefix for a stratum, derived from its own constant.

    `generalisation` -> `GEN`, `pam_library` -> `LIB`: the last underscored
    segment's first three letters. Derived rather than tabulated so a renamed
    stratum cannot keep an id prefix that no longer names it.
    """
    _require_known_strata([stratum])
    return stratum.split("_")[-1][:3].upper()


def _primary_error(frame: pd.DataFrame) -> np.ndarray:
    column = f"abs_error_{PRIMARY_HEAD}"
    if column not in frame.columns:
        raise ValueError(f"the frame is missing {column!r}; join predictions first")
    return frame[column].to_numpy(dtype=float)


def _row_index(frame: pd.DataFrame) -> np.ndarray:
    if "row_index" not in frame.columns:
        raise ValueError("the frame is missing row_index")
    return frame["row_index"].to_numpy(dtype=np.int64)


def error_quartile(frame: pd.DataFrame) -> np.ndarray:
    """Quartile 1-4 of the `SpCas9` absolute error, by rank within the frame.

    Rank-based rather than value-based so the four cells are equally sized
    whatever the error distribution looks like — a balanced draw is the point,
    and `pd.qcut` on a distribution with mass at one value collapses edges.
    Ties break on `row_index` ascending, so the cells do not shuffle between
    runs on a frame with repeated errors.
    """
    error = _primary_error(frame)
    rows = _row_index(frame)
    order = np.lexsort((rows, error))
    position = np.empty(error.size, dtype=np.int64)
    position[order] = np.arange(error.size)
    quartile = position * ERROR_QUARTILE_COUNT // max(error.size, 1) + 1
    return np.minimum(quartile, ERROR_QUARTILE_COUNT)


def head_gap(frame: pd.DataFrame) -> np.ndarray:
    """max - min of the analysed heads' absolute errors, per row.

    The multi-head failure mode: the heads disagree about how wrong they are
    on the same sequence. It needs no second model to define, which is why it
    replaces the SaCas9 pipeline's two-model Discordant class.
    """
    columns = [f"abs_error_{head}" for head in ANALYSED_HEADS]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"the frame is missing {missing}; join predictions first")
    errors = frame.loc[:, columns].to_numpy(dtype=float)
    return errors.max(axis=1) - errors.min(axis=1)


def headgap_threshold(
    frame: pd.DataFrame, *, quantile: float = HEADGAP_QUANTILE
) -> float:
    """The stratum's own gap quantile, which is the `HEADGAP` pool's floor."""
    return float(np.quantile(head_gap(frame), quantile))


@dataclass(frozen=True)
class StratumScores:
    """One stratum's rows with everything both selections rank on.

    The quartile, the gap and the targets are each derived from the same frame
    and are each wanted by the subset draw, the case selection *and* the
    written thresholds. Computing them once and passing this around keeps the
    three from disagreeing — and they would disagree silently, since the
    quartile is rank-based and a re-derivation on a differently-indexed frame
    is not an error, just a different answer.
    """

    stratum: str
    frame: pd.DataFrame
    quartile: np.ndarray
    gap: np.ndarray
    targets: pd.Series

    @property
    def error(self) -> np.ndarray:
        return _primary_error(self.frame)

    @property
    def row_index(self) -> np.ndarray:
        return _row_index(self.frame)

    @property
    def is_ngg(self) -> np.ndarray:
        return v3_strata.is_ngg(self.frame["sequence"])

    @property
    def headgap_floor(self) -> float:
        """The stratum's own top-decile floor: the `HEADGAP` pool's edge."""
        return float(np.quantile(self.gap, HEADGAP_QUANTILE))

    def cells(self) -> np.ndarray:
        """The balancing cell each row falls in, as a label per row.

        The axes come from `BALANCING`, so the generalisation stratum labels
        on the error quartile alone — it is already single-valued on NGG —
        and the library stratum prefixes the NGG axis it also balances on.
        """
        labels = np.array([f"errq{q}" for q in self.quartile], dtype=object)
        if "is_ngg" not in BALANCING[self.stratum]:
            return labels
        return np.array(
            [
                f"{'ngg' if ngg else 'nonngg'}-{label}"
                for ngg, label in zip(self.is_ngg, labels, strict=True)
            ],
            dtype=object,
        )

    def cell_groups(self) -> dict[str, list[str]]:
        """Cells grouped by their outer balancing axis, outer key sorted.

        One group for a single-axis stratum, and one group per NGG value for
        the library — which is what lets the draw balance the outer axis
        before evening the quartiles inside it.
        """
        groups: dict[str, list[str]] = {}
        for cell in sorted(set(self.cells())):
            outer = cell.split("-")[0] if len(BALANCING[self.stratum]) > 1 else ""
            groups.setdefault(outer, []).append(cell)
        return groups


def score_stratum(stratum: str, frame: pd.DataFrame) -> StratumScores:
    """Compute a stratum's ranking inputs once."""
    _require_known_strata([stratum])
    frame = frame.reset_index(drop=True)
    return StratumScores(
        stratum=stratum,
        frame=frame,
        quartile=error_quartile(frame),
        gap=head_gap(frame),
        targets=v3_strata.protospacers(frame["sequence"]),
    )


def ig_subset_cells(frame: pd.DataFrame, stratum: str) -> np.ndarray:
    """The balancing cell each row falls in. See `StratumScores.cells`."""
    return score_stratum(stratum, frame).cells()


def allocate_cell_draws(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Spread `total` draws over cells as evenly as their capacity allows.

    One row at a time to the emptiest cell that still has capacity, ties by
    cell name. That makes the result even when every cell is large, and
    redistributes a short cell's shortfall over the others rather than coming
    back under budget — a 200-row subset that silently became 183 because one
    NGG quartile was thin is the failure this avoids.
    """
    allocation = {cell: 0 for cell in sorted(sizes)}
    capacity = sum(sizes.values())
    if capacity < total:
        raise ValueError(f"the cells hold only {capacity} rows, short of {total}")
    for _ in range(total):
        pick = min(
            (cell for cell in allocation if allocation[cell] < sizes[cell]),
            key=lambda cell: (allocation[cell], cell),
        )
        allocation[pick] += 1
    return allocation


def allocate_nested(
    groups: Mapping[str, Mapping[str, int]], total: int
) -> dict[str, int]:
    """Allocate over the outer axis first, then within each group.

    AC7 asks the library stratum to be balanced on NGG / non-NGG **and** the
    error quartile. Evening all eight cells as one flat axis does not do that:
    the stratum's NGG rows cluster, so the flat draw comes back 80/120 on the
    axis the stratum exists to contrast. Allocating the outer axis first gives
    100/100 and then evens the quartiles within each half as far as capacity
    allows, which is both ACs rather than the inner one only.
    """
    outer_sizes = {outer: sum(cells.values()) for outer, cells in groups.items()}
    outer_allocation = allocate_cell_draws(outer_sizes, total)
    allocation: dict[str, int] = {}
    for outer, cells in groups.items():
        allocation.update(allocate_cell_draws(cells, outer_allocation[outer]))
    return allocation


@dataclass(frozen=True)
class SubsetManifest:
    """The frozen IG membership list: rows, their cells, and how they were drawn.

    `entries` is stored `row_index` ascending; `selection_order` records the
    draw sequence as a field so the two axes never overlap. `digest` is the
    same order-sensitive population digest an IG sidecar carries, so "does
    this artifact address the manifest?" is a string comparison.
    """

    entries: pd.DataFrame
    seed: int
    rows_per_stratum: int

    @property
    def digest(self) -> str:
        return v3_predict.population_digest(self.entries["row_index"])

    def rows_for(self, stratum: str) -> np.ndarray:
        """One stratum's rows, `row_index` ascending."""
        _require_known_strata([stratum])
        rows = self.entries.loc[self.entries["stratum"] == stratum, "row_index"]
        return np.sort(rows.to_numpy(dtype=np.int64))

    @property
    def row_index(self) -> np.ndarray:
        return self.entries["row_index"].to_numpy(dtype=np.int64)

    def to_payload(self) -> dict[str, Any]:
        return {
            "model": "9-Hydra v3",
            "analysis": "ig_subset",
            "seed": int(self.seed),
            "rows": len(self.entries),
            "rows_per_stratum": int(self.rows_per_stratum),
            "membership_digest": self.digest,
            "primary_head": PRIMARY_HEAD,
            "heads": list(ANALYSED_HEADS),
            "storage_order": "row_index ascending",
            "storage_order_note": (
                "Selection order, stratum and cell are recorded per entry. "
                "They are not the storage axis: overlaying the two is how a "
                "re-drawn manifest silently reorders a positionally joined "
                "array."
            ),
            "balancing": {name: list(axes) for name, axes in BALANCING.items()},
            "balancing_note": (
                "Outer axis first, then the axes inside it. The library "
                "stratum's NGG rows cluster, so evening all eight cells as "
                "one flat axis would return 80 NGG / 120 non-NGG."
            ),
            "label_scale": "fraction",
            "prediction_scale": v3_predict.PREDICTION_SCALE,
            "versions": v3_predict.library_versions(),
            "entries": json.loads(
                self.entries.to_json(orient="records", double_precision=15)
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SubsetManifest:
        entries = pd.DataFrame(payload["entries"])
        manifest = cls(
            entries=entries.sort_values("row_index").reset_index(drop=True),
            seed=int(payload["seed"]),
            rows_per_stratum=int(payload["rows_per_stratum"]),
        )
        recorded = payload.get("membership_digest")
        if recorded != manifest.digest:
            raise ValueError(
                "the manifest's membership digest does not match its entries: "
                f"recorded {recorded}, computed {manifest.digest}"
            )
        return manifest


def select_ig_subset(
    strata: Mapping[str, pd.DataFrame],
    *,
    seed: int = IG_SUBSET_SEED,
    rows_per_stratum: int = IG_SUBSET_ROWS_PER_STRATUM,
) -> SubsetManifest:
    """Draw the IG subset: `rows_per_stratum` rows from each stratum.

    The strata are drawn **independently**: each gets its own substream of the
    seed, keyed by its position in `v3_strata.STRATUM_NAMES`. Growing or
    fixing one stratum therefore leaves the other's rows alone, where a single
    shared stream would silently re-roll both.
    """
    _require_known_strata(list(strata))
    selected: list[dict[str, Any]] = []
    order = 0
    for ordinal, name in enumerate(STRATUM_NAMES):
        if name not in strata:
            continue
        scores = score_stratum(name, strata[name])
        cells = scores.cells()
        counts = pd.Series(cells).value_counts().to_dict()
        groups = {
            outer: {cell: counts[cell] for cell in group}
            for outer, group in scores.cell_groups().items()
        }
        try:
            allocation = allocate_nested(groups, rows_per_stratum)
        except ValueError as short:
            raise ValueError(f"stratum {name!r}: {short}") from short

        rng = np.random.default_rng([seed, ordinal])
        rows, quartile, ngg = scores.row_index, scores.quartile, scores.is_ngg
        for cell in sorted(allocation):
            positions = np.flatnonzero(cells == cell)
            drawn = rng.choice(positions, size=allocation[cell], replace=False)
            for position in drawn:
                order += 1
                selected.append(
                    {
                        "row_index": int(rows[position]),
                        "stratum": name,
                        "cell": cell,
                        "selection_order": order,
                        "error_quartile": int(quartile[position]),
                        "is_ngg": bool(ngg[position]),
                        "protospacer": str(scores.targets.iat[position]),
                    }
                )

    entries = pd.DataFrame(selected).sort_values("row_index").reset_index(drop=True)
    if not entries["row_index"].is_unique:
        raise ValueError("the subset drew a row twice; the strata must be disjoint")
    return SubsetManifest(entries=entries, seed=seed, rows_per_stratum=rows_per_stratum)


def write_ig_subset(
    out_dir: str | Path,
    manifest: SubsetManifest,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `ig_subset.json` and its checksum file; return the manifest's path.

    Two pins, because they answer different questions: the **membership
    digest** inside the manifest says which rows an IG artifact must address,
    and the **checksum file** says the manifest's bytes have not been edited
    since it was drawn.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_payload()
    if extra:
        payload.update(extra)
    path = v3_predict.write_sidecar(out_dir, IG_SUBSET_NAME, payload)
    v3_predict.write_sidecar(
        out_dir,
        IG_SUBSET_CHECKSUM_NAME,
        {"files": {IG_SUBSET_NAME: v3_predict.sha256_of(path)}},
    )
    return path


def load_ig_subset(path: str | Path) -> SubsetManifest:
    """Read a written manifest, verifying both pins before handing it back.

    Every IG consumer goes through here, so "all IG artifacts address the same
    rows" is a property of the one reader rather than a convention each
    caller has to remember.
    """
    path = Path(path)
    checksum_path = path.parent / IG_SUBSET_CHECKSUM_NAME
    if not checksum_path.exists():
        raise ValueError(f"{checksum_path.name} is missing; the manifest is unpinned")
    recorded = json.loads(checksum_path.read_text(encoding="utf-8"))["files"]
    actual = v3_predict.sha256_of(path)
    if recorded.get(path.name) != actual:
        raise ValueError(
            f"{path.name} does not match its recorded sha256: "
            f"recorded {recorded.get(path.name)}, computed {actual}"
        )
    return SubsetManifest.from_payload(json.loads(path.read_text(encoding="utf-8")))


def _cell_order(scores: StratumScores, criterion: str) -> np.ndarray:
    """Positions in a cell, best candidate first, ties on `row_index`.

    `ERRLO` and `ERRHI` rank on the `SpCas9` absolute error inside its lowest
    and highest quartile; `HEADGAP` ranks on the cross-head gap inside the
    stratum's own top decile. The quartile and the decile bound the **pool**,
    and the rank picks within it, so a cell that has to go deeper — for a
    distinct target, or past a row another cell already took — cannot quietly
    leave the pool.
    """
    error = scores.error
    if criterion == ERRLO:
        pool = scores.quartile == 1
        key = error
    elif criterion == ERRHI:
        pool = scores.quartile == ERROR_QUARTILE_COUNT
        key = -error
    elif criterion == HEADGAP:
        pool = scores.gap >= scores.headgap_floor
        key = -scores.gap
    else:
        raise ValueError(f"unknown criterion {criterion!r}; expected {list(CRITERIA)}")
    positions = np.flatnonzero(pool)
    rows = scores.row_index
    return positions[np.lexsort((rows[positions], key[positions]))]


def select_cases(strata: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The case table: up to `CASES_PER_CELL` rows in each of the six cells.

    Two constraints, both of which can push a cell further down its rank:

    - In the library stratum a cell takes **at most one row per distinct
      protospacer**, so its three cases are three different targets rather
      than three PAM variants of one.
    - **A row is a case once.** The criteria are taken in `CRITERIA` order, so
      `HEADGAP` yields to the two error cells. On real data the widest
      cross-head gaps *are* the worst-predicted rows — measured on the real
      split, two of three `HEADGAP` cases per stratum collided with `ERRHI` —
      and a row already reported as max-error teaches nothing about multi-head
      disagreement, which is the only thing that cell is for. The collision
      rate itself is a result, and `write_cases` records it rather than
      letting the de-duplication hide it.

    Where a cell cannot fill under either constraint it comes back short: the
    design's budget is "at most 18".
    """
    _require_known_strata(list(strata))
    records: list[dict[str, Any]] = []
    for name in STRATUM_NAMES:
        if name not in strata:
            continue
        scores = score_stratum(name, strata[name])
        frame = scores.frame
        distinct_targets = name in DISTINCT_PROTOSPACER_STRATA
        prefix = stratum_prefix(name)
        claimed: set[int] = set()

        for criterion in CRITERIA:
            used: set[str] = set()
            taken = 0
            for position in _cell_order(scores, criterion):
                if taken == CASES_PER_CELL:
                    break
                row = int(scores.row_index[position])
                target = str(scores.targets.iat[position])
                if row in claimed or (distinct_targets and target in used):
                    continue
                claimed.add(row)
                used.add(target)
                taken += 1
                record: dict[str, Any] = {
                    "case_id": f"{prefix}-{criterion}-{taken:02d}",
                    "stratum": name,
                    "criterion": criterion,
                    "selection_rank": taken,
                    "row_index": row,
                    "sequence": str(frame.at[position, "sequence"]),
                    "protospacer": target,
                    "is_ngg": bool(scores.is_ngg[position]),
                    "error_quartile": int(scores.quartile[position]),
                    "head_gap": float(scores.gap[position]),
                    "reporting_qualifier": LIBRARY_QUALIFIER
                    if distinct_targets
                    else "",
                }
                for head in ANALYSED_HEADS:
                    for field in ("label", "pred", "abs_error"):
                        column = f"{field}_{head}"
                        record[column] = float(frame.at[position, column])
                records.append(record)

    return pd.DataFrame(records).sort_values("case_id").reset_index(drop=True)


def cell_overlap(strata: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    """How many rows each cell's own extremes share with an earlier cell.

    Measured **before** de-duplication, per stratum and criterion, so the
    error/gap correlation that `select_cases` resolves is still reported. A
    `HEADGAP` count of 2 says the model's cross-head disagreement is
    concentrated in its high-error tail — a finding, not a defect.
    """
    _require_known_strata(list(strata))
    overlap: dict[str, dict[str, int]] = {}
    for name, frame in strata.items():
        scores = score_stratum(name, frame)
        claimed: set[int] = set()
        counts: dict[str, int] = {}
        for criterion in CRITERIA:
            extremes = [
                int(scores.row_index[position])
                for position in _cell_order(scores, criterion)[:CASES_PER_CELL]
            ]
            counts[criterion] = sum(row in claimed for row in extremes)
            claimed.update(extremes)
        overlap[name] = counts
    return overlap


def assert_no_attribution_reference(payload: Any, *, path: str = "") -> None:
    """Refuse a payload that names an attribution artifact, anywhere in it.

    AC9's non-circularity, enforced where the bytes are written rather than
    asked for in a comment. Selecting cases by the attribution that the case
    study then explains is circular, and the one-line regression that restores
    the circularity is an annotation step writing a rank back into the
    selection. Keys and string values are both checked, on word boundaries, so
    "mechanism" passes and `ism_delta_rank` does not.
    """
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _assert_clean_text(str(key), f"{path}.{key}" if path else str(key))
            assert_no_attribution_reference(
                value, path=f"{path}.{key}" if path else str(key)
            )
    elif isinstance(payload, (list, tuple)):
        for position, value in enumerate(payload):
            assert_no_attribution_reference(value, path=f"{path}[{position}]")
    elif isinstance(payload, str):
        _assert_clean_text(payload, path)


def _assert_clean_text(text: str, path: str) -> None:
    found = _ATTRIBUTION_PATTERN.search(text.lower())
    if found:
        raise ValueError(
            f"{path or 'the payload'} names the attribution artifact "
            f"({found.group(1)!r}): cases.json must be derivable from "
            "predictions and labels alone, or the case study is circular"
        )


def write_cases(
    out_dir: str | Path,
    cases: pd.DataFrame,
    strata: Mapping[str, pd.DataFrame],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `cases.json` and its checksum file; return the table's path.

    The payload records each stratum's own thresholds, because the `HEADGAP`
    pool is a per-stratum decile and a reader who assumed one absolute
    threshold would mis-read every library case.
    """
    _require_known_strata(list(strata))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds: dict[str, Any] = {}
    for name, frame in strata.items():
        scores = score_stratum(name, frame)
        error, quartile = scores.error, scores.quartile
        thresholds[name] = {
            "rows": len(scores.frame),
            "distinct_protospacers": int(scores.targets.nunique()),
            "error_quartile_1_max": float(error[quartile == 1].max()),
            "error_quartile_4_min": float(
                error[quartile == ERROR_QUARTILE_COUNT].min()
            ),
            "head_gap_median": float(np.median(scores.gap)),
            "head_gap_p90": scores.headgap_floor,
            "distinct_protospacer_per_cell": name in DISTINCT_PROTOSPACER_STRATA,
        }

    payload: dict[str, Any] = {
        "model": "9-Hydra v3",
        "analysis": "case_selection",
        "inputs": ["predictions", "labels"],
        "non_circularity": (
            "Selection reads predictions and labels only and references no "
            "downstream artifact. The annotation step points back here by "
            "case_id and never rewrites this selection."
        ),
        "criteria": list(CRITERIA),
        "cases_per_cell": CASES_PER_CELL,
        "cases_budget": CASES_PER_CELL * len(CRITERIA) * len(STRATUM_NAMES),
        "primary_head": PRIMARY_HEAD,
        "heads": list(ANALYSED_HEADS),
        "head_gap_definition": (
            "max - min of the analysed heads' absolute errors, pooled within "
            "the stratum's own top decile"
        ),
        "one_case_per_row": True,
        "cell_overlap_before_dedup": cell_overlap(strata),
        "cell_overlap_note": (
            "How many of each cell's own three extremes an earlier cell had "
            "already claimed, before de-duplication. Criteria are taken in "
            "order, so HEADGAP yields to the error cells: a row already "
            "reported as max-error says nothing about multi-head "
            "disagreement. A non-zero HEADGAP count is a result about the "
            "model - its cross-head disagreement sits in the high-error tail."
        ),
        "label_scale": "fraction",
        "prediction_scale": v3_predict.PREDICTION_SCALE,
        "strata": thresholds,
        "versions": v3_predict.library_versions(),
    }
    if extra:
        payload.update(extra)
    # `cases_budget` is "at most 18"; `cases` is what the cells actually
    # filled, which is the smaller number when a library cell runs out of
    # distinct targets. Both are recorded so a short cell reads as a short
    # cell rather than as a miscount.
    payload["cases"] = json.loads(
        cases.sort_values("case_id").to_json(orient="records", double_precision=15)
    )

    assert_no_attribution_reference(payload)
    path = v3_predict.write_sidecar(out_dir, CASES_NAME, payload)
    v3_predict.write_sidecar(
        out_dir,
        CASES_CHECKSUM_NAME,
        {"files": {CASES_NAME: v3_predict.sha256_of(path)}},
    )
    return path


def consumed_inputs(
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    predictions_csv: str | Path = DEFAULT_PREDICTIONS_CSV,
    *,
    split_checksum: str | Path | None = DEFAULT_SPLIT_CHECKSUM,
) -> dict[str, Any]:
    """The two inputs' digests, with the split's own pin checked.

    "Which split did this selection read?" is answered in the artifact rather
    than from memory, and a `test.csv` that does not match
    `published_split/split_checksum.json` stops the selection here instead of
    producing a frozen list against the wrong bytes.
    """
    split_csv, predictions_csv = Path(split_csv), Path(predictions_csv)
    digests = {
        "test.csv": v3_predict.sha256_of(split_csv),
        "predictions.csv": v3_predict.sha256_of(predictions_csv),
    }
    pinned = None
    if split_checksum is not None and Path(split_checksum).exists():
        pinned = json.loads(Path(split_checksum).read_text(encoding="utf-8"))["files"][
            "test.csv"
        ]
        if pinned != digests["test.csv"]:
            raise ValueError(
                "test.csv does not match split_checksum.json: pinned "
                f"{pinned}, read {digests['test.csv']}"
            )
    return {
        "split": split_csv.name,
        "predictions": predictions_csv.name,
        "digests": digests,
        # "checked", not "matched": a mismatch raises above, so the only thing
        # False can mean here is that no pin file was there to check against.
        "split_checksum_pin_checked": pinned is not None,
    }


def build_selections(
    *,
    split_csv: str | Path = DEFAULT_SPLIT_CSV,
    predictions_csv: str | Path = DEFAULT_PREDICTIONS_CSV,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
    seed: int = IG_SUBSET_SEED,
    rows_per_stratum: int = IG_SUBSET_ROWS_PER_STRATUM,
) -> dict[str, Path]:
    """Write both frozen lists from the split and the predictions export.

    One entry point for both because they share the inputs and the error
    quartile: drawing them in separate runs would let the subset and the cases
    disagree about which predictions they were cut from.
    """
    inputs = consumed_inputs(split_csv, predictions_csv)
    strata = v3_strata.build(pd.read_csv(split_csv), pd.read_csv(predictions_csv))
    artifacts_root = Path(artifacts_root)

    manifest = select_ig_subset(strata, seed=seed, rows_per_stratum=rows_per_stratum)
    subset_path = write_ig_subset(
        artifacts_root / IG_SUBSET_DIR, manifest, extra={"consumed": inputs}
    )
    cases_path = write_cases(
        artifacts_root / CASES_DIR,
        select_cases(strata),
        strata,
        extra={"consumed": inputs},
    )
    return {"ig_subset": subset_path, "cases": cases_path}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--predictions-csv", default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--seed", type=int, default=IG_SUBSET_SEED)
    arguments = parser.parse_args(argv)

    written = build_selections(
        split_csv=arguments.split_csv,
        predictions_csv=arguments.predictions_csv,
        artifacts_root=arguments.artifacts_root,
        seed=arguments.seed,
    )
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - driver
    raise SystemExit(main())
