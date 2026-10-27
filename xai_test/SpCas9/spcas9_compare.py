"""Compare 9-Hydra v1 and v2 on the rows v1 never fitted, by category (issue #49).

What v2 bought, stated over rows that give v1 no advantage it did not earn:
the **1,740** rows of the 5,186-row 9-Hydra Testset that fall in neither of
9-Hydra v1's train and val roles, reported **per category and never pooled**.

- **The basis is a row-level exclusion, not a sequence one.** v1's roles
  number the 29,303-row all-9-heads frame while the 9-Hydra split numbers the
  37,996-row any-head frame, so the two `row_index` columns address different
  rows. `frame_row_map` bridges them through the position each row holds in
  the one normalized table both frames are filtered from. Dropping v1's train
  and val rows leaves 401 rows of v1's own test role plus 1,339 the all-9-heads
  frame never held: 1,740.
- **Two strata, judged against every row v1 fitted** - train *and* val, since
  val drove its early stopping: 606 **Unseen target** (the protospacer appears
  in no fitted row) and 1,134 **PAM-variant overlap** (it does). Pooling the
  two would let the PAM-variant rows, where v1 knew the target already, be
  read as a fair comparison; `pooling` in the report says so rather than
  leaving the absence to be noticed.
- **The PAM-variant stratum says how much of itself is memorization.** 990 of
  its 1,134 rows repeat a 30-mer v1 fitted outright, which `CONTEXT.md` calls
  **Exact duplicate**, not a known target with a new PAM. The stratum is kept
  whole as the ticket defines it, with `composition` counting the two apart,
  so nobody reads all 1,134 as a generalization test.
- **A head whose coverage is too thin in a stratum is named, not scored.**
  Below `MIN_N_FOR_COMPARISON` rows the delta is noise, so the head carries
  `excluded` and the row count that excluded it instead of a figure. On the
  real data that drops `SpCas9-NG` and `VRQR variant` from the PAM-variant
  overlap stratum, where they hold about 190 rows each - not from the
  Unseen-target stratum, where they hold about 345.
- **`SpCas9` carries a paired bootstrap CI** on its Spearman and MAE deltas
  (`paired_bootstrap.paired_bootstrap_delta_ci`), so the headline difference
  is an interval rather than a bare point.

v1 is read, not re-run: its predictions come from `SpCas9/v1_audit/`, looked
up by 30-mer, which is sound because v1 is deterministic per sequence. 319
basis rows hold a 30-mer the audit never scored; `main` puts those - and only
those - through the same pinned v1 checkpoint, and `basis_checksum.json`
records how many rows came from each source. Nothing in `v1_audit/` or
`v2_eval/` is rewritten.

Per #42 and `docs/adr/0004`, a v2 figure below v1's published one is the
expected result of removing the leak. It is reported as it falls.

Metrics are on the scale both models were trained on: the published
percentage divided by 100.

Writes only the three files in `SpCas9/v1_v2_compare/`.

The rescoring path is not unit-tested; running the script is the check, as
with the two scoring steps it reads. Everything else here is pure frame
arithmetic and is.

Usage: python -m SpCas9.spcas9_compare [--n-resamples 1000] [--seed 42]
       [--v1-predictions ...] [--v2-predictions ...] [--checkpoint ...]
       [--output-dir ...]
"""

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from paired_bootstrap import (
    clustered_paired_bootstrap_delta_ci,
    paired_bootstrap_delta_ci,
)

from SpCas9.spcas9_data import (
    BASE_DIR,
    HEAD_NAMES,
    PROTOSPACER_SLICE,
    SEQUENCE_COLUMN,
    load_spcas9_table,
)
from SpCas9.spcas9_eval import OUTPUT_DIR as V2_EVAL_DIR
from SpCas9.spcas9_eval import PREDICTIONS_FILENAME as V2_PREDICTIONS_FILENAME
from SpCas9.spcas9_eval import SCALE_NOTE
from SpCas9.spcas9_metrics import head_metrics, pred_column, true_column
from SpCas9.spcas9_split import (
    PROTOSPACER_COLUMN,
    ROW_INDEX_COLUMN,
    file_sha256,
    write_json,
)
from SpCas9.spcas9_v1_audit import CHECKPOINT_PATH as V1_CHECKPOINT_PATH
from SpCas9.spcas9_v1_audit import PREDICTIONS_PATH as V1_PREDICTIONS_PATH
from SpCas9.spcas9_v1_audit import score_checkpoint as score_v1
from SpCas9.spcas9_v1_membership import (
    CATEGORY_COLUMN,
    PAM_VARIANT_OVERLAP,
    ROLE_COLUMN,
    TRAIN,
    UNSEEN_TARGET,
    VAL,
)

OUTPUT_DIR = BASE_DIR / "SpCas9" / "v1_v2_compare"
V2_PREDICTIONS_PATH = V2_EVAL_DIR / V2_PREDICTIONS_FILENAME

BASIS_FILENAME = "basis.csv"
BASIS_CHECKSUM_FILENAME = "basis_checksum.json"
RESULTS_FILENAME = "results.json"

V1, V2 = "v1", "v2"
MODELS = (V1, V2)

V1_ROW_INDEX_COLUMN = "v1_row_index"
V1_ROLE_COLUMN = "v1_role"
EXACT_DUPLICATE_COLUMN = "v1_exact_duplicate"

# What a row's `v1_row_index` reads when the all-9-heads frame never held it.
ABSENT_ROW = -1

# The role such a row is given, alongside v1's own three. It is not one of
# `FITTED_ROLES`, which is what keeps those rows in the comparison basis.
ABSENT = "absent"

# The roles v1 fitted: train took gradients, val drove the early stopping.
FITTED_ROLES = (TRAIN, VAL)

# The two strata, in the order the report reads them. `Exact duplicate` is
# not a third stratum here - a row whose 30-mer v1 fitted is counted under
# `composition` inside PAM-variant overlap (see the module docstring).
STRATA = (UNSEEN_TARGET, PAM_VARIANT_OVERLAP)

# Below this many comparable rows in a stratum, a head's v1 <-> v2 delta is
# withheld. This is a new line drawn for this comparison, not an existing
# project threshold: `leakage_check.MIN_N_FOR_SPEARMAN` is 10, and the v1
# audit's 41-row Exact duplicate group was handled by reporting error metrics
# only rather than by a constant. 300 separates the two censored heads'
# ~190-row PAM-variant coverage from every other head/stratum pair, all of
# which clear 344. A pair that close to the line should be read with the
# distinct-sequence count beside it (see `DISTINCT_NOTE`).
MIN_N_FOR_COMPARISON = 300

# The head the ticket asks for an interval on, and the metrics it covers:
# the headline correlation and the error the delta is read off.
CI_HEAD = "SpCas9"
CI_METRICS = ("spearman", "mae")

# The bootstrap's own defaults, named here so every function that threads
# them through agrees on what an unqualified run means.
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 42

DISTINCT_NOTE = (
    "Both models are a pure function of the 30-mer, and a stratum holds fewer "
    "distinct 30-mers than rows - 466 SpCas9 Unseen-target rows over 297 of "
    "them. `distinct_sequences` is reported beside every `n` because the two "
    "are not interchangeable: a row-resampling bootstrap draws a repeated "
    "30-mer as independent evidence, so `delta_ci95` is narrower than the "
    "distinct targets support, and `n` against MIN_N_FOR_COMPARISON counts "
    "rows rather than targets."
)

POOLING_NOTE = (
    "No pooled v1 <-> v2 figure is reported. The PAM-variant overlap rows gave "
    "v1 a protospacer it had already fitted, so pooling them with the "
    "Unseen-target rows would present a partially advantaged comparison as a "
    "fair one."
)

BASIS_NOTE = (
    "The 9-Hydra Testset rows falling in neither of 9-Hydra v1's train and val "
    "roles, matched row by row through the position each row holds in the "
    "normalized spreadsheet - never by joining the two frames' row_index "
    "columns, which count different frames."
)


def frame_row_map(table: pd.DataFrame) -> pd.DataFrame:
    """Row-level bridge from the any-head frame to the all-9-heads frame.

    One row per any-head row, in that frame's own order: `row_index` is the
    position the 9-Hydra split numbers it by, `v1_row_index` the position the
    v1 membership rebuild numbers it by, or `ABSENT_ROW` where the all-9-heads
    frame never held the row. Both filters reset their index over the same
    normalized table, so the table's own positions are what relates them.
    """
    measured = table[list(HEAD_NAMES)].notna()
    any_head = measured.any(axis=1).to_numpy()
    all_heads = measured.all(axis=1).to_numpy()

    v1_row = np.full(len(table), ABSENT_ROW, dtype=np.int64)
    v1_row[all_heads] = np.arange(int(all_heads.sum()))
    return pd.DataFrame(
        {
            ROW_INDEX_COLUMN: np.arange(int(any_head.sum())),
            V1_ROW_INDEX_COLUMN: v1_row[any_head],
        }
    )


def attach_v1_roles(
    test_frame: pd.DataFrame, row_map: pd.DataFrame, v1_frame: pd.DataFrame
) -> pd.DataFrame:
    """`test_frame` plus the role v1 gave each row, `ABSENT` where it held none."""
    frame = test_frame.merge(row_map, on=ROW_INDEX_COLUMN, how="left", validate="1:1")
    if frame[V1_ROW_INDEX_COLUMN].isna().any():
        raise ValueError("a test row is missing from the any-head row map")

    roles = v1_frame.set_index(ROW_INDEX_COLUMN)[ROLE_COLUMN]
    frame[V1_ROLE_COLUMN] = (
        frame[V1_ROW_INDEX_COLUMN].map(roles).fillna(ABSENT).to_numpy()
    )
    return frame


def comparison_basis(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows v1 neither trained nor validated on."""
    return frame[~frame[V1_ROLE_COLUMN].isin(FITTED_ROLES)].reset_index(drop=True)


def fitted_rows(v1_frame: pd.DataFrame) -> pd.DataFrame:
    """The v1 rows the model actually fitted - its train and val roles."""
    return v1_frame[v1_frame[ROLE_COLUMN].isin(FITTED_ROLES)]


def label_categories(basis: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    """`basis` plus its v1 category and whether v1 fitted the same 30-mer.

    The category is the protospacer-level cut `CONTEXT.md` defines, taken
    against every row v1 fitted. `EXACT_DUPLICATE_COLUMN` is the finer fact
    the stratum's `composition` is counted from: the whole 30-mer appears in a
    fitted row, which is memorization rather than a new PAM on a known target.
    """
    fitted_protospacers = set(fitted[SEQUENCE_COLUMN].str[PROTOSPACER_SLICE])
    fitted_sequences = set(fitted[SEQUENCE_COLUMN])

    labelled = basis.copy()
    labelled[CATEGORY_COLUMN] = np.where(
        labelled[PROTOSPACER_COLUMN].isin(fitted_protospacers),
        PAM_VARIANT_OVERLAP,
        UNSEEN_TARGET,
    )
    labelled[EXACT_DUPLICATE_COLUMN] = labelled[SEQUENCE_COLUMN].isin(fitted_sequences)
    return labelled


def model_pred_column(model: str, head: str) -> str:
    """`<model>_<head>_pred` - the two models' predictions, side by side."""
    return f"{model}_{pred_column(head)}"


def _tagging(model: str) -> dict[str, str]:
    """`<head>_pred` -> `<model>_<head>_pred` for every head.

    The one map the two directions below share: a scoring step writes its
    predictions under the single shared name, while the comparison holds two
    models at once and so tags each column with whose it is.
    """
    return {pred_column(head): model_pred_column(model, head) for head in HEAD_NAMES}


def as_model_predictions(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """`frame`'s predictions tagged as `model`'s."""
    return frame.rename(columns=_tagging(model))


def model_frame(paired: pd.DataFrame, model: str) -> pd.DataFrame:
    """`paired` re-keyed so the shared metric helpers read `model`'s column."""
    return paired.rename(columns={v: k for k, v in _tagging(model).items()})


def v1_predictions_by_sequence(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """v1's prediction per 30-mer, indexed by sequence.

    v1 is deterministic on its input, so a 30-mer the audit already scored
    carries v1's prediction for every other row holding that same 30-mer -
    which is how the basis reaches v1 without re-running the audit. Duplicate
    sequences collapse to their first row; a disagreement would mean the
    frames came from different checkpoints.
    """
    columns = [SEQUENCE_COLUMN, *(pred_column(head) for head in HEAD_NAMES)]
    stacked = pd.concat([frame[columns] for frame in frames], ignore_index=True)
    deduped = stacked.drop_duplicates(subset=SEQUENCE_COLUMN).set_index(SEQUENCE_COLUMN)
    return deduped.rename(columns=_tagging(V1))


def missing_v1_sequences(basis: pd.DataFrame, lookup: pd.DataFrame) -> list[str]:
    """The basis 30-mers `lookup` holds no v1 prediction for, de-duplicated."""
    unknown = basis.loc[~basis[SEQUENCE_COLUMN].isin(lookup.index), SEQUENCE_COLUMN]
    return list(dict.fromkeys(unknown))


def attach_v1_predictions(basis: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """`basis` plus v1's prediction per head, joined on the 30-mer."""
    return basis.join(lookup, on=SEQUENCE_COLUMN)


def rescored_frame(
    sequences: Sequence[str], preds: dict[str, np.ndarray]
) -> pd.DataFrame:
    """v1's fresh predictions shaped like an audit frame, for the lookup."""
    frame = pd.DataFrame({SEQUENCE_COLUMN: list(sequences)})
    for head in HEAD_NAMES:
        frame[pred_column(head)] = np.asarray(preds[head], dtype=np.float64).reshape(-1)
    return frame


def comparable_rows(frame: pd.DataFrame, head: str) -> pd.DataFrame:
    """Rows where `head` was measured and both models have a prediction.

    A head is scored only where its variant was actually measured - the same
    rule the v2 scoring step applies - and only where v1's prediction is
    known, so the two models are always read off the same rows.
    """
    present = frame[true_column(head)].notna()
    for model in MODELS:
        present &= frame[model_pred_column(model, head)].notna()
    return frame[present]


def metric_delta(v2_metrics: dict, v1_metrics: dict) -> dict:
    """v2 minus v1 on every metric both sides defined; `None` where either is."""
    return {
        name: (
            None
            if v2_metrics[name] is None or v1_metrics[name] is None
            else v2_metrics[name] - v1_metrics[name]
        )
        for name in ("mae", "mse", "spearman", "pearson")
    }


def _ci_arrays(
    rows: pd.DataFrame, head: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`head`'s labels and both models' predictions, in `preds_a = v2` order."""
    labels, v2, v1 = (
        rows[column].to_numpy(dtype=np.float64)
        for column in (
            true_column(head),
            model_pred_column(V2, head),
            model_pred_column(V1, head),
        )
    )
    return labels, v2, v1


def delta_ci(rows: pd.DataFrame, head: str, n_resamples: int, seed: int) -> dict:
    """Paired bootstrap CI on v2 minus v1, per metric, resampling rows."""
    arrays = _ci_arrays(rows, head)
    return {
        metric: list(
            paired_bootstrap_delta_ci(
                *arrays, metric, n_resamples=n_resamples, seed=seed
            )
        )
        for metric in CI_METRICS
    }


def delta_ci_by_sequence(
    rows: pd.DataFrame, head: str, n_resamples: int, seed: int
) -> dict:
    """The same CI resampling whole 30-mers, which is the independent unit.

    Both models are a pure function of the 30-mer, so replicate rows of one
    30-mer carry one prediction between them and are not independent draws.
    This is the interval to quote; `delta_ci` is kept beside it because #49
    names the row-level call, and the gap between the two is the measure of
    how much the replicates were inflating it.
    """
    arrays = _ci_arrays(rows, head)
    return {
        metric: list(
            clustered_paired_bootstrap_delta_ci(
                *arrays,
                rows[SEQUENCE_COLUMN].to_numpy(),
                metric,
                n_resamples=n_resamples,
                seed=seed,
            )
        )
        for metric in CI_METRICS
    }


def head_comparison(
    rows: pd.DataFrame,
    head: str,
    category: str,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    min_n: int = MIN_N_FOR_COMPARISON,
) -> dict:
    """`head`'s v1 and v2 metrics on one stratum, with the delta between them.

    Under `min_n` comparable rows the delta is withheld: the report carries
    the count and why, rather than a figure a reader would quote.
    """
    distinct = int(rows[SEQUENCE_COLUMN].nunique())
    if len(rows) < min_n:
        return {
            "n": len(rows),
            "distinct_sequences": distinct,
            "excluded": True,
            "reason": (
                f"{len(rows)} comparable rows in the {category} stratum, below the "
                f"{min_n}-row minimum for a v1 <-> v2 delta"
            ),
        }

    scored = {model: head_metrics(model_frame(rows, model), head) for model in MODELS}
    comparison = {
        "n": len(rows),
        "distinct_sequences": distinct,
        "excluded": False,
        V1: scored[V1],
        V2: scored[V2],
        "delta": metric_delta(scored[V2], scored[V1]),
    }
    if head == CI_HEAD:
        comparison["delta_ci95"] = delta_ci(rows, head, n_resamples, seed)
        comparison["delta_ci95_by_sequence"] = delta_ci_by_sequence(
            rows, head, n_resamples, seed
        )
    return comparison


def composition(stratum_rows: pd.DataFrame) -> dict:
    """How many of a stratum's rows repeat a 30-mer v1 fitted, and how many do not.

    The PAM-variant overlap stratum is mostly exact repeats, which is
    memorization rather than the generalization test its name suggests. The
    split is counted here so the stratum cannot be read as all one thing.
    """
    exact = int(stratum_rows[EXACT_DUPLICATE_COLUMN].sum())
    return {"exact_duplicate": exact, "pam_variant": len(stratum_rows) - exact}


def category_report(
    paired: pd.DataFrame,
    category: str,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """One stratum: its size, what it is made of, and every head's comparison."""
    stratum_rows = paired[paired[CATEGORY_COLUMN] == category]
    return {
        "rows": len(stratum_rows),
        "distinct_sequences": int(stratum_rows[SEQUENCE_COLUMN].nunique()),
        "composition": composition(stratum_rows),
        "heads": {
            head: head_comparison(
                comparable_rows(stratum_rows, head),
                head,
                category,
                n_resamples=n_resamples,
                seed=seed,
            )
            for head in HEAD_NAMES
        },
    }


def build_report(
    paired: pd.DataFrame, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict:
    """The JSON-serialisable v1 <-> v2 comparison, per stratum and per head."""
    categories = {
        category: category_report(paired, category, n_resamples=n_resamples, seed=seed)
        for category in STRATA
    }
    return {
        "models": {V1: "9-Hydra v1", V2: "9-Hydra v2"},
        "testset": "9-Hydra Testset",
        "basis": {"rows": len(paired), "definition": BASIS_NOTE},
        "scale": SCALE_NOTE,
        "min_n_for_comparison": MIN_N_FOR_COMPARISON,
        "categories": categories,
        "excluded": {
            category: [
                head for head, scored in report["heads"].items() if scored["excluded"]
            ]
            for category, report in categories.items()
        },
        "pooling": POOLING_NOTE,
        "distinct_sequences": DISTINCT_NOTE,
        "bootstrap": {
            "head": CI_HEAD,
            "metrics": list(CI_METRICS),
            "units": {
                "delta_ci95": "rows",
                "delta_ci95_by_sequence": "distinct 30-mers (the interval to quote)",
            },
            "n_resamples": n_resamples,
            "seed": seed,
            "ci": 0.95,
        },
    }


def write_basis(
    frame: pd.DataFrame,
    v1_predictions_path: Path = V1_PREDICTIONS_PATH,
    v2_predictions_path: Path = V2_PREDICTIONS_PATH,
    rescored: int = 0,
    checkpoint_path: Path = V1_CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Write the paired basis CSV and its checksum sidecar; return the record.

    The sidecar pins both prediction files the basis was built from, so a
    reader can tell which v1 and v2 scoring runs these deltas belong to, says
    how many rows needed v1's checkpoint beyond the audit's own output, and
    pins that checkpoint too when any did.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    basis_path = output_dir / BASIS_FILENAME
    # Fixed line ending so `file_sha256` matches across Windows and Colab.
    frame.to_csv(basis_path, index=False, lineterminator="\n")

    record = {
        "rows": len(frame),
        "categories": {
            category: int((frame[CATEGORY_COLUMN] == category).sum())
            for category in STRATA
        },
        "v1_predictions": {
            "from_audit": len(frame) - rescored,
            "rescored": rescored,
            "checkpoint": checkpoint_path.name if rescored else None,
            "checkpoint_sha256": file_sha256(checkpoint_path) if rescored else None,
            "note": (
                "v1 is deterministic per 30-mer, so a basis row repeating a "
                "sequence the audit scored reuses that prediction. The rest went "
                "through the same pinned v1 checkpoint; the audit's own files "
                "were not rewritten."
            ),
        },
        "sources": {
            model: {"file": path.name, "file_sha256": file_sha256(path)}
            for model, path in ((V1, v1_predictions_path), (V2, v2_predictions_path))
        },
        "file": BASIS_FILENAME,
        "file_sha256": file_sha256(basis_path),
    }
    write_json(record, output_dir / BASIS_CHECKSUM_FILENAME)
    return record


def build_basis(
    table: pd.DataFrame, v1_frame: pd.DataFrame, v2_frame: pd.DataFrame
) -> pd.DataFrame:
    """The labelled basis: v2's test rows minus everything v1 fitted."""
    with_roles = attach_v1_roles(
        as_model_predictions(v2_frame, V2), frame_row_map(table), v1_frame
    )
    return label_categories(comparison_basis(with_roles), fitted_rows(v1_frame))


def _print_summary(report: dict) -> None:
    for category, part in report["categories"].items():
        made_of = part["composition"]
        print(
            f"\n{category}: {part['rows']} rows "
            f"({made_of['exact_duplicate']} repeat a 30-mer v1 fitted, "
            f"{made_of['pam_variant']} do not)"
        )
        for head, scored in part["heads"].items():
            if scored["excluded"]:
                print(f"{head:>13}: n={scored['n']:>5} excluded - {scored['reason']}")
                continue
            delta = scored["delta"]
            line = (
                f"{head:>13}: n={scored['n']:>5} "
                f"Spearman {scored[V1]['spearman']:.3f} -> {scored[V2]['spearman']:.3f} "
                f"({delta['spearman']:+.3f}), "
                f"MAE {scored[V1]['mae']:.4f} -> {scored[V2]['mae']:.4f} "
                f"({delta['mae']:+.4f})"
            )
            if "delta_ci95_by_sequence" in scored:
                low, high = scored["delta_ci95_by_sequence"]["spearman"]
                line += f" [Spearman delta 95% CI by 30-mer {low:+.3f}, {high:+.3f}]"
            print(line)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--v1-predictions", type=Path, default=V1_PREDICTIONS_PATH)
    parser.add_argument("--v2-predictions", type=Path, default=V2_PREDICTIONS_PATH)
    parser.add_argument("--checkpoint", type=Path, default=V1_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    v1_frame = pd.read_csv(args.v1_predictions)
    v2_frame = pd.read_csv(args.v2_predictions)
    basis = build_basis(load_spcas9_table(), v1_frame, v2_frame)

    lookup = v1_predictions_by_sequence([v1_frame])
    unscored = missing_v1_sequences(basis, lookup)
    if unscored:
        print(f"Scoring v1 on {len(unscored)} 30-mers the audit never held")
        fresh = rescored_frame(unscored, score_v1(unscored, args.checkpoint))
        lookup = v1_predictions_by_sequence([v1_frame, fresh])
    paired = attach_v1_predictions(basis, lookup)
    rescored = int(paired[SEQUENCE_COLUMN].isin(unscored).sum())

    checksum = write_basis(
        paired,
        v1_predictions_path=args.v1_predictions,
        v2_predictions_path=args.v2_predictions,
        rescored=rescored,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    report = build_report(paired, n_resamples=args.n_resamples, seed=args.seed)
    report["basis"].update(checksum)
    results_path = write_json(report, args.output_dir / RESULTS_FILENAME)

    _print_summary(report)
    print(f"\n{report['pooling']}")
    print(f"Wrote {args.output_dir / BASIS_FILENAME}, {results_path}")
    return report


if __name__ == "__main__":
    main()
