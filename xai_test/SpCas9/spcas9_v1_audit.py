"""Score 9-Hydra v1 and report the per-category leakage audit (issue #44).

Uses the split membership rebuilt by `spcas9_v1_membership` (#43) to answer
how much of 9-Hydra v1's reported score comes from overlap with its training
rows, and which v1 numbers can be set beside Kim 2020's.

- Loads `SpCas9/Multi-Cas_1IN_9Hydra_Divide_Testset.keras` on CPU and scores
  all 29,303 all-9-heads rows once. Each output is matched to its label by
  head name, through `spcas9_metrics.predictions_by_head`, never by
  position.
- Writes the predictions to `SpCas9/v1_audit/predictions.csv`, with
  `predictions_checksum.json` beside it (seed, row counts, dataset SHA256,
  checkpoint SHA256 and the CSV's own SHA256), so the write-up can cite
  numbers without reloading TensorFlow.
- Per head, reports n / Spearman / Pearson / MAE / MSE (the shared
  `full_metrics`) for Unseen target and
  PAM-variant overlap, and n / MAE / MSE only for Exact duplicate (41 rows:
  no correlation and no test).
- Per head, compares Unseen target against PAM-variant overlap absolute
  errors with `leakage_check.compare_abs_errors` (Mann-Whitney U plus a
  bootstrap CI), Holm-Bonferroni corrected across the nine heads.
- Reports v1 on the paper's `Use = Test` rows split by v1 role
  (2,781 train / 665 val / 401 test). Only the 401 test-role rows are
  offered as comparable to Kim 2020.
- Sanity check: v1's error should sit lower on its train role than on val and
  test, and that gap should vanish when the roles are rebuilt under other
  seeds. That is the evidence the rebuilt split matches the checkpoint we
  hold. The ticket's stricter train -> val -> test ordering is reported too.

Metrics are on the scale v1 was trained on: the published percentage divided
by 100.

Read-only with respect to the checkpoint and every dataset file; writes only
the three files in `SpCas9/v1_audit/`.

Usage: python -m SpCas9.spcas9_v1_audit [--n-resamples 10000] [--seed 42]
"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from leakage_check import compare_abs_errors, group_metrics

from SpCas9.spcas9_data import (
    BASE_DIR,
    DATASET_PATH,
    HEAD_NAMES,
    SEQUENCE_COLUMN,
    USE_COLUMN,
    filter_all_heads,
    head_labels,
    is_published_test,
    library_versions,
    load_spcas9_table,
    one_hot,
)
from SpCas9.spcas9_metrics import (
    error_metrics,
    full_metrics,
    head_arrays,
    pred_column,
    predictions_by_head,
    true_column,
)
from SpCas9.spcas9_v1_membership import (
    CATEGORIES,
    CATEGORY_COLUMN,
    EXACT_DUPLICATE,
    PAM_VARIANT_OVERLAP,
    RANDOM_STATE,
    ROLE_COLUMN,
    ROLES,
    TEST,
    TEST_SIZE,
    TRAIN,
    UNSEEN_TARGET,
    VAL,
    VALIDATION_SPLIT,
    build_membership,
    rebuild_roles,
    role_labels,
    summarize,
)

CHECKPOINT_PATH = BASE_DIR / "SpCas9" / "Multi-Cas_1IN_9Hydra_Divide_Testset.keras"
OUTPUT_DIR = BASE_DIR / "SpCas9" / "v1_audit"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
CHECKSUM_PATH = OUTPUT_DIR / "predictions_checksum.json"
RESULTS_PATH = OUTPUT_DIR / "results.json"


# Exact duplicates are reported for size and error only (CONTEXT.md).
CORRELATED_CATEGORIES = (UNSEEN_TARGET, PAM_VARIANT_OVERLAP)

# Seeds other than v1's 42, for the role sanity check's null comparison.
CONTROL_SEEDS = tuple(range(10))


def score_checkpoint(
    sequences: Sequence[str], checkpoint_path: Path = CHECKPOINT_PATH
) -> dict[str, np.ndarray]:
    """v1's predictions for `sequences`, keyed by head name. Not unit-tested.

    v1's functional model returns its outputs as a list, so they are zipped
    against the model's own `output_names` before `predictions_by_head` keys
    them by head name.
    """
    import keras

    model = keras.models.load_model(checkpoint_path, compile=False)
    outputs = model.predict(one_hot(sequences), batch_size=1024, verbose=0)
    return predictions_by_head(dict(zip(model.output_names, outputs, strict=True)))


def build_predictions(
    membership: pd.DataFrame,
    labels: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Membership columns plus each head's label and prediction, side by side."""
    frame = membership[
        [
            "row_index",
            SEQUENCE_COLUMN,
            "protospacer",
            USE_COLUMN,
            ROLE_COLUMN,
            CATEGORY_COLUMN,
            "in_pam_library",
        ]
    ].copy()
    for head in HEAD_NAMES:
        frame[true_column(head)] = labels[head]
        frame[pred_column(head)] = preds[head]
    return frame


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order, capped at 1."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    stepped = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted.tolist()


def category_metrics(test_rows: pd.DataFrame) -> dict:
    """Per head, per category metrics over v1's test role."""
    report = {}
    for head in HEAD_NAMES:
        per_category = {}
        for category in CATEGORIES:
            y_true, y_pred = head_arrays(
                test_rows[test_rows[CATEGORY_COLUMN] == category], head
            )
            metric = (
                full_metrics if category in CORRELATED_CATEGORIES else error_metrics
            )
            per_category[category] = metric(y_true, y_pred)
        report[head] = per_category
    return report


def category_tests(
    test_rows: pd.DataFrame, n_resamples: int = 10000, seed: int = 42
) -> dict:
    """Unseen target vs PAM-variant overlap |error| per head, Holm-corrected.

    `compare_abs_errors` calls its groups seen/unseen; here PAM-variant
    overlap plays "seen" and Unseen target plays "unseen", so a positive
    delta means v1 does worse on protospacers it never trained on.
    """
    tests = {}
    for head in HEAD_NAMES:
        abs_err = {}
        for category in CORRELATED_CATEGORIES:
            y_true, y_pred = head_arrays(
                test_rows[test_rows[CATEGORY_COLUMN] == category], head
            )
            abs_err[category] = np.abs(y_pred - y_true)
        raw = compare_abs_errors(
            abs_err[PAM_VARIANT_OVERLAP],
            abs_err[UNSEEN_TARGET],
            n_resamples=n_resamples,
            seed=seed,
        )
        tests[head] = {
            "mae_delta_unseen_target_minus_pam_variant_overlap": raw[
                "mae_delta_unseen_minus_seen"
            ],
            "mae_delta_bootstrap_ci95": raw["mae_delta_bootstrap_ci95"],
            "bootstrap_resamples": raw["bootstrap_resamples"],
            "mann_whitney_u": raw["mann_whitney_u"],
            "mann_whitney_p": raw["mann_whitney_p"],
        }

    adjusted = holm_bonferroni([tests[head]["mann_whitney_p"] for head in HEAD_NAMES])
    for head, p_holm in zip(HEAD_NAMES, adjusted, strict=True):
        tests[head]["mann_whitney_p_holm"] = p_holm
    return tests


def published_test_by_role(frame: pd.DataFrame) -> dict:
    """v1 on the paper's `Use = Test` rows, split by the role v1 gave them."""
    published = frame[is_published_test(frame)]
    report = {}
    for role in ROLES:
        rows = published[published[ROLE_COLUMN] == role]
        report[role] = {
            "n": len(rows),
            "comparable_to_kim_2020": role == TEST,
            "heads": {
                head: full_metrics(*head_arrays(rows, head)) for head in HEAD_NAMES
            },
        }
    return report


def _role_maes(frame: pd.DataFrame, head: str) -> dict[str, float]:
    return {
        role: group_metrics(*head_arrays(frame[frame[ROLE_COLUMN] == role], head))[
            "mae"
        ]
        for role in ROLES
    }


def _held_out_gap(abs_err: np.ndarray, roles: np.ndarray) -> float:
    """MAE over val + test minus MAE over train."""
    return float(abs_err[roles != TRAIN].mean() - abs_err[roles == TRAIN].mean())


def role_sanity_check(
    frame: pd.DataFrame, control_seeds: Sequence[int] = CONTROL_SEEDS
) -> dict:
    """Does v1's error say it trained on exactly the rebuilt train role?

    The ticket's check is that error rises train -> val -> test. Val and test
    are both rows v1 took no gradient from, though, so their order is noise;
    the evidence is train sitting below both. `control_seeds` rebuilds the
    roles under other seeds over the same rows: if the rebuild were wrong,
    seed 42's held-out gap would look like theirs (about zero).
    """

    def rises(maes: dict[str, float]) -> bool:
        return maes[TRAIN] < maes[VAL] < maes[TEST]

    def train_lowest(maes: dict[str, float]) -> bool:
        return maes[TRAIN] < min(maes[VAL], maes[TEST])

    per_head = {}
    for head in HEAD_NAMES:
        maes = _role_maes(frame, head)
        per_head[head] = {
            "mae": maes,
            "rises_train_to_val_to_test": rises(maes),
            "train_below_val_and_test": train_lowest(maes),
        }
    mean_mae = {
        role: float(np.mean([per_head[head]["mae"][role] for head in HEAD_NAMES]))
        for role in ROLES
    }

    abs_err = np.mean(
        [np.abs(np.subtract(*head_arrays(frame, head)[::-1])) for head in HEAD_NAMES],
        axis=0,
    )
    control_gaps = {
        str(seed): _held_out_gap(
            abs_err, role_labels(rebuild_roles(len(frame), random_state=seed))
        )
        for seed in control_seeds
    }
    return {
        "per_head": per_head,
        "mean_mae_over_heads": mean_mae,
        "rises_train_to_val_to_test": rises(mean_mae),
        "heads_rising": sum(h["rises_train_to_val_to_test"] for h in per_head.values()),
        "train_below_val_and_test": train_lowest(mean_mae),
        "heads_train_below_val_and_test": sum(
            h["train_below_val_and_test"] for h in per_head.values()
        ),
        "held_out_minus_train_mae": _held_out_gap(
            abs_err, frame[ROLE_COLUMN].to_numpy()
        ),
        "control_seed_held_out_minus_train_mae": control_gaps,
    }


def build_report(frame: pd.DataFrame, n_resamples: int = 10000, seed: int = 42) -> dict:
    """Assemble the JSON-serialisable audit from the predictions frame."""
    test_rows = frame[frame[ROLE_COLUMN] == TEST]
    return {
        "membership": summarize(frame),
        "scale": "published percentage / 100 (the scale v1 was trained on)",
        "category_metrics": category_metrics(test_rows),
        "unseen_target_vs_pam_variant_overlap": {
            "groups": {"seen": PAM_VARIANT_OVERLAP, "unseen": UNSEEN_TARGET},
            "correction": "Holm-Bonferroni across the nine heads",
            "heads": category_tests(test_rows, n_resamples=n_resamples, seed=seed),
        },
        "published_test_by_role": published_test_by_role(frame),
        "role_sanity_check": role_sanity_check(frame),
        "bootstrap": {"n_resamples": n_resamples, "seed": seed, "ci": 0.95},
    }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_predictions(
    frame: pd.DataFrame,
    checkpoint_path: Path,
    predictions_path: Path = PREDICTIONS_PATH,
    checksum_path: Path = CHECKSUM_PATH,
    dataset_path: Path = DATASET_PATH,
) -> dict:
    """Write the predictions CSV and its checksum record; return the record."""
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed line ending so `file_sha256` matches across Windows and Colab.
    frame.to_csv(predictions_path, index=False, lineterminator="\n")
    record = {
        "seed": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_split": VALIDATION_SPLIT,
        "rows": len(frame),
        "roles": {role: int((frame[ROLE_COLUMN] == role).sum()) for role in ROLES},
        "dataset": dataset_path.name,
        "dataset_sha256": sha256_of(dataset_path),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_of(checkpoint_path),
        "file": predictions_path.name,
        "file_sha256": sha256_of(predictions_path),
    }
    with open(checksum_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args(argv)

    table = filter_all_heads(load_spcas9_table())
    membership = build_membership(table)
    preds = score_checkpoint(membership[SEQUENCE_COLUMN].tolist(), args.checkpoint)
    frame = build_predictions(membership, head_labels(table), preds)
    checksum = write_predictions(frame, args.checkpoint)

    report = build_report(frame, n_resamples=args.n_resamples, seed=args.seed)
    report["predictions"] = checksum
    report["libraries"] = library_versions()
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    for head, tests in report["unseen_target_vs_pam_variant_overlap"]["heads"].items():
        categories = report["category_metrics"][head]
        print(
            f"{head:>13}: Spearman unseen {categories[UNSEEN_TARGET]['spearman']:.3f} "
            f"/ PAM-variant {categories[PAM_VARIANT_OVERLAP]['spearman']:.3f}, "
            f"MAE exact dup {categories[EXACT_DUPLICATE]['mae']:.4f}, "
            f"|err| delta {tests['mae_delta_unseen_target_minus_pam_variant_overlap']:+.4f} "
            f"(Holm p {tests['mann_whitney_p_holm']:.3g})"
        )
    for role, part in report["published_test_by_role"].items():
        spearman = part["heads"]["SpCas9"]["spearman"]
        print(
            f"Use = Test, v1 {role:>5}: n={part['n']}, SpCas9 Spearman {spearman:.3f}"
        )
    sanity = report["role_sanity_check"]
    controls = sanity["control_seed_held_out_minus_train_mae"].values()
    print(
        f"Mean MAE train/val/test: {sanity['mean_mae_over_heads']} "
        f"(train lowest: {sanity['heads_train_below_val_and_test']}/9 heads, "
        f"strictly rising: {sanity['heads_rising']}/9 heads)"
    )
    print(
        f"Held-out minus train MAE: {sanity['held_out_minus_train_mae']:+.4f} "
        f"(control seeds {min(controls):+.4f} to {max(controls):+.4f})"
    )
    print(f"Wrote {PREDICTIONS_PATH}, {CHECKSUM_PATH}, {RESULTS_PATH}")
    return report


if __name__ == "__main__":
    main()
