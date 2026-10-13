"""Testset leakage check: Model B error on seen vs unseen sequences (issue #32).

The NT regression checkpoint and production XGBoost behind Model B were trained on the
old `sacas9_Dataset/` split (`train.csv`/`val.csv`/`test.csv`), which predates
the seed=42 Testset isolation. This script measures how much of that overlap
shows up in Model B's Testset error, fully offline:

- Joins `test_metadata.csv` with the cached
  `final_analysis_result/model_b_testset_predictions.csv` (no model loading).
- Labels each Testset row **seen** (sequence in the old `train.csv`) or
  **unseen**; old `val.csv`/`test.csv` membership is reported separately.
- Reports n/MAE/MSE/Spearman per group, plus a Mann-Whitney U test and a
  bootstrap CI on the MAE difference of per-sample absolute errors.
- Tags each Case Study sample (from `model_analysis_summary.json`) seen/unseen.

Read-only with respect to every existing file; writes only
`leakage_check_result.json`.

Usage: python leakage_check.py
"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "test_metadata.csv"
PREDICTIONS_PATH = (
    BASE_DIR / "final_analysis_result" / "model_b_testset_predictions.csv"
)
SUMMARY_PATH = BASE_DIR / "final_analysis_result" / "model_analysis_summary.json"
OLD_SPLIT_DIR = BASE_DIR / "sacas9_Dataset"
OLD_SPLITS = ("train", "val", "test")
OUTPUT_PATH = BASE_DIR / "leakage_check_result.json"

# Spearman on fewer points than this is too unstable to be worth reporting.
MIN_N_FOR_SPEARMAN = 10

LOW_POWER_NOTE = (
    "The unseen group is tiny, so its metrics and the seen-vs-unseen tests "
    "have very low statistical power; treat them as indicative only. Rows "
    "are treated as independent, but duplicate sequences make them mildly "
    "correlated, so the p-value and CI are slightly optimistic."
)


def _normalize(sequences: pd.Series) -> pd.Series:
    return sequences.astype(str).str.strip().str.upper()


def load_old_splits(split_dir: Path = OLD_SPLIT_DIR) -> dict[str, set[str]]:
    """Normalized sequence sets of the old `sacas9_Dataset/` split, keyed by split name."""
    return {
        name: set(_normalize(pd.read_csv(split_dir / f"{name}.csv")["input_sequence"]))
        for name in OLD_SPLITS
    }


def label_membership(
    sequences: pd.Series, splits: Mapping[str, set[str]]
) -> pd.DataFrame:
    """Per-row `in_<split>` flags plus `seen` (= in the old train split)."""
    normalized = _normalize(sequences).reset_index(drop=True)
    labels = pd.DataFrame(
        {f"in_{name}": normalized.isin(seqs) for name, seqs in splits.items()}
    )
    labels["seen"] = labels["in_train"]
    return labels


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """n/MAE/MSE always; Spearman only when n >= MIN_N_FOR_SPEARMAN (else None)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    n = len(y_true)
    spearman = float(spearmanr(y_true, y_pred)[0]) if n >= MIN_N_FOR_SPEARMAN else None
    return {
        "n": n,
        "mae": float(np.mean(np.abs(residual))) if n else None,
        "mse": float(np.mean(residual**2)) if n else None,
        "spearman": spearman,
    }


def compare_abs_errors(
    err_seen: np.ndarray,
    err_unseen: np.ndarray,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Seen-vs-unseen comparison of per-sample absolute errors.

    The two groups are independent samples (different sequences), so each
    bootstrap draw resamples each group separately with replacement. The
    delta is `MAE(unseen) - MAE(seen)`: positive means Model B does worse on
    sequences it never trained on, which is what memorization would predict.
    The Mann-Whitney U test is two-sided.
    """
    err_seen = np.asarray(err_seen, dtype=np.float64)
    err_unseen = np.asarray(err_unseen, dtype=np.float64)

    rng = np.random.default_rng(seed)
    seen_draws = rng.choice(err_seen, size=(n_resamples, len(err_seen))).mean(axis=1)
    unseen_draws = rng.choice(err_unseen, size=(n_resamples, len(err_unseen))).mean(
        axis=1
    )
    deltas = unseen_draws - seen_draws
    alpha = (1.0 - ci) / 2.0
    lower, upper = np.quantile(deltas, [alpha, 1.0 - alpha])

    mw = mannwhitneyu(err_unseen, err_seen, alternative="two-sided")
    return {
        "mae_delta_unseen_minus_seen": float(err_unseen.mean() - err_seen.mean()),
        "mae_delta_bootstrap_ci95": [float(lower), float(upper)],
        "bootstrap_resamples": n_resamples,
        "mann_whitney_u": float(mw.statistic),
        "mann_whitney_p": float(mw.pvalue),
    }


def case_study_membership(
    case_studies: list[dict], seen_by_sample: Mapping[int, bool]
) -> dict:
    """Tag each Case Study seen/unseen and count per `case_type` and overall."""
    cases = []
    counts_by_type: dict[str, dict[str, int]] = {}
    totals = {"seen": 0, "unseen": 0}
    for case in case_studies:
        seen = bool(seen_by_sample[int(case["sample_id"])])
        key = "seen" if seen else "unseen"
        cases.append(
            {
                "case_id": case["case_id"],
                "sample_id": int(case["sample_id"]),
                "case_type": case["case_type"],
                "seen": seen,
            }
        )
        counts_by_type.setdefault(case["case_type"], {"seen": 0, "unseen": 0})[key] += 1
        totals[key] += 1
    return {"cases": cases, "counts_by_type": counts_by_type, "totals": totals}


def build_report(
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
    splits: Mapping[str, set[str]],
    case_studies: list[dict],
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """Assemble the full JSON-serialisable leakage report."""
    joined = metadata.merge(
        predictions[["sample_id", "pred_raw"]], on="sample_id", how="inner"
    )
    if len(joined) != len(metadata) or len(joined) != len(predictions):
        raise ValueError(
            f"Predictions do not align with metadata by sample_id: "
            f"{len(metadata)} metadata rows, {len(predictions)} prediction rows, "
            f"{len(joined)} joined"
        )

    labels = label_membership(joined["sequence"], splits)
    joined = pd.concat([joined.reset_index(drop=True), labels], axis=1)
    joined["normalized_sequence"] = _normalize(joined["sequence"])
    seen = joined["seen"].to_numpy()

    unique = joined.drop_duplicates("normalized_sequence")
    membership = {
        "seen_definition": "sequence appears in sacas9_Dataset/train.csv",
        "unique_sequences": {
            "total": len(unique),
            "seen": int(unique["seen"].sum()),
            "unseen": int((~unique["seen"]).sum()),
        },
        "rows": {
            "total": len(joined),
            "seen": int(seen.sum()),
            "unseen": int((~seen).sum()),
        },
        "unique_sequences_in_old_split": {
            name: int(unique[f"in_{name}"].sum()) for name in splits
        },
        "unseen_unique_sequences": sorted(
            unique.loc[~unique["seen"], "normalized_sequence"].tolist()
        ),
    }

    y_true = joined["true_score"].to_numpy(dtype=np.float64)
    y_pred = joined["pred_raw"].to_numpy(dtype=np.float64)
    abs_err = np.abs(y_pred - y_true)

    model_b = {
        "all": group_metrics(y_true, y_pred),
        "seen": group_metrics(y_true[seen], y_pred[seen]),
        "unseen": group_metrics(y_true[~seen], y_pred[~seen]),
        "note": LOW_POWER_NOTE,
    }
    if seen.any() and (~seen).any():
        model_b["seen_vs_unseen_abs_error"] = compare_abs_errors(
            abs_err[seen], abs_err[~seen], n_resamples=n_resamples, seed=seed
        )

    seen_by_sample = dict(zip(joined["sample_id"].astype(int), joined["seen"]))
    return {
        "membership": membership,
        "model_b": model_b,
        "case_studies": case_study_membership(case_studies, seen_by_sample),
    }


def main() -> dict:
    metadata = pd.read_csv(METADATA_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        case_studies = json.load(f)["case_studies"]

    report = build_report(metadata, predictions, load_old_splits(), case_studies)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    membership, model_b = report["membership"], report["model_b"]
    print(
        f"Unique sequences: {membership['unique_sequences']}  "
        f"rows: {membership['rows']}"
    )
    for group in ("all", "seen", "unseen"):
        print(f"Model B [{group}]: {model_b[group]}")
    if "seen_vs_unseen_abs_error" in model_b:
        print(f"Seen vs unseen |error|: {model_b['seen_vs_unseen_abs_error']}")
    print(f"Case Studies: {report['case_studies']['totals']}")
    print(f"NOTE: {LOW_POWER_NOTE}")
    print(f"Wrote {OUTPUT_PATH}")
    return report


if __name__ == "__main__":
    main()
