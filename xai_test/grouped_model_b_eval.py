"""Evaluate leak-free Model B on the grouped Testset (issue #38).

Turns the Colab outputs of `colab_grouped_model_b.ipynb` (#37) into the
leak-free Model B results, written to `grouped_model_b/results.json`:

- **Metrics** for each XGBoost head in `grouped_model_b/test_predictions.csv`
  (`pred_all_rows`, `pred_seq_mean`): row-level Spearman (headline) plus
  Pearson/MAE/MSE via `paired_bootstrap.compute_metrics`; sequence-level
  Spearman, with labels and predictions averaged per normalized sequence
  first; and row-level Spearman per `cell_line_block` (L1/L2/L3).
- **Reference row:** production Model B (`NT_sacas9_fintuned_model/` +
  `hybrid_xgb_model.json`) scored on the same 514 rows with the same metrics,
  labelled `production (leaky)` - its backbone saw almost every Testset
  sequence during fine-tuning (#36).
- **Significance:** row-level paired-bootstrap CIs for each head minus
  production (`paired_bootstrap.paired_bootstrap_delta_ci`).
- **Parity check:** `NT_sacas9_grouped_model/` + the two head JSONs, loaded
  through `Model_B_Predictor`, must reproduce `test_predictions.csv` within
  `PARITY_ATOL`. Skipped (and recorded as skipped) until the checkpoint has
  been downloaded from Drive.

Models are loaded one at a time and freed before the next. Read-only with
respect to every existing file; writes only `grouped_model_b/results.json`.

Usage: python grouped_model_b_eval.py [--n-resamples 1000] [--batch-size 64]
"""

import argparse
import gc
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from grouped_model_b import CHECKPOINT_DIRNAME, PREDICTION_COLUMNS
from grouped_split import BASE_DIR, OUTPUT_DIR, normalize_sequences
from grouped_testset import TESTSET_FILENAME
from ism_sweep import dispatch_in_batches
from paired_bootstrap import METRIC_FUNCS, compute_metrics, paired_bootstrap_delta_ci
from scipy.stats import spearmanr

RESULTS_DIR = BASE_DIR / "grouped_model_b"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
METADATA_PATH = OUTPUT_DIR / TESTSET_FILENAME
OUTPUT_PATH = RESULTS_DIR / "results.json"

GROUPED_CHECKPOINT_DIR = BASE_DIR / CHECKPOINT_DIRNAME
HEAD_XGB_PATHS = {
    "pred_all_rows": RESULTS_DIR / "xgb_all_rows.json",
    "pred_seq_mean": RESULTS_DIR / "xgb_seq_mean.json",
}
HEADS = tuple(HEAD_XGB_PATHS)
HEAD_LABELS = {
    "pred_all_rows": "grouped all-rows (leak-free)",
    "pred_seq_mean": "grouped seq-mean (leak-free)",
}

PRODUCTION_NT_DIR = BASE_DIR / "NT_sacas9_fintuned_model"
PRODUCTION_XGB_PATH = BASE_DIR / "hybrid_xgb_model.json"
PRODUCTION_LABEL = "production (leaky)"

# Colab predicted through Model_B_Predictor in fp32 on GPU; the local check
# runs fp32 on CPU. Kernel-level float noise in the embeddings can tip an
# XGBoost split and move a prediction by a few thousandths, but a different
# checkpoint or head moves it by far more than this.
PARITY_ATOL = 0.01

# Earlier Model B numbers, both inflated by Testset leakage (#36).
REFERENCE_NUMBERS = {
    "production_old_testset_spearman": 0.896,
    "clean_split_ablation_spearman": 0.845,
    "note": (
        "Row-level Spearman on the old seed=42 Testset (test_metadata.csv): "
        "production Model B, and the clean-split ablation from #13. Both use "
        "NT_sacas9_fintuned_model, whose fine-tune saw almost every Testset "
        "sequence, so neither is leak-free."
    ),
}


def load_predictions(
    predictions_path: Path = PREDICTIONS_PATH, metadata_path: Path = METADATA_PATH
) -> pd.DataFrame:
    """`test_predictions.csv`, checked row by row against the grouped Testset."""
    predictions = pd.read_csv(predictions_path)
    metadata = pd.read_csv(metadata_path)
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            f"{predictions_path} columns {list(predictions.columns)}, "
            f"expected {PREDICTION_COLUMNS}"
        )
    if len(predictions) != len(metadata) or not np.array_equal(
        predictions["sample_id"], metadata["sample_id"]
    ):
        raise ValueError("Predictions and Testset disagree on sample_id")

    for column in ("original_id", "sequence", "cell_line_block"):
        if not predictions[column].equals(metadata[column]):
            raise ValueError(f"Predictions and Testset disagree on {column}")
    if not np.allclose(
        predictions["true_score"], metadata["true_score"], rtol=0.0, atol=1e-9
    ):
        raise ValueError("Predictions and Testset disagree on true_score")
    return predictions


def sequence_level(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Mean `true_score` and mean prediction per normalized sequence."""
    return (
        pd.DataFrame(
            {
                "normalized_sequence": normalize_sequences(df["sequence"]),
                "true_score": df["true_score"].to_numpy(dtype=np.float64),
                "pred": df[pred_col].to_numpy(dtype=np.float64),
            }
        )
        .groupby("normalized_sequence", as_index=False)
        .mean()
    )


def evaluate_predictions(df: pd.DataFrame, pred_col: str) -> dict:
    """Row-level metrics, sequence-level Spearman and per-block Spearman."""
    y_true = df["true_score"].to_numpy(dtype=np.float64)
    y_pred = df[pred_col].to_numpy(dtype=np.float64)
    per_sequence = sequence_level(df, pred_col)

    per_block = {}
    for block, part in df.groupby("cell_line_block"):
        per_block[str(block)] = {
            "n": len(part),
            "spearman": float(spearmanr(part["true_score"], part[pred_col])[0]),
        }
    return {
        "n_rows": len(df),
        "row_level": compute_metrics(y_true, y_pred),
        "sequence_level": {
            "n_sequences": len(per_sequence),
            "spearman": float(
                spearmanr(per_sequence["true_score"], per_sequence["pred"])[0]
            ),
        },
        "per_cell_line_block": per_block,
    }


def compare_to_reference(
    y_true: np.ndarray,
    preds: np.ndarray,
    reference: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """`metric(preds) - metric(reference)` with a paired-bootstrap 95% CI, per metric."""
    y_true = np.asarray(y_true, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    ours, theirs = compute_metrics(y_true, preds), compute_metrics(y_true, reference)
    return {
        metric: {
            "delta": ours[metric] - theirs[metric],
            "ci95": list(
                paired_bootstrap_delta_ci(
                    y_true, preds, reference, metric, n_resamples=n_resamples, seed=seed
                )
            ),
        }
        for metric in METRIC_FUNCS
    }


def check_parity(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    atol: float = PARITY_ATOL,
) -> dict:
    """Per-head agreement of `actual` with `expected`; `passed` only if every row is within `atol`."""
    heads = {}
    for head, expected_preds in expected.items():
        expected_preds = np.asarray(expected_preds, dtype=np.float64)
        actual_preds = np.asarray(actual[head], dtype=np.float64)
        diff = np.abs(actual_preds - expected_preds)
        heads[head] = {
            "max_abs_diff": float(diff.max()),
            "n_outside_atol": int((diff > atol).sum()),
            "spearman": float(spearmanr(actual_preds, expected_preds)[0]),
        }
    passed = all(h["n_outside_atol"] == 0 for h in heads.values())
    return {"status": "passed" if passed else "failed", "atol": atol, "heads": heads}


def score_heads(
    nt_model_dir: Path,
    xgb_paths: Mapping[str, Path],
    sequences: list[str],
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Predict `sequences` with one NT backbone and each XGBoost head in `xgb_paths`.

    Features are built exactly as `Model_B_Predictor.predict` does, once per
    distinct sequence. The model is freed before returning.
    """
    from model_b_wrapper import Model_B_Predictor

    predictor = Model_B_Predictor(
        nt_model_dir=str(nt_model_dir),
        xgb_model_path=str(next(iter(xgb_paths.values()))),
    )
    unique = sorted(set(sequences))
    embeddings = dispatch_in_batches(
        predictor.extract_nt_embeddings, unique, batch_size, stack=np.vstack
    )
    features = np.hstack([embeddings, predictor.compute_physical_features(unique)])
    assert features.shape == (len(unique), 8964), features.shape
    row_of = {seq: i for i, seq in enumerate(unique)}
    rows = [row_of[seq] for seq in sequences]

    preds = {}
    for name, path in xgb_paths.items():
        head = predictor._load_xgb_model(str(path))
        preds[name] = np.asarray(head.predict(features), dtype=np.float64).reshape(-1)[
            rows
        ]
    del predictor
    gc.collect()
    return preds


def score_production(df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    """Production Model B's predictions on the Testset rows of `df`."""
    return score_heads(
        PRODUCTION_NT_DIR,
        {"production": PRODUCTION_XGB_PATH},
        df["sequence"].tolist(),
        batch_size,
    )["production"]


def run_parity_check(
    df: pd.DataFrame,
    checkpoint_dir: Path = GROUPED_CHECKPOINT_DIR,
    batch_size: int = 64,
) -> dict:
    """Re-predict `df` with the downloaded grouped checkpoint; skipped if it is absent."""
    if not checkpoint_dir.is_dir():
        return {
            "status": "skipped",
            "reason": (
                f"{checkpoint_dir.name}/ not found; download the fine-tuned "
                f"checkpoint from Google Drive into {checkpoint_dir} and rerun"
            ),
        }
    actual = score_heads(
        checkpoint_dir, HEAD_XGB_PATHS, df["sequence"].tolist(), batch_size
    )
    return check_parity({head: df[head].to_numpy() for head in HEADS}, actual)


def build_results(
    df: pd.DataFrame,
    production_preds: np.ndarray,
    parity: dict,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Assemble the JSON-serialisable `results.json` payload."""
    production_preds = np.asarray(production_preds, dtype=np.float64).reshape(-1)
    if len(production_preds) != len(df):
        raise ValueError(
            f"{len(production_preds)} production predictions for {len(df)} Testset rows"
        )
    scored = df.assign(production=production_preds)
    y_true = df["true_score"].to_numpy(dtype=np.float64)

    models = {
        head: {
            "label": HEAD_LABELS[head],
            "leaky": False,
            **evaluate_predictions(scored, head),
        }
        for head in HEADS
    }
    models["production"] = {
        "label": PRODUCTION_LABEL,
        "leaky": True,
        **evaluate_predictions(scored, "production"),
    }
    return {
        "testset": {
            "rows": len(df),
            "sequences": int(normalize_sequences(df["sequence"]).nunique()),
        },
        "models": models,
        "vs_production": {
            head: compare_to_reference(
                y_true, df[head].to_numpy(), production_preds, n_resamples, seed
            )
            for head in HEADS
        },
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "ci": 0.95,
            "level": "row",
        },
        "parity": parity,
        "old_reference": REFERENCE_NUMBERS,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    df = load_predictions()
    production = score_production(df, args.batch_size)
    parity = run_parity_check(df, batch_size=args.batch_size)
    if parity["status"] == "skipped":
        print(f"Parity check skipped: {parity['reason']}")

    results = build_results(df, production, parity, args.n_resamples, args.seed)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for model in results["models"].values():
        print(
            f"{model['label']:>30}: row Spearman {model['row_level']['spearman']:.3f}, "
            f"sequence Spearman {model['sequence_level']['spearman']:.3f}, "
            "per block "
            + ", ".join(
                f"{b} {m['spearman']:.3f}"
                for b, m in model["per_cell_line_block"].items()
            )
        )
    for head, deltas in results["vs_production"].items():
        lower, upper = deltas["spearman"]["ci95"]
        print(
            f"{head} - production: Spearman delta {deltas['spearman']['delta']:+.3f} "
            f"(95% CI {lower:+.3f}, {upper:+.3f})"
        )
    print(f"Parity: {parity['status']}")
    print(f"Wrote {OUTPUT_PATH}")
    if parity["status"] == "failed":
        raise SystemExit(f"Parity check failed: {parity['heads']}")
    return results


if __name__ == "__main__":
    main()
