"""Leak-free Model B on the sequence-grouped split (issue #37).

Building blocks for `colab_grouped_model_b.ipynb`, which retrains Model B on
`grouped_split/` (#34/#35) instead of the stale, leaky `train.csv`/`val.csv`:

- Rows of `SaCas9_v4.xlsx` are joined to `grouped_split/assignments.csv` on
  `original_id`; train/val/test must share no sequence, and the test rows
  must match `grouped_split/test_metadata.csv` and its checksum.
- A new NT regression checkpoint is trained with the `nt_pretrain.py` recipe
  unchanged (`FINETUNE_ARGS` mirrors its `TrainingArguments`); only the data
  differs.
- Two XGBoost heads are trained with `ablation.XGB_HYPERPARAMS` on the
  `Model_B_Predictor` features: **all-rows** (one sample per replicate row,
  as in production) and **seq-mean** (one sample per sequence, labels
  averaged across its replicates).

The notebook needs a GPU for the fine-tune and feature extraction; everything
here is also runnable on CPU against small inputs, which is how it is tested.
"""

import importlib.metadata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from ablation import XGB_HYPERPARAMS, train_ablation_variant
from grouped_split import SPLITS, normalize_sequences
from grouped_testset import column_checksum
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

CHECKPOINT_DIRNAME = "NT_sacas9_grouped_model"

SEED = 42
NT_BASE_MODEL = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
MAX_LENGTH = 64
EARLY_STOPPING_PATIENCE = 3
# nt_pretrain.py's TrainingArguments, minus its Colab-specific paths, plus an
# explicit seed (nt_pretrain.py relied on the Trainer default, also 42).
FINETUNE_ARGS = {
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "fp16": True,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "logging_steps": 10,
    "seed": SEED,
}
# Disk only, not part of the recipe: each epoch checkpoint (weights + optimizer
# state) is ~6 GB, and the Trainer never deletes the best one.
CHECKPOINT_STORAGE_ARGS = {"save_total_limit": 2}

PREDICTION_COLUMNS = [
    "sample_id",
    "original_id",
    "sequence",
    "cell_line_block",
    "true_score",
    "pred_all_rows",
    "pred_seq_mean",
]
LIBRARIES = (
    "torch",
    "transformers",
    "accelerate",
    "xgboost",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "biopython",
)


def join_grouped_splits(
    full_df: pd.DataFrame, assignments: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Dataset rows per split, joined on `original_id`, in source row order."""
    rows = full_df[["original_id", "sequence", "true_score"]].merge(
        assignments[["original_id", "normalized_sequence", "cell_line_block", "split"]],
        on="original_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = rows[rows["_merge"] != "both"]
    if len(unmatched):
        raise ValueError(
            f"{len(unmatched)} original_id values are in only one of the dataset "
            f"and assignments.csv, e.g. {unmatched['original_id'].head().tolist()}"
        )
    rows = rows.drop(columns="_merge").sort_values("original_id", kind="stable")
    return {
        split: rows[rows["split"] == split].drop(columns="split").reset_index(drop=True)
        for split in SPLITS
    }


def verify_no_sequence_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """No normalized sequence may appear in more than one split."""
    for a, b in combinations(SPLITS, 2):
        shared = set(normalize_sequences(splits[a]["sequence"])) & set(
            normalize_sequences(splits[b]["sequence"])
        )
        if shared:
            raise ValueError(
                f"{a}/{b} sequence overlap: {len(shared)} sequences, "
                f"e.g. {sorted(shared)[:3]}"
            )


def verify_grouped_testset(
    test_df: pd.DataFrame, testset_metadata: pd.DataFrame, checksum_record: dict
) -> None:
    """Joined test rows must equal `grouped_split/test_metadata.csv` + its checksum."""
    if (
        column_checksum(testset_metadata["sequence"])
        != checksum_record["dataset_sha256"]
    ):
        raise ValueError(
            "grouped_split/test_metadata.csv does not match the dataset checksum "
            "in grouped_split/test_checksum.json"
        )
    if len(test_df) != checksum_record["rows"]:
        raise ValueError(
            f"{len(test_df)} test rows, expected {checksum_record['rows']}"
        )

    columns = ["original_id", "sequence", "true_score", "cell_line_block"]
    try:
        pd.testing.assert_frame_equal(
            test_df[columns].reset_index(drop=True),
            testset_metadata[columns].reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as err:
        raise ValueError(
            f"Grouped test rows do not match grouped_split/test_metadata.csv: {err}"
        ) from err


def finetune_compute_metrics(eval_pred) -> dict[str, float]:
    """`nt_pretrain.py`'s `compute_metrics`, unchanged."""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()

    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if labels.ndim > 1:
        labels = labels.flatten()

    if np.std(predictions) == 0 or np.std(labels) == 0:
        return {"spearman_rho": 0.0, "pearson_r": 0.0, "mae": 0.0, "mse": 0.0}

    return {
        "spearman_rho": float(spearmanr(labels, predictions)[0]),
        "pearson_r": float(pearsonr(labels, predictions)[0]),
        "mae": float(mean_absolute_error(labels, predictions)),
        "mse": float(mean_squared_error(labels, predictions)),
    }


def _tokenize(tokenizer, df: pd.DataFrame) -> list[dict]:
    """`nt_pretrain.py`'s `preprocess_function`, as a list the Trainer can index."""
    tokenized = tokenizer(
        df["sequence"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    return [
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": float(label),
        }
        for input_ids, attention_mask, label in zip(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            df["true_score"],
            strict=True,
        )
    ]


def finetune_nt(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str | Path,
    model_name: str = NT_BASE_MODEL,
    seed: int = SEED,
    overrides: dict | None = None,
):
    """Fine-tune NT with the `nt_pretrain.py` recipe; returns `(trainer, tokenizer)`.

    The trainer holds the best (lowest val loss) checkpoint after training.
    `overrides` exists only so tests can run the recipe on CPU.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, ignore_mismatched_sizes=True
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        **{
            **FINETUNE_ARGS,
            **CHECKPOINT_STORAGE_ARGS,
            "seed": seed,
            **(overrides or {}),
        },
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_tokenize(tokenizer, train_df),
        eval_dataset=_tokenize(tokenizer, val_df),
        compute_metrics=finetune_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        ],
    )
    trainer.train()
    return trainer, tokenizer


def summarize_finetune_history(log_history: list[dict]) -> dict:
    """Per-epoch val metrics from `trainer.state.log_history`, plus the best epoch."""
    epochs = [
        {k: v for k, v in entry.items() if k == "epoch" or k.startswith("eval_")}
        for entry in log_history
        if "eval_loss" in entry
    ]
    best = min(epochs, key=lambda e: e["eval_loss"])
    return {
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_eval_loss": best["eval_loss"],
    }


def sequence_mean(
    X: np.ndarray, y: np.ndarray, groups: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """One sample per sequence (first-occurrence order), labels averaged per sequence.

    Replicates of a sequence must carry identical features, since features
    are a function of the sequence alone.
    """
    codes, _ = pd.factorize(pd.Series(groups).reset_index(drop=True))
    first_row = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
    if not np.array_equal(X, X[first_row[codes]]):
        raise ValueError("Replicates of the same sequence have different features")

    counts = np.bincount(codes)
    return X[first_row], np.bincount(codes, weights=y) / counts


@dataclass
class HeadFit:
    model: xgb.XGBRegressor
    n_train_samples: int
    n_val_samples: int


def train_heads(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: pd.Series,
    X_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: pd.Series,
) -> dict[str, HeadFit]:
    """The all-rows and seq-mean XGBoost heads, both early-stopped on val."""
    X_train_seq, y_train_seq = sequence_mean(X_train, y_train, groups_train)
    X_val_seq, y_val_seq = sequence_mean(X_val, y_val, groups_val)
    datasets = {
        "all_rows": (X_train, y_train, X_val, y_val),
        "seq_mean": (X_train_seq, y_train_seq, X_val_seq, y_val_seq),
    }
    return {
        name: HeadFit(
            model=train_ablation_variant(X_fit, y_fit, X_eval, y_eval),
            n_train_samples=len(X_fit),
            n_val_samples=len(X_eval),
        )
        for name, (X_fit, y_fit, X_eval, y_eval) in datasets.items()
    }


def build_test_predictions(
    testset_metadata: pd.DataFrame,
    pred_all_rows: np.ndarray,
    pred_seq_mean: np.ndarray,
) -> pd.DataFrame:
    """`test_predictions.csv`: one row per grouped Testset row."""
    n = len(testset_metadata)
    if len(pred_all_rows) != n or len(pred_seq_mean) != n:
        raise ValueError(
            f"Expected {n} predictions per head, got "
            f"{len(pred_all_rows)} (all-rows) and {len(pred_seq_mean)} (seq-mean)"
        )
    table = testset_metadata.reset_index(drop=True).copy()
    table["pred_all_rows"] = np.asarray(pred_all_rows, dtype=float)
    table["pred_seq_mean"] = np.asarray(pred_seq_mean, dtype=float)
    return table[PREDICTION_COLUMNS]


def library_versions() -> dict[str, str | None]:
    """Installed version of each of `LIBRARIES`, `None` if missing."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_training_log(
    log_history: list[dict],
    heads: dict[str, HeadFit],
    n_rows: dict[str, int],
    seed: int = SEED,
) -> dict:
    """`training_log.json`: fine-tune val metrics, XGBoost fits, seed, versions."""
    return {
        "seed": seed,
        "n_rows": n_rows,
        "recipe": {
            "base_model": NT_BASE_MODEL,
            "max_length": MAX_LENGTH,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "finetune_args": FINETUNE_ARGS,
            "xgb_hyperparams": XGB_HYPERPARAMS,
        },
        "finetune": summarize_finetune_history(log_history),
        "xgboost": {
            name: {
                "best_iteration": int(fit.model.best_iteration),
                "best_score": float(fit.model.best_score),
                "n_train_samples": fit.n_train_samples,
                "n_val_samples": fit.n_val_samples,
            }
            for name, fit in heads.items()
        },
        "library_versions": library_versions(),
    }
