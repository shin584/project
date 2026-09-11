"""Model B physical-feature ablation + paired bootstrap (Step 4-1, issue #13).

Trains three XGBoost regressors - Embedding-only (8,960d), Physical-only
(4d), and Full (8,964d) - on the same dev split, seed, and hyperparameters
as production Model B (`nt_feature_stacking.py`'s Phase 4), then paired-
bootstraps Full against each ablated variant on the Testset. No cross-model
or cross-variant score rescaling is applied anywhere in this module.
"""

import numpy as np
import xgboost as xgb
from paired_bootstrap import compute_metrics, paired_bootstrap_delta_ci

# Same protocol as nt_feature_stacking.py's Phase 4 XGBoost fit - shared here
# so all three ablation variants (and production Model B) are provably using
# identical hyperparameters, not three independently-typed copies of them.
XGB_HYPERPARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist",
    "early_stopping_rounds": 20,
}

EMBEDDING_DIM = 8960
PHYSICAL_DIM = 4

# Bootstrap deltas are always Full minus the named ablated variant, so a
# negative delta_mse / positive delta_spearman means Full is better.
BOOTSTRAP_METRICS = ("spearman", "mse")


def split_ablation_features(
    X_full: np.ndarray, embedding_dim: int = EMBEDDING_DIM
) -> dict[str, np.ndarray]:
    """Slice a `(n, embedding_dim + physical_dim)` feature matrix into the 3 variants' inputs."""
    return {
        "embedding": X_full[:, :embedding_dim],
        "physical": X_full[:, embedding_dim:],
        "full": X_full,
    }


def train_ablation_variant(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> xgb.XGBRegressor:
    """Train one XGBoost regressor under the shared `XGB_HYPERPARAMS` protocol."""
    model = xgb.XGBRegressor(**XGB_HYPERPARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bootstrap_seed: int = 42,
    embedding_dim: int = EMBEDDING_DIM,
) -> dict:
    """Train all 3 variants, evaluate on the Testset, and paired-bootstrap Full vs each.

    Returns a dict with `ablation_results` (per-variant Spearman/Pearson/MAE/MSE
    on the Testset, raw/unrescaled) and `paired_bootstrap_95ci` (95% CI for
    `metric(full) - metric(variant)`, for each of `BOOTSTRAP_METRICS`).
    `embedding_dim` only needs overriding in tests against synthetic,
    lower-dimensional feature matrices.
    """
    train_splits = split_ablation_features(X_train, embedding_dim)
    val_splits = split_ablation_features(X_val, embedding_dim)
    test_splits = split_ablation_features(X_test, embedding_dim)

    # These three key names ("embedding_only"/"physical_only"/"full_model") are
    # mandated verbatim by the Standard Export Schema (plan_realize.md section
    # 4, `ablation_results`), which issue #17 must reproduce exactly. This is
    # a deliberate exception to CONTEXT.md's glossary guidance to avoid
    # "embedding_only" (there, it warns against conflating it with "NT-only
    # regression head performance", a different, unrelated pipeline) - kept
    # here only because it's always nested under `ablation_results`, which
    # disambiguates it as the Step 4(1) ablation variant, never that other
    # metric.
    result_key = {
        "embedding": "embedding_only",
        "physical": "physical_only",
        "full": "full_model",
    }

    metrics_by_variant = {}
    predictions_by_variant = {}
    for name in ("embedding", "physical", "full"):
        model = train_ablation_variant(
            train_splits[name], y_train, val_splits[name], y_val
        )
        preds = model.predict(test_splits[name])
        predictions_by_variant[name] = preds
        metrics_by_variant[name] = compute_metrics(y_test, preds)

    bootstrap_cis = {}
    for name in ("embedding", "physical"):
        for metric in BOOTSTRAP_METRICS:
            lower, upper = paired_bootstrap_delta_ci(
                y_test,
                predictions_by_variant["full"],
                predictions_by_variant[name],
                metric=metric,
                seed=bootstrap_seed,
            )
            bootstrap_cis[f"delta_{metric}_full_vs_{name}_only"] = [lower, upper]

    return {
        "ablation_results": {
            result_key[name]: metrics_by_variant[name]
            for name in ("embedding", "physical", "full")
        },
        "paired_bootstrap_95ci": bootstrap_cis,
    }
