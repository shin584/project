"""Tree SHAP analysis for production Model B-Full (Step 4-2, issue #14).

Computes exact per-feature Tree SHAP contributions via XGBoost's native
`pred_contribs` prediction mode - the same Tree SHAP algorithm the `shap`
package implements, built directly into the booster, so no extra dependency
is needed. Separates the top-20 sequence-embedding features from the 4 named
physical features (MFE, delta-G, Tm, GC) and reports an aggregate physical
SHAP contribution ratio, matching issue #14's acceptance criteria.
"""

import os

import numpy as np
import xgboost as xgb

EMBEDDING_DIM = 8960
PHYSICAL_DIM = 4
# Column order must match Model_B_Predictor.compute_physical_features's
# hstack order (mfe, dg, tm, gc) - see model_b_wrapper.py.
PHYSICAL_FEATURE_NAMES = ("MFE", "delta_G", "Tm", "GC")

DEFAULT_TOP_K = 20


def compute_tree_shap(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Exact per-feature Tree SHAP contributions for `X`, bias column dropped.

    `pred_contribs=True` returns one extra trailing column (the expected-value
    bias term) needed for additivity - `row.sum() + bias == raw prediction`.
    That bias isn't a feature contribution, so it's sliced off here; every
    downstream consumer works with a plain `(n_samples, n_features)` matrix.
    """
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    return contribs[:, :-1]


def split_shap_by_group(
    shap_values: np.ndarray,
    embedding_dim: int = EMBEDDING_DIM,
    physical_dim: int = PHYSICAL_DIM,
) -> dict[str, np.ndarray]:
    """Slice a `(n, embedding_dim + physical_dim)` SHAP matrix by feature group."""
    assert shap_values.shape[1] == embedding_dim + physical_dim, (
        f"Expected {embedding_dim + physical_dim} SHAP columns, got {shap_values.shape[1]}"
    )
    return {
        "embedding": shap_values[:, :embedding_dim],
        "physical": shap_values[:, embedding_dim:],
    }


def top_k_embedding_features(
    shap_embedding: np.ndarray, k: int = DEFAULT_TOP_K
) -> list[dict]:
    """Rank embedding features by mean |SHAP| across samples, descending.

    `feature_index` is local to the embedding sub-block (0..embedding_dim-1,
    e.g. 0-8959 in production), not the full 8,964-wide matrix - callers
    that need a global column index must add `embedding_dim` themselves.
    Caps at the number of available columns rather than raising when `k`
    exceeds the embedding width (relevant for tests against synthetic,
    lower-dimensional embeddings).
    """
    mean_abs = np.abs(shap_embedding).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:k]
    return [
        {"feature_index": int(i), "mean_abs_shap": float(mean_abs[i])} for i in order
    ]


def physical_feature_shap_summary(shap_physical: np.ndarray) -> list[dict]:
    """Mean |SHAP| per named physical feature, in `PHYSICAL_FEATURE_NAMES` order."""
    mean_abs = np.abs(shap_physical).mean(axis=0)
    return [
        {"feature_name": name, "mean_abs_shap": float(mean_abs[i])}
        for i, name in enumerate(PHYSICAL_FEATURE_NAMES)
    ]


def aggregate_physical_contribution_ratio(
    shap_values: np.ndarray, embedding_dim: int = EMBEDDING_DIM
) -> float:
    """`sum(|SHAP_physical|) / sum(|SHAP_all|)` - a proportion in [0, 1]."""
    total_abs = np.abs(shap_values).sum()
    physical_abs = np.abs(shap_values[:, embedding_dim:]).sum()
    return float(physical_abs / total_abs)


def dependence_plot_data(
    X_physical: np.ndarray,
    shap_physical: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Per physical feature: its raw values paired with their own SHAP column."""
    return {
        name: {
            "feature_value": X_physical[:, i],
            "shap_value": shap_physical[:, i],
        }
        for i, name in enumerate(PHYSICAL_FEATURE_NAMES)
    }


def save_dependence_plots(
    X_physical: np.ndarray,
    shap_physical: np.ndarray,
    output_dir: str,
) -> dict[str, str]:
    """Render one feature-value-vs-SHAP-value scatter plot per physical feature."""
    # Imported lazily so importing this module (e.g. for the pure array logic
    # above, as every test in test_shap_analysis.py does) never pays for
    # matplotlib or forces its Agg backend unless a plot is actually requested.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    data = dependence_plot_data(X_physical, shap_physical)

    paths = {}
    for name, values in data.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(values["feature_value"], values["shap_value"], s=10, alpha=0.6)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel(name)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Tree SHAP dependence: {name}")
        path = os.path.join(output_dir, f"dependence_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[name] = path
    return paths


def run_shap_analysis(
    model: xgb.XGBRegressor,
    X: np.ndarray,
    embedding_dim: int = EMBEDDING_DIM,
    physical_dim: int = PHYSICAL_DIM,
    top_k: int = DEFAULT_TOP_K,
    dependence_plot_dir: str | None = None,
) -> dict:
    """Run the full Tree SHAP analysis bundle against issue #14's acceptance criteria.

    Returns `top_embedding_features` (top-`top_k`, by mean |SHAP|),
    `physical_feature_shap` (all 4 named physical features), the aggregate
    `physical_contribution_ratio`, the raw `shap_values` matrix (for an
    `.npz` export), and `dependence_plot_paths` (empty unless
    `dependence_plot_dir` is given, since rendering plots is the only step
    here with a side effect - tests exercising the rest of this bundle don't
    need to write files).
    """
    shap_values = compute_tree_shap(model, X)
    groups = split_shap_by_group(shap_values, embedding_dim, physical_dim)

    dependence_plot_paths = {}
    if dependence_plot_dir is not None:
        X_physical = X[:, embedding_dim:]
        dependence_plot_paths = save_dependence_plots(
            X_physical, groups["physical"], dependence_plot_dir
        )

    return {
        "top_embedding_features": top_k_embedding_features(
            groups["embedding"], k=top_k
        ),
        "physical_feature_shap": physical_feature_shap_summary(groups["physical"]),
        "physical_contribution_ratio": aggregate_physical_contribution_ratio(
            shap_values, embedding_dim
        ),
        "shap_values": shap_values,
        "dependence_plot_paths": dependence_plot_paths,
    }
