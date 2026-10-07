"""Model B predictions + physical features for the full Testset (issue #19).

`model_analysis_summary.json` only carries Model B's prediction and its 4
physical features (MFE, ΔG, Tm, GC) for the 15 Case Studies. The Model B
explanation dashboard (`visualization/plan_draft.md`) needs them for all 514
Testset samples, so this exports one small table keyed by `sample_id`.

It is a CSV rather than another `.npz` key: one row per sample with 6 scalar
columns is a table, not a multi-dimensional tensor, and the dashboard reads
it straight into Pandas.

Integrated Gradients stays limited to the Case Studies (out of scope here).
"""

import numpy as np
import pandas as pd
from analysis_export import PHYSICAL_FEATURE_KEYS

EXPORT_COLUMNS = ("sample_id", "pred_raw", *PHYSICAL_FEATURE_KEYS)

# Case Study values were produced by an earlier Colab GPU run; a fresh NT
# forward pass can differ by float32/cuDNN noise, never by a real amount.
CASE_STUDY_MATCH_ATOL = 1e-4


def build_model_b_testset_table(
    meta_df: pd.DataFrame, pred_b: np.ndarray, physical: np.ndarray
) -> pd.DataFrame:
    """One row per Testset sample: `sample_id`, `pred_raw`, then `mfe`/`dg`/`tm`/`gc`.

    `physical` is `Model_B_Predictor.compute_physical_features`'s `(n, 4)`
    output, in `PHYSICAL_FEATURE_KEYS` order.
    """
    n = len(meta_df)
    pred_b = np.asarray(pred_b, dtype=np.float64).reshape(-1)
    physical = np.asarray(physical, dtype=np.float64)

    if len(pred_b) != n:
        raise ValueError(f"pred_b has {len(pred_b)} rows, Testset has {n}")
    if physical.shape != (n, len(PHYSICAL_FEATURE_KEYS)):
        raise ValueError(
            f"physical has shape {physical.shape}, expected "
            f"({n}, {len(PHYSICAL_FEATURE_KEYS)})"
        )
    sample_ids = meta_df["sample_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_ids, np.arange(n)):
        raise ValueError(f"sample_id must be exactly 0..{n - 1} in row order")
    if np.isnan(pred_b).any() or np.isnan(physical).any():
        raise ValueError("pred_b/physical contain NaN")

    table = pd.DataFrame({"sample_id": sample_ids, "pred_raw": pred_b})
    for j, key in enumerate(PHYSICAL_FEATURE_KEYS):
        table[key] = physical[:, j]
    return table


def check_case_studies_match(
    table: pd.DataFrame, case_studies: list[dict], atol: float = CASE_STUDY_MATCH_ATOL
) -> None:
    """Raise `ValueError` listing every Case Study field that disagrees with `table`.

    `case_studies` is `model_analysis_summary.json`'s `case_studies` list.
    """
    rows = table.set_index("sample_id")
    mismatches = []
    for case in case_studies:
        sample_id = case["sample_id"]
        if sample_id not in rows.index:
            mismatches.append(f"sample_id={sample_id}: not in table")
            continue
        row = rows.loc[sample_id]
        expected = {
            "pred_raw": case["model_b"]["pred_raw"],
            **case["per_sample_physical_values"],
        }
        for key, value in expected.items():
            if not np.isclose(row[key], value, rtol=0.0, atol=atol):
                mismatches.append(
                    f"sample_id={sample_id} {key}: table={row[key]!r}, "
                    f"case study={value!r}"
                )
    if mismatches:
        raise ValueError(
            "Model B Testset export disagrees with the Case Studies:\n"
            + "\n".join(mismatches)
        )


def write_model_b_testset_table(table: pd.DataFrame, path) -> None:
    table.to_csv(path, index=False, columns=list(EXPORT_COLUMNS))


def load_model_b_testset_table(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sample_id": np.int64})[list(EXPORT_COLUMNS)]
