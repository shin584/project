"""Standardized artifact export: Case Study bundling + schema assembly (Step 5, issue #17).

This is the pipeline's final assembly step (see plan_realize.md section 4 /
issue #9's Implementation Decisions): it does not compute new explainability
signal itself (that's Steps 2-4, issues #11-#16) - it selects the 15
representative Case Study samples (`case_study_selection.py`), bundles each
with its per-sample detail, and assembles the standard two-artifact export:

- `model_analysis_arrays.npz`: every large multi-dimensional tensor
  (ISM/mismatch/attention/SHAP arrays for the full Testset).
- `model_analysis_summary.json`: metadata, `global_evaluation`,
  `complex_mismatch_metadata`, and the `case_studies` list - no large arrays,
  per issue #9's AC24/AC25.
"""

import json

import numpy as np

# Order matches Model_B_Predictor.compute_physical_features's hstack order
# (mfe, dg, tm, gc) - see model_b_wrapper.py / shap_analysis.py.
PHYSICAL_FEATURE_KEYS = ("mfe", "dg", "tm", "gc")

PHYSICAL_SHAP_POSITIVE_INTERPRETATION = (
    "Physical features' SHAP contribution shifted Model B's prediction "
    "toward the ground truth."
)
PHYSICAL_SHAP_NEGATIVE_INTERPRETATION = (
    "Physical features' SHAP contribution shifted Model B's prediction "
    "away from the ground truth."
)

REQUIRED_TOP_LEVEL_KEYS = (
    "metadata",
    "global_evaluation",
    "complex_mismatch_metadata",
    "case_studies",
)

MAX_CASE_STUDIES = 15

# Large multi-dimensional arrays that must live only in the .npz - never
# embedded as a JSON key anywhere in the summary (issue #9 AC24/#17 AC5).
LARGE_ARRAY_KEYS = frozenset(
    {
        "ism_delta",
        "ism_delta_model_a",
        "ism_delta_model_b",
        "mismatch_single_delta",
        "mismatch_single_delta_model_a",
        "mismatch_single_delta_model_b",
        "mismatch_single_min",
        "mismatch_single_min_model_a",
        "mismatch_single_min_model_b",
        "mismatch_complex",
        "mismatch_complex_model_a",
        "mismatch_complex_model_b",
        "attention_rollout",
        "shap_values",
    }
)


def to_jsonable(value):
    """Recursively convert numpy scalars/arrays to plain Python/JSON-safe types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def physical_shap_interpretation(
    true_score: float, pred_b: float, physical_shap_values
) -> str:
    """Direction-aware interpretation of a sample's physical-feature SHAP contribution.

    Rather than a fixed narrative string, this compares Model B's actual
    error against the counterfactual error it would have had without the
    physical features' contribution (`pred_b` minus the sum of their SHAP
    values, since SHAP contributions are additive around the model's
    expected value): if removing the physical contribution would have
    pushed the prediction further from `true_score`, the physical features
    are reported as having moved the prediction *toward* the truth, and
    vice versa.
    """
    physical_shap_sum = float(np.sum(physical_shap_values))
    pred_without_physical = pred_b - physical_shap_sum
    error_with_physical = abs(true_score - pred_b)
    error_without_physical = abs(true_score - pred_without_physical)

    if error_with_physical <= error_without_physical:
        return PHYSICAL_SHAP_POSITIVE_INTERPRETATION
    return PHYSICAL_SHAP_NEGATIVE_INTERPRETATION


def build_case_study_entry(
    case: dict,
    *,
    sample_id: int,
    original_id: int,
    sequence: str,
    true_score: float,
    pred_a: float,
    error_a: float,
    pred_b: float,
    error_b: float,
    physical_values: dict,
    physical_shap: dict,
    ig_model_a_norm_attr_36bp,
    ig_model_b_norm_attr_36bp,
    attention_rollout_info: dict,
) -> dict:
    """Bundle one selected Case Study's full per-sample explainability detail.

    `case` is one entry from `case_study_selection.select_case_studies`
    (`case_id`/`case_type`); everything else is that same sample's already-
    computed data. `physical_values`/`physical_shap` are dicts keyed by
    `PHYSICAL_FEATURE_KEYS`. Large arrays (the attention matrix itself) are
    deliberately not embedded here - only `attention_rollout_info`'s
    dimensions/note - per issue #9 AC24/#17 AC5; the IG attribution arrays
    are a fixed, small 36-length vector so they're embedded directly, same
    as plan_realize.md section 4's schema example.
    """
    interpretation = physical_shap_interpretation(
        true_score, pred_b, [physical_shap[k] for k in PHYSICAL_FEATURE_KEYS]
    )

    return {
        "case_id": case["case_id"],
        "sample_id": int(sample_id),
        "original_id": int(original_id),
        "sequence": sequence,
        "true_score": float(true_score),
        "model_a": {"pred_raw": float(pred_a), "error": float(error_a)},
        "model_b": {"pred_raw": float(pred_b), "error": float(error_b)},
        "case_type": case["case_type"],
        "per_sample_physical_values": {
            k: float(physical_values[k]) for k in PHYSICAL_FEATURE_KEYS
        },
        "per_sample_physical_shap": {
            **{k: float(physical_shap[k]) for k in PHYSICAL_FEATURE_KEYS},
            "interpretation": interpretation,
        },
        "integrated_gradients": {
            "model_a_norm_attr_36bp": to_jsonable(
                np.asarray(ig_model_a_norm_attr_36bp)
            ),
            "model_b_phase4_projected_norm_attr_36bp": to_jsonable(
                np.asarray(ig_model_b_norm_attr_36bp)
            ),
        },
        "attention_rollout": dict(attention_rollout_info),
    }


def assemble_case_studies(
    selection: list[dict],
    *,
    meta_df,
    pred_a: np.ndarray,
    error_a: np.ndarray,
    pred_b: np.ndarray,
    error_b: np.ndarray,
    physical_values_by_index: dict,
    physical_shap_by_index: dict,
    ig_by_index: dict,
    attention_rollout_info: dict,
) -> list[dict]:
    """Build the full `case_studies` list from a `select_case_studies` selection.

    `pred_a`/`error_a`/`pred_b`/`error_b` are full-Testset-length arrays
    (residual/quantile computation over the whole Testset is what selection
    itself needed), indexed by each case's `index`. `physical_values_by_index`
    /`physical_shap_by_index`/`ig_by_index` instead map a selected sample's
    index directly to its already-computed data (a 4-value
    `PHYSICAL_FEATURE_KEYS`-ordered array-like, and an
    `{"model_a_norm_attr_36bp": ..., "model_b_phase4_projected_norm_attr_36bp": ...}`
    dict, respectively) - physical values/SHAP/Integrated Gradients are only
    ever computed for the (at most 15) selected Case Study samples, never
    the full Testset, since that's all this export needs (issue #17's AC3
    scopes IG/attention detail to the Case Studies themselves), so there is
    no full-population array to index into for these three.
    """
    entries = []
    for case in selection:
        idx = case["index"]
        row = meta_df.iloc[idx]
        entries.append(
            build_case_study_entry(
                case,
                sample_id=row["sample_id"],
                original_id=row["original_id"],
                sequence=row["sequence"],
                true_score=row["true_score"],
                pred_a=pred_a[idx],
                error_a=error_a[idx],
                pred_b=pred_b[idx],
                error_b=error_b[idx],
                physical_values=dict(
                    zip(PHYSICAL_FEATURE_KEYS, physical_values_by_index[idx])
                ),
                physical_shap=dict(
                    zip(PHYSICAL_FEATURE_KEYS, physical_shap_by_index[idx])
                ),
                ig_model_a_norm_attr_36bp=ig_by_index[idx]["model_a_norm_attr_36bp"],
                ig_model_b_norm_attr_36bp=ig_by_index[idx][
                    "model_b_phase4_projected_norm_attr_36bp"
                ],
                attention_rollout_info=attention_rollout_info,
            )
        )
    return entries


def build_model_analysis_summary(
    metadata: dict,
    global_evaluation: dict,
    complex_mismatch_metadata: list,
    case_studies: list,
) -> dict:
    """Assemble the standard `model_analysis_summary.json` top-level dict."""
    return {
        "metadata": metadata,
        "global_evaluation": global_evaluation,
        "complex_mismatch_metadata": complex_mismatch_metadata,
        "case_studies": case_studies,
    }


def _is_leaked_array_value(value) -> bool:
    """Whether `value` looks like actual leaked tensor data, not descriptive metadata.

    An `ndarray` is always a leak. A `list`/`tuple` is only a leak if it
    isn't purely made of strings - real array data (or its post-
    `to_jsonable` nested-list form) is numbers/further nesting, whereas
    `metadata.array_axis_definitions` legitimately holds plain lists of
    human-readable axis-label strings under these same key names (per
    plan_realize.md section 4's schema) and must not be flagged.
    """
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, (list, tuple)):
        return not all(isinstance(item, str) for item in value)
    return False


def _find_large_array_keys(obj) -> set:
    """Recursively collect any `LARGE_ARRAY_KEYS` name whose *array-shaped* value appears anywhere in `obj`.

    A `LARGE_ARRAY_KEYS` name used as a key is only a violation when its
    value actually looks like leaked array data (see
    `_is_leaked_array_value`). It is legitimate for a Case Study entry's
    small `attention_rollout` *metadata* dict (dims + note, no matrix data -
    see `build_case_study_entry`) and for `metadata.array_axis_definitions`'s
    axis-label-string lists to reuse these same names, so neither is itself
    a violation.
    """
    found = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in LARGE_ARRAY_KEYS and _is_leaked_array_value(value):
                found.add(key)
            found |= _find_large_array_keys(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found |= _find_large_array_keys(item)
    return found


def validate_summary_schema(summary: dict) -> None:
    """Validate `summary` against the Section-4 export schema (issue #17 AC4/AC5).

    Checks: all required top-level keys present, `case_studies` count capped
    at `MAX_CASE_STUDIES`, no `LARGE_ARRAY_KEYS` name leaked into the JSON
    (they must live only in the `.npz`), and the summary is actually
    JSON-serializable (numpy scalars/arrays would raise `TypeError` here
    instead of silently producing malformed JSON).
    """
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in summary]
    if missing:
        raise ValueError(
            f"model_analysis_summary.json missing required keys: {missing}"
        )

    if len(summary["case_studies"]) > MAX_CASE_STUDIES:
        raise ValueError(
            f"case_studies has {len(summary['case_studies'])} entries, "
            f"exceeding the cap of {MAX_CASE_STUDIES}"
        )

    leaked = _find_large_array_keys(summary)
    if leaked:
        raise ValueError(
            f"Large array key(s) {sorted(leaked)} must not appear in "
            "model_analysis_summary.json - they belong only in model_analysis_arrays.npz"
        )

    json.dumps(summary)  # raises TypeError on any un-jsonified numpy leakage


def export_artifacts(
    summary: dict,
    arrays: dict,
    json_path: str,
    npz_path: str,
) -> None:
    """Validate and write the two standard export artifacts.

    `arrays` is everything destined for `model_analysis_arrays.npz` (full
    Testset ISM/mismatch/attention/SHAP tensors); `summary` is the
    JSON-only metadata + Case Studies dict. Validation runs before either
    file is written, so a schema violation never leaves a half-written pair
    on disk.
    """
    validate_summary_schema(summary)

    jsonable_summary = to_jsonable(summary)
    with open(json_path, "w") as f:
        json.dump(jsonable_summary, f, indent=2)

    np.savez(npz_path, **arrays)
