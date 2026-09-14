"""Step 5 runner: Case Study selection + standardized artifact export (issue #17).

Assembles the final `model_analysis_arrays.npz` / `model_analysis_summary.json`
pair from this Testset's predictions plus every prior step's own output
artifact (issues #11-#16). Requires (in the working directory):

- `test_metadata.csv` (issue #1, `test_dataset_verifier.py`)
- `ism_delta.npz` (issue #11, `step3_ism.py`)
- `mismatch_delta.npz` + `mismatch_complex_metadata.json` (issue #12, `step3_mismatch.py`)
- `ablation_results.json` (issue #13, `step4_ablation.py`)
- `shap_results.json` + `shap_values.npz` (issue #14, `step4_shap.py`)
- `attention_rollout_matrix.npz` + `attention_rollout_summary.json` (issue #16, `step4_attention_rollout.py`)

Integrated Gradients (issue #15) has no standalone runner/artifact of its
own - it is only ever needed for the (at most 15) selected Case Study
samples here, so this script computes it directly for just those sequences
rather than requiring a full-Testset IG artifact no other step produces.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from analysis_export import (
    assemble_case_studies,
    build_model_analysis_summary,
    export_artifacts,
)
from case_study_selection import select_case_studies
from integrated_gradients import (
    integrated_gradients_model_a,
    integrated_gradients_model_b_tokens,
    l1_normalize,
    project_model_b_attributions_to_nucleotides,
)
from model_a_wrapper import Model_A_Predictor
from model_b_wrapper import Model_B_Predictor
from model_b_xai_wrapper import Model_B_XAIPredictor
from shap_analysis import (
    EMBEDDING_DIM,
    PHYSICAL_DIM,
    compute_tree_shap,
    split_shap_by_group,
)

TEST_METADATA_PATH = "test_metadata.csv"
MODEL_A_WEIGHT_PATH = "best_model_fold1.pth"
NT_MODEL_DIR = "./NT_sacas9_fintuned_model"
XGB_MODEL_PATH = "hybrid_xgb_model.json"

ISM_ARRAYS_PATH = "ism_delta.npz"
MISMATCH_ARRAYS_PATH = "mismatch_delta.npz"
MISMATCH_COMPLEX_METADATA_PATH = "mismatch_complex_metadata.json"
ABLATION_RESULTS_PATH = "ablation_results.json"
SHAP_RESULTS_PATH = "shap_results.json"
SHAP_VALUES_PATH = "shap_values.npz"
ATTENTION_ROLLOUT_ARRAYS_PATH = "attention_rollout_matrix.npz"
ATTENTION_ROLLOUT_SUMMARY_PATH = "attention_rollout_summary.json"

OUTPUT_JSON_PATH = "model_analysis_summary.json"
OUTPUT_NPZ_PATH = "model_analysis_arrays.npz"

# Derived from the same constants integrated_gradients.py/shap_analysis.py
# already define for these dimensions, rather than re-hardcoding them here
# and risking the two definitions drifting apart.
MODEL_B_ARCHITECTURE = {
    "tokenizer": "6-mer",
    "tokens": Model_B_XAIPredictor.EXPECTED_SEQ_LEN,
    "hidden_dim": Model_B_XAIPredictor.EXPECTED_HIDDEN_DIM,
    "flattened_dim": EMBEDDING_DIM,
    "physical_dim": PHYSICAL_DIM,
    "total_feature_dim": EMBEDDING_DIM + PHYSICAL_DIM,
}

ARRAY_AXIS_DEFINITIONS = {
    "ism_delta_model_a": [
        "sample_id (0~513)",
        "position (0~35)",
        "alternative_base (lexicographical 3 bases)",
    ],
    "ism_delta_model_b": [
        "sample_id (0~513)",
        "position (0~35)",
        "alternative_base (lexicographical 3 bases)",
    ],
    "mismatch_single_delta_model_a": [
        "sample_id (0~513)",
        "position (0~35)",
        "alternative_base (3 bases)",
    ],
    "mismatch_single_delta_model_b": [
        "sample_id (0~513)",
        "position (0~35)",
        "alternative_base (3 bases)",
    ],
    "mismatch_single_min_model_a": ["sample_id (0~513)", "position (0~35)"],
    "mismatch_single_min_model_b": ["sample_id (0~513)", "position (0~35)"],
    "mismatch_complex_model_a": ["sample_id (0~513)", "scenario_id (0~14)"],
    "mismatch_complex_model_b": ["sample_id (0~513)", "scenario_id (0~14)"],
    "shap_values": [
        "sample_id (0~513)",
        "feature_index (0~8963, embedding then physical)",
    ],
    "attention_rollout": [
        "sample_id (0~513)",
        "row_position (0~35)",
        "column_position (0~35)",
    ],
}


def _require(path: str, produced_by: str) -> str:
    # Same "{path} not found. {script}를 먼저 실행해 주세요." phrasing every other
    # stepN_*.py runner in this repo uses (step1_verify.py, step3_ism.py, ...).
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. {produced_by}를 먼저 실행해 주세요."
        )
    return path


def _load_npz_as_dict(path: str) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _dataset_sha256(sequences: pd.Series) -> str:
    # Same whole-column hashing convention as test_dataset_verifier.py, so
    # this checksum is directly comparable to the one printed at Testset
    # isolation time.
    return hashlib.sha256(pd.util.hash_pandas_object(sequences).values).hexdigest()


def build_metadata(meta_df: pd.DataFrame) -> dict:
    return {
        "testset_version": "SaCas9_v4_independent_10pct",
        "total_samples": len(meta_df),
        "sha256_checksum": _dataset_sha256(meta_df["sequence"]),
        "model_b_architecture": MODEL_B_ARCHITECTURE,
        "array_axis_definitions": ARRAY_AXIS_DEFINITIONS,
    }


def build_global_evaluation(ablation_results: dict, shap_results: dict) -> dict:
    # plan_realize.md section 4 nests `paired_bootstrap_95ci` inside
    # `ablation_results`, but `ablation.py`/`step4_ablation.py` (issue #13)
    # write it as a sibling key in `ablation_results.json` - re-nest here so
    # the final export matches the documented schema shape.
    return {
        "ablation_results": {
            **ablation_results["ablation_results"],
            "paired_bootstrap_95ci": ablation_results["paired_bootstrap_95ci"],
        },
        "aggregate_physical_shap_contribution_ratio": shap_results[
            "physical_contribution_ratio"
        ],
    }


def compute_case_study_ig(
    predictor_a: Model_A_Predictor,
    xai_predictor_b: Model_B_XAIPredictor,
    selection: list[dict],
    sequences: list[str],
) -> dict:
    """Integrated Gradients for just the selected Case Study sequences (see module docstring)."""
    selected_indices = [case["index"] for case in selection]
    selected_sequences = [sequences[i] for i in selected_indices]

    ig_a_norm = l1_normalize(
        integrated_gradients_model_a(predictor_a, selected_sequences)
    )
    ig_b_tokens = integrated_gradients_model_b_tokens(
        xai_predictor_b, selected_sequences
    )
    ig_b_norm = l1_normalize(project_model_b_attributions_to_nucleotides(ig_b_tokens))

    return {
        idx: {
            "model_a_norm_attr_36bp": ig_a_norm[j],
            "model_b_phase4_projected_norm_attr_36bp": ig_b_norm[j],
        }
        for j, idx in enumerate(selected_indices)
    }


def compute_case_study_physical_shap(
    predictor_b: Model_B_Predictor,
    selection: list[dict],
    sequences: list[str],
    shap_values_full: np.ndarray | None,
) -> np.ndarray:
    """Per-selected-sample physical-feature SHAP, shape `(len(selection), 4)`.

    Reuses `shap_values.npz`'s full `(514, 8964)` matrix when available
    (the same values Tree SHAP over the whole Testset already produced);
    otherwise computes Tree SHAP directly for just the selected sequences -
    cheap at 10 samples, unlike the full 514-sample run issue #14 owns.
    """
    selected_indices = [case["index"] for case in selection]

    if shap_values_full is not None:
        physical_shap = split_shap_by_group(shap_values_full)["physical"]
        return physical_shap[selected_indices]

    selected_sequences = [sequences[i] for i in selected_indices]
    embeddings = predictor_b.extract_nt_embeddings(selected_sequences)
    physical = predictor_b.compute_physical_features(selected_sequences)
    X_selected = np.hstack([embeddings, physical])
    shap_values = compute_tree_shap(predictor_b.xgb_model, X_selected)
    return split_shap_by_group(shap_values, physical_dim=PHYSICAL_DIM)["physical"]


def main():
    print("=== [Step 5] Case Study Selection + Standardized Artifact Export ===")

    _require(TEST_METADATA_PATH, "test_dataset_verifier.py")
    meta_df = pd.read_csv(TEST_METADATA_PATH)
    sequences = meta_df["sequence"].tolist()
    true_scores = meta_df["true_score"].to_numpy()
    print(f"Loaded testset: {len(sequences)} samples.")

    print("\nInitializing Model A / Model B predictors...")
    predictor_a = Model_A_Predictor(weight_path=MODEL_A_WEIGHT_PATH)
    predictor_b = Model_B_Predictor(
        nt_model_dir=NT_MODEL_DIR, xgb_model_path=XGB_MODEL_PATH
    )

    print("\nScoring the Testset with both models (no cross-model rescaling)...")
    pred_a = np.asarray(predictor_a.predict(sequences), dtype=np.float64)
    pred_b = np.asarray(predictor_b.predict(sequences), dtype=np.float64)
    error_a = np.abs(true_scores - pred_a)
    error_b = np.abs(true_scores - pred_b)

    print(
        "\nSelecting Case Studies (Primary -> Secondary fallback -> "
        "Reverse Primary -> Reverse Secondary fallback -> Concordant)..."
    )
    selection = select_case_studies(error_a, error_b)
    print(
        f" - Selected {len(selection)} Case Studies: "
        f"{[c['case_id'] for c in selection]}"
    )

    print("\nLoading prior steps' precomputed artifacts...")
    ism_arrays = _load_npz_as_dict(_require(ISM_ARRAYS_PATH, "step3_ism.py"))
    mismatch_arrays = _load_npz_as_dict(
        _require(MISMATCH_ARRAYS_PATH, "step3_mismatch.py")
    )
    with open(_require(MISMATCH_COMPLEX_METADATA_PATH, "step3_mismatch.py")) as f:
        complex_mismatch_metadata = json.load(f)
    with open(_require(ABLATION_RESULTS_PATH, "step4_ablation.py")) as f:
        ablation_results = json.load(f)
    with open(_require(SHAP_RESULTS_PATH, "step4_shap.py")) as f:
        shap_results = json.load(f)
    shap_values_full = None
    if os.path.exists(SHAP_VALUES_PATH):
        shap_values_full = _load_npz_as_dict(SHAP_VALUES_PATH)["shap_values"]
    attention_arrays = _load_npz_as_dict(
        _require(ATTENTION_ROLLOUT_ARRAYS_PATH, "step4_attention_rollout.py")
    )
    with open(
        _require(ATTENTION_ROLLOUT_SUMMARY_PATH, "step4_attention_rollout.py")
    ) as f:
        attention_rollout_summary = json.load(f)

    print(
        "\nComputing per-Case-Study physical values, physical SHAP, and Integrated Gradients..."
    )
    xai_predictor_b = Model_B_XAIPredictor(nt_model_dir=NT_MODEL_DIR)
    selected_indices = [case["index"] for case in selection]
    physical_values_selected = predictor_b.compute_physical_features(
        [sequences[i] for i in selected_indices]
    )
    physical_values_by_index = dict(zip(selected_indices, physical_values_selected))

    physical_shap_selected = compute_case_study_physical_shap(
        predictor_b, selection, sequences, shap_values_full
    )
    physical_shap_by_index = dict(zip(selected_indices, physical_shap_selected))

    ig_by_index = compute_case_study_ig(
        predictor_a, xai_predictor_b, selection, sequences
    )

    attention_rollout_info = {
        "token_matrix_dim": attention_rollout_summary["token_matrix_dim"],
        "projected_nucleotide_matrix_dim": attention_rollout_summary[
            "projected_nucleotide_matrix_dim"
        ],
        "note": attention_rollout_summary["note"],
    }

    case_studies = assemble_case_studies(
        selection,
        meta_df=meta_df,
        pred_a=pred_a,
        error_a=error_a,
        pred_b=pred_b,
        error_b=error_b,
        physical_values_by_index=physical_values_by_index,
        physical_shap_by_index=physical_shap_by_index,
        ig_by_index=ig_by_index,
        attention_rollout_info=attention_rollout_info,
    )

    summary = build_model_analysis_summary(
        metadata=build_metadata(meta_df),
        global_evaluation=build_global_evaluation(ablation_results, shap_results),
        complex_mismatch_metadata=complex_mismatch_metadata,
        case_studies=case_studies,
    )

    arrays = {**ism_arrays, **mismatch_arrays, **attention_arrays}
    if shap_values_full is not None:
        arrays["shap_values"] = shap_values_full

    print(f"\nWriting {OUTPUT_JSON_PATH} and {OUTPUT_NPZ_PATH}...")
    export_artifacts(summary, arrays, OUTPUT_JSON_PATH, OUTPUT_NPZ_PATH)
    print("Done.")

    print("\n=== Step 5 (Case Study Export) Execution Completed ===")


if __name__ == "__main__":
    main()
