import json
import os

import numpy as np
import pandas as pd
import pytest
from analysis_export import (
    LARGE_ARRAY_KEYS,
    PHYSICAL_FEATURE_KEYS,
    PHYSICAL_SHAP_NEGATIVE_INTERPRETATION,
    PHYSICAL_SHAP_POSITIVE_INTERPRETATION,
    assemble_case_studies,
    build_case_study_entry,
    build_model_analysis_summary,
    export_artifacts,
    physical_shap_interpretation,
    to_jsonable,
    validate_summary_schema,
)
from case_study_selection import select_case_studies


def test_physical_shap_interpretation_positive_when_it_reduces_error():
    # true=1.0, pred_b=0.9 (error 0.1). Without the physical contribution
    # (sum=+0.3), the prediction would have been 0.6 (error 0.4) - worse -
    # so the physical contribution moved the prediction toward the truth.
    result = physical_shap_interpretation(
        true_score=1.0, pred_b=0.9, physical_shap_values=[0.1, 0.1, 0.05, 0.05]
    )
    assert result == PHYSICAL_SHAP_POSITIVE_INTERPRETATION


def test_physical_shap_interpretation_negative_when_it_increases_error():
    # true=1.0, pred_b=0.5 (error 0.5). Without the physical contribution
    # (sum=-0.4), the prediction would have been 0.9 (error 0.1) - much
    # better - so the physical contribution moved the prediction away from
    # the truth.
    result = physical_shap_interpretation(
        true_score=1.0, pred_b=0.5, physical_shap_values=[-0.1, -0.1, -0.1, -0.1]
    )
    assert result == PHYSICAL_SHAP_NEGATIVE_INTERPRETATION


def _sample_case_entry(
    index=0, case_id="DISCORDANT_P01", case_type="Primary Discordant"
):
    case = {"index": index, "case_id": case_id, "case_type": case_type}
    return build_case_study_entry(
        case,
        sample_id=42,
        original_id=418,
        sequence="A" * 36,
        true_score=0.82,
        pred_a=0.58,
        error_a=0.24,
        pred_b=0.801,
        error_b=0.019,
        physical_values={"mfe": -12.4, "dg": -28.6, "tm": 68.5, "gc": 55.5},
        physical_shap={"mfe": 0.0412, "dg": 0.0785, "tm": -0.0051, "gc": 0.0120},
        ig_model_a_norm_attr_36bp=np.linspace(0, 1, 36),
        ig_model_b_norm_attr_36bp=np.linspace(1, 0, 36),
        attention_rollout_info={
            "token_matrix_dim": [6, 6],
            "projected_nucleotide_matrix_dim": [36, 36],
            "note": "Represents internal information dependency, not physical cleavage proof",
        },
    )


def test_build_case_study_entry_matches_the_documented_schema_shape():
    entry = _sample_case_entry()

    assert entry["case_id"] == "DISCORDANT_P01"
    assert entry["sample_id"] == 42
    assert entry["original_id"] == 418
    assert entry["case_type"] == "Primary Discordant"
    assert entry["model_a"] == {"pred_raw": 0.58, "error": 0.24}
    assert entry["model_b"] == {"pred_raw": 0.801, "error": 0.019}
    assert set(entry["per_sample_physical_values"]) == set(PHYSICAL_FEATURE_KEYS)
    assert set(entry["per_sample_physical_shap"]) == set(PHYSICAL_FEATURE_KEYS) | {
        "interpretation"
    }
    assert len(entry["integrated_gradients"]["model_a_norm_attr_36bp"]) == 36
    assert (
        len(entry["integrated_gradients"]["model_b_phase4_projected_norm_attr_36bp"])
        == 36
    )
    assert entry["attention_rollout"]["projected_nucleotide_matrix_dim"] == [36, 36]
    # No raw numpy types should have leaked into the entry - it must survive
    # a plain json.dumps unchanged.
    json.dumps(entry)


def test_build_case_study_entry_converts_numpy_arrays_to_plain_lists():
    entry = _sample_case_entry()

    assert isinstance(entry["integrated_gradients"]["model_a_norm_attr_36bp"], list)
    assert all(
        isinstance(v, float)
        for v in entry["integrated_gradients"]["model_a_norm_attr_36bp"]
    )


def test_to_jsonable_handles_nested_numpy_scalars_and_arrays():
    data = {
        "a": np.float64(1.5),
        "b": np.array([1, 2, 3]),
        "c": [np.int64(4), {"d": np.float32(2.5)}],
    }

    result = to_jsonable(data)

    assert result == {"a": 1.5, "b": [1, 2, 3], "c": [4, {"d": pytest.approx(2.5)}]}
    json.dumps(result)


def _synthetic_meta_df(n=10):
    return pd.DataFrame(
        {
            "sample_id": range(n),
            "original_id": [100 + i for i in range(n)],
            "sequence": [f"SEQ{i:02d}" * 4 + "AAAA" for i in range(n)],
            "true_score": np.linspace(0, 1, n),
        }
    )


def test_assemble_case_studies_wires_each_selected_sample_correctly():
    n = 10
    meta_df = _synthetic_meta_df(n)
    e_a = np.array([100, 95, 90, 85, 80, 10, 9, 8, 7, 6], dtype=np.float64)
    e_b = np.array([1, 2, 40, 38, 35, 9, 8, 7, 6, 5], dtype=np.float64)
    pred_a = meta_df["true_score"].to_numpy() - e_a * 0.001
    pred_b = meta_df["true_score"].to_numpy() - e_b * 0.001

    selection = select_case_studies(e_a, e_b)
    selected_indices = [case["index"] for case in selection]
    # Only the selected indices ever get physical values/SHAP/IG computed -
    # there is no full-population array for these three (see
    # assemble_case_studies's docstring).
    physical_values_by_index = {
        idx: np.array([1.0, 2.0, 3.0, 4.0]) for idx in selected_indices
    }
    physical_shap_by_index = {
        idx: np.array([0.1, 0.1, -0.05, 0.05]) for idx in selected_indices
    }
    ig_by_index = {
        idx: {
            "model_a_norm_attr_36bp": np.zeros(36),
            "model_b_phase4_projected_norm_attr_36bp": np.zeros(36),
        }
        for idx in selected_indices
    }
    attention_rollout_info = {
        "token_matrix_dim": [6, 6],
        "projected_nucleotide_matrix_dim": [36, 36],
        "note": "note",
    }

    entries = assemble_case_studies(
        selection,
        meta_df=meta_df,
        pred_a=pred_a,
        error_a=e_a,
        pred_b=pred_b,
        error_b=e_b,
        physical_values_by_index=physical_values_by_index,
        physical_shap_by_index=physical_shap_by_index,
        ig_by_index=ig_by_index,
        attention_rollout_info=attention_rollout_info,
    )

    assert len(entries) == len(selection)
    for case, entry in zip(selection, entries):
        idx = case["index"]
        assert entry["sample_id"] == int(meta_df.iloc[idx]["sample_id"])
        assert entry["original_id"] == int(meta_df.iloc[idx]["original_id"])
        assert entry["sequence"] == meta_df.iloc[idx]["sequence"]
        assert entry["model_a"]["error"] == pytest.approx(e_a[idx])
        assert entry["model_b"]["error"] == pytest.approx(e_b[idx])
        assert entry["case_id"] == case["case_id"]
        assert entry["case_type"] == case["case_type"]


def _minimal_valid_summary(n_case_studies=1):
    case_studies = [_sample_case_entry(index=i) for i in range(n_case_studies)]
    return build_model_analysis_summary(
        metadata={"testset_version": "v", "total_samples": 514},
        global_evaluation={"ablation_results": {}},
        complex_mismatch_metadata=[{"scenario_index": 0}],
        case_studies=case_studies,
    )


def test_validate_summary_schema_accepts_a_well_formed_summary():
    summary = _minimal_valid_summary()
    validate_summary_schema(summary)  # must not raise


def test_validate_summary_schema_rejects_missing_top_level_keys():
    summary = _minimal_valid_summary()
    del summary["global_evaluation"]

    with pytest.raises(ValueError, match="missing required keys"):
        validate_summary_schema(summary)


def test_validate_summary_schema_rejects_more_than_ten_case_studies():
    summary = _minimal_valid_summary(n_case_studies=11)

    with pytest.raises(ValueError, match="exceeding the cap"):
        validate_summary_schema(summary)


def test_validate_summary_schema_rejects_a_leaked_large_array_key():
    summary = _minimal_valid_summary()
    summary["global_evaluation"]["ism_delta_model_a"] = [1, 2, 3]

    with pytest.raises(ValueError, match="must not appear"):
        validate_summary_schema(summary)


def test_validate_summary_schema_allows_array_axis_definitions_name_collision():
    # metadata.array_axis_definitions (plan_realize.md section 4) legitimately
    # reuses LARGE_ARRAY_KEYS names as keys, but its values are plain lists of
    # axis-label strings, not real tensor data - must not be flagged as a leak.
    summary = _minimal_valid_summary()
    summary["metadata"]["array_axis_definitions"] = {
        "ism_delta_model_a": ["sample_id (0~513)", "position (0~35)"],
        "shap_values": ["sample_id (0~513)", "feature_index (0~8963)"],
        "attention_rollout": ["sample_id (0~513)", "row (0~35)", "column (0~35)"],
    }

    validate_summary_schema(summary)  # must not raise


def test_validate_summary_schema_rejects_non_jsonable_numpy_leakage():
    summary = _minimal_valid_summary()
    summary["global_evaluation"]["stray_array"] = np.array([1, 2, 3])

    with pytest.raises(TypeError):
        validate_summary_schema(summary)


def test_export_artifacts_writes_json_and_npz_and_keeps_large_arrays_out_of_json(
    tmp_path,
):
    summary = _minimal_valid_summary(n_case_studies=2)
    arrays = {
        "ism_delta_model_a": np.zeros((514, 36, 3)),
        "ism_delta_model_b": np.zeros((514, 36, 3)),
        "attention_rollout": np.zeros((514, 36, 36)),
    }
    json_path = str(tmp_path / "model_analysis_summary.json")
    npz_path = str(tmp_path / "model_analysis_arrays.npz")

    export_artifacts(summary, arrays, json_path, npz_path)

    with open(json_path) as f:
        written_summary = json.load(f)
    assert set(written_summary) >= {
        "metadata",
        "global_evaluation",
        "complex_mismatch_metadata",
        "case_studies",
    }
    # The full (514, 36, 3)/(514, 36, 36) tensors must never appear in the
    # JSON at all - "attention_rollout" is exempt from this substring check
    # only because a Case Study entry legitimately carries a small
    # dims+note metadata dict under that same name (see
    # build_case_study_entry), not the matrix itself.
    for key in LARGE_ARRAY_KEYS - {"attention_rollout"}:
        assert key not in json.dumps(written_summary)
    for entry in written_summary["case_studies"]:
        assert set(entry["attention_rollout"]) == {
            "token_matrix_dim",
            "projected_nucleotide_matrix_dim",
            "note",
        }

    loaded_npz = np.load(npz_path)
    assert set(loaded_npz.keys()) == set(arrays.keys())
    np.testing.assert_array_equal(
        loaded_npz["ism_delta_model_a"], arrays["ism_delta_model_a"]
    )


def test_export_artifacts_does_not_write_files_when_schema_invalid(tmp_path):
    summary = _minimal_valid_summary(n_case_studies=11)  # invalid: > 10
    json_path = str(tmp_path / "model_analysis_summary.json")
    npz_path = str(tmp_path / "model_analysis_arrays.npz")

    with pytest.raises(ValueError):
        export_artifacts(summary, {}, json_path, npz_path)

    assert not os.path.exists(json_path)
    assert not os.path.exists(npz_path)
