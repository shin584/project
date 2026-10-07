import numpy as np
import pandas as pd
import pytest
from analysis_export import PHYSICAL_FEATURE_KEYS
from model_b_testset_export import (
    EXPORT_COLUMNS,
    build_model_b_testset_table,
    check_case_studies_match,
    load_model_b_testset_table,
    write_model_b_testset_table,
)


def _meta_df(n=5):
    return pd.DataFrame({"sample_id": np.arange(n), "sequence": ["A" * 36] * n})


def _physical(n=5):
    # (n, 4) in Model_B_Predictor.compute_physical_features's (mfe, dg, tm, gc) order.
    return np.column_stack(
        [
            np.linspace(-12, -8, n),
            np.linspace(-80, -70, n),
            np.linspace(60, 70, n),
            np.linspace(40, 60, n),
        ]
    ).astype(np.float32)


def test_table_is_keyed_by_sample_id_with_pred_and_physical_columns():
    pred_b = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    physical = _physical()

    table = build_model_b_testset_table(_meta_df(), pred_b, physical)

    assert list(table.columns) == list(EXPORT_COLUMNS)
    assert EXPORT_COLUMNS == ("sample_id", "pred_raw", *PHYSICAL_FEATURE_KEYS)
    assert table["sample_id"].tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(table["pred_raw"], pred_b)
    for j, key in enumerate(PHYSICAL_FEATURE_KEYS):
        np.testing.assert_allclose(table[key], physical[:, j])


def test_table_rejects_row_count_mismatch():
    with pytest.raises(ValueError, match="pred_b"):
        build_model_b_testset_table(_meta_df(5), np.zeros(4), _physical(5))
    with pytest.raises(ValueError, match="physical"):
        build_model_b_testset_table(_meta_df(5), np.zeros(5), _physical(4))


def test_table_rejects_wrong_physical_width():
    with pytest.raises(ValueError, match="physical"):
        build_model_b_testset_table(_meta_df(5), np.zeros(5), np.zeros((5, 3)))


def test_table_rejects_non_contiguous_sample_ids():
    meta_df = _meta_df(5)
    meta_df["sample_id"] = [0, 1, 2, 4, 5]
    with pytest.raises(ValueError, match="sample_id"):
        build_model_b_testset_table(meta_df, np.zeros(5), _physical(5))


def test_table_rejects_nan():
    pred_b = np.zeros(5)
    pred_b[2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build_model_b_testset_table(_meta_df(5), pred_b, _physical(5))


def _case_study(sample_id, table):
    row = table.loc[table["sample_id"] == sample_id].iloc[0]
    return {
        "sample_id": sample_id,
        "model_b": {"pred_raw": float(row["pred_raw"]), "error": 0.1},
        "per_sample_physical_values": {k: float(row[k]) for k in PHYSICAL_FEATURE_KEYS},
    }


def test_case_studies_match_passes_on_identical_values():
    table = build_model_b_testset_table(
        _meta_df(), np.linspace(0.1, 0.9, 5), _physical()
    )
    case_studies = [_case_study(1, table), _case_study(4, table)]

    check_case_studies_match(table, case_studies)


def test_case_studies_match_tolerates_float_noise():
    table = build_model_b_testset_table(
        _meta_df(), np.linspace(0.1, 0.9, 5), _physical()
    )
    case = _case_study(2, table)
    case["model_b"]["pred_raw"] += 1e-6

    check_case_studies_match(table, [case])


def test_case_studies_match_reports_every_mismatching_field():
    table = build_model_b_testset_table(
        _meta_df(), np.linspace(0.1, 0.9, 5), _physical()
    )
    case = _case_study(3, table)
    case["model_b"]["pred_raw"] += 0.05
    case["per_sample_physical_values"]["tm"] += 1.0

    with pytest.raises(ValueError) as excinfo:
        check_case_studies_match(table, [case])

    message = str(excinfo.value)
    assert "sample_id=3" in message
    assert "pred_raw" in message
    assert "tm" in message
    assert "mfe" not in message


def test_case_studies_match_rejects_unknown_sample_id():
    table = build_model_b_testset_table(
        _meta_df(), np.linspace(0.1, 0.9, 5), _physical()
    )
    case = _case_study(0, table)
    case["sample_id"] = 99

    with pytest.raises(ValueError, match="sample_id=99"):
        check_case_studies_match(table, [case])


def test_write_then_load_round_trips_exactly(tmp_path):
    table = build_model_b_testset_table(
        _meta_df(), np.linspace(0.1, 0.9, 5, dtype=np.float32), _physical()
    )
    path = tmp_path / "model_b_testset_predictions.csv"

    write_model_b_testset_table(table, path)
    loaded = load_model_b_testset_table(path)

    pd.testing.assert_frame_equal(loaded, table)
