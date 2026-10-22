import numpy as np
import pandas as pd
import pytest
from grouped_model_b import CHECKPOINT_DIRNAME, PREDICTION_COLUMNS
from grouped_model_b_eval import (
    HEADS,
    METADATA_PATH,
    PARITY_ATOL,
    PREDICTIONS_PATH,
    PRODUCTION_LABEL,
    REFERENCE_NUMBERS,
    build_results,
    check_parity,
    compare_to_reference,
    evaluate_predictions,
    load_predictions,
    run_parity_check,
    sequence_level,
)
from grouped_split import BASE_DIR, normalize_sequences
from paired_bootstrap import compute_metrics

EXPECTED_ROWS = 514
EXPECTED_SEQUENCES = 180


def _toy_predictions() -> pd.DataFrame:
    """12 rows: 4 sequences x 3 replicates, one replicate per cell_line_block.

    `pred_all_rows` ranks the rows perfectly within every block and also ranks
    the sequence means perfectly; `pred_seq_mean` reverses the order of the
    sequence means.
    """
    sequences = ["AAAA", "CCCC", "GGGG", "TTTT"]
    rows = []
    for b, block in enumerate(("L1", "L2", "L3")):
        for s, seq in enumerate(sequences):
            true = 0.1 * (s + 1) + 0.01 * b
            rows.append(
                {
                    "sample_id": len(rows),
                    "original_id": 100 + len(rows),
                    # Lower-case/padded variants must still group with their sequence.
                    "sequence": f" {seq.lower()} " if (b, s) == (1, 0) else seq,
                    "cell_line_block": block,
                    "true_score": true,
                    "pred_all_rows": true + 0.05,
                    "pred_seq_mean": 1.0 - 0.1 * s,
                }
            )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def test_sequence_level_averages_labels_and_predictions_per_normalized_sequence():
    df = _toy_predictions()

    seq = sequence_level(df, "pred_all_rows")

    assert seq["normalized_sequence"].tolist() == ["AAAA", "CCCC", "GGGG", "TTTT"]
    np.testing.assert_allclose(seq["true_score"], [0.11, 0.21, 0.31, 0.41])
    np.testing.assert_allclose(seq["pred"], [0.16, 0.26, 0.36, 0.46])


def test_evaluate_predictions_reports_row_sequence_and_per_block_metrics():
    df = _toy_predictions()

    result = evaluate_predictions(df, "pred_all_rows")

    assert result["row_level"] == pytest.approx(
        compute_metrics(df["true_score"].to_numpy(), df["pred_all_rows"].to_numpy())
    )
    assert result["row_level"]["spearman"] == pytest.approx(1.0)
    assert result["row_level"]["mae"] == pytest.approx(0.05)
    assert result["n_rows"] == 12
    assert result["sequence_level"] == {
        "n_sequences": 4,
        "spearman": pytest.approx(1.0),
    }
    assert result["per_cell_line_block"] == {
        "L1": {"n": 4, "spearman": pytest.approx(1.0)},
        "L2": {"n": 4, "spearman": pytest.approx(1.0)},
        "L3": {"n": 4, "spearman": pytest.approx(1.0)},
    }


def test_evaluate_predictions_sequence_level_differs_from_row_level():
    df = _toy_predictions()

    result = evaluate_predictions(df, "pred_seq_mean")

    # Sequence means are ranked in exactly reverse order...
    assert result["sequence_level"]["spearman"] == pytest.approx(-1.0)
    # ...and so is every block, since each block holds one replicate per sequence.
    for block in ("L1", "L2", "L3"):
        assert result["per_cell_line_block"][block]["spearman"] == pytest.approx(-1.0)
    # Row level mixes blocks, so it is negative but not exactly -1.
    assert -1.0 < result["row_level"]["spearman"] < 0.0


def test_compare_to_reference_reports_delta_and_ci_per_metric():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(size=200)
    good = y_true + rng.normal(scale=0.01, size=200)
    bad = rng.uniform(size=200)

    result = compare_to_reference(y_true, good, bad, n_resamples=300, seed=0)

    assert set(result) == {"spearman", "pearson", "mae", "mse"}
    for metric, entry in result.items():
        expected = (
            compute_metrics(y_true, good)[metric] - compute_metrics(y_true, bad)[metric]
        )
        assert entry["delta"] == pytest.approx(expected)
        lower, upper = entry["ci95"]
        assert lower <= entry["delta"] <= upper
    # Clearly better ranking: the whole Spearman CI is above zero; errors below.
    assert result["spearman"]["ci95"][0] > 0
    assert result["mae"]["ci95"][1] < 0


def test_compare_to_reference_is_deterministic_for_a_seed():
    y_true = np.linspace(0, 1, 50)
    a_preds = y_true + np.sin(np.arange(50)) * 0.1
    b_preds = y_true[::-1]

    first = compare_to_reference(y_true, a_preds, b_preds, n_resamples=100, seed=3)
    second = compare_to_reference(y_true, a_preds, b_preds, n_resamples=100, seed=3)

    assert first == second


def test_check_parity_passes_within_tolerance():
    expected = {"pred_all_rows": np.array([0.1, 0.5, 0.9])}
    actual = {"pred_all_rows": np.array([0.1, 0.5, 0.9]) + PARITY_ATOL / 2}

    result = check_parity(expected, actual)

    assert result["status"] == "passed"
    head = result["heads"]["pred_all_rows"]
    assert head["max_abs_diff"] == pytest.approx(PARITY_ATOL / 2)
    assert head["n_outside_atol"] == 0
    assert head["spearman"] == pytest.approx(1.0)


def test_check_parity_fails_when_any_head_is_outside_tolerance():
    expected = {
        "pred_all_rows": np.array([0.1, 0.5, 0.9]),
        "pred_seq_mean": np.array([0.2, 0.4, 0.6]),
    }
    actual = {
        "pred_all_rows": np.array([0.1, 0.5, 0.9]),
        "pred_seq_mean": np.array([0.2, 0.4, 0.6 + 2 * PARITY_ATOL]),
    }

    result = check_parity(expected, actual)

    assert result["status"] == "failed"
    assert result["heads"]["pred_all_rows"]["n_outside_atol"] == 0
    assert result["heads"]["pred_seq_mean"]["n_outside_atol"] == 1


def test_build_results_labels_production_leaky_and_compares_each_head_to_it():
    df = _toy_predictions()
    production = df["true_score"].to_numpy()[::-1].copy()
    parity = {"status": "skipped", "reason": "no checkpoint"}

    results = build_results(df, production, parity, n_resamples=50, seed=0)

    models = results["models"]
    assert set(models) == {*HEADS, "production"}
    assert models["production"]["label"] == PRODUCTION_LABEL
    assert models["production"]["leaky"] is True
    for head in HEADS:
        assert models[head]["leaky"] is False
        assert models[head] == {
            **models[head],
            **evaluate_predictions(df, head),
        }
    assert models["production"]["row_level"]["spearman"] == pytest.approx(
        compute_metrics(df["true_score"].to_numpy(), production)["spearman"]
    )
    assert set(results["vs_production"]) == set(HEADS)
    assert results["vs_production"]["pred_all_rows"]["spearman"]["delta"] > 0
    assert results["bootstrap"] == {
        "n_resamples": 50,
        "seed": 0,
        "ci": 0.95,
        "level": "row",
    }
    assert results["parity"] == parity
    assert results["old_reference"] == REFERENCE_NUMBERS
    assert results["testset"] == {"rows": 12, "sequences": 4}


def test_build_results_rejects_misaligned_production_predictions():
    df = _toy_predictions()

    with pytest.raises(ValueError, match="production"):
        build_results(df, np.zeros(len(df) - 1), {"status": "skipped"}, n_resamples=10)


def test_load_predictions_rejects_rows_that_disagree_with_the_testset(tmp_path):
    df = _toy_predictions()
    predictions_path = tmp_path / "test_predictions.csv"
    metadata_path = tmp_path / "test_metadata.csv"
    df.to_csv(predictions_path, index=False)
    df.drop(columns=["pred_all_rows", "pred_seq_mean"]).to_csv(
        metadata_path, index=False
    )

    assert len(load_predictions(predictions_path, metadata_path)) == 12

    tampered = df.copy()
    tampered.loc[3, "true_score"] += 0.5
    tampered.to_csv(predictions_path, index=False)
    with pytest.raises(ValueError, match="true_score"):
        load_predictions(predictions_path, metadata_path)


def test_real_colab_predictions_cover_the_grouped_testset():
    if not PREDICTIONS_PATH.exists():
        pytest.skip(f"{PREDICTIONS_PATH} not present yet (Colab run from #37)")

    df = load_predictions(PREDICTIONS_PATH, METADATA_PATH)

    assert len(df) == EXPECTED_ROWS
    assert normalize_sequences(df["sequence"]).nunique() == EXPECTED_SEQUENCES
    assert set(df["cell_line_block"]) == {"L1", "L2", "L3"}


def test_parity_against_real_colab_outputs():
    checkpoint = BASE_DIR / CHECKPOINT_DIRNAME
    if not checkpoint.is_dir():
        pytest.skip(f"{checkpoint} not downloaded yet; parity check skipped")

    df = load_predictions(PREDICTIONS_PATH, METADATA_PATH)
    result = run_parity_check(df)

    assert result["status"] == "passed", result
