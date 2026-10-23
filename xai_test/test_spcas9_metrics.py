import json

import numpy as np
import pytest

from SpCas9.spcas9_data import HEAD_NAMES, output_name
from SpCas9.spcas9_metrics import full_metrics, predictions_by_head

# --- metrics ---


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_full_metrics_turns_an_undefined_correlation_into_none():
    metrics = full_metrics(np.linspace(0, 1, 12), np.full(12, 0.5))
    assert metrics["spearman"] is None
    assert metrics["pearson"] is None
    json.dumps(metrics, allow_nan=False)


def test_full_metrics_reports_n_spearman_pearson_mae_and_mse():
    y_true = np.linspace(0.0, 1.0, 20)

    metrics = full_metrics(y_true, y_true + 0.1)

    assert metrics["n"] == 20
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.1)
    assert metrics["mse"] == pytest.approx(0.01)


def test_full_metrics_leaves_a_correlation_out_below_the_minimum_n():
    y_true = np.linspace(0.0, 1.0, 9)

    metrics = full_metrics(y_true, y_true)

    assert metrics["n"] == 9
    assert metrics["spearman"] is None
    assert metrics["pearson"] is None
    assert metrics["mae"] == pytest.approx(0.0)


def test_full_metrics_of_an_empty_group_is_strict_json():
    metrics = full_metrics(np.array([]), np.array([]))

    assert metrics["n"] == 0
    assert metrics["mae"] is None
    json.dumps(metrics, allow_nan=False)


# --- model output plumbing (no model loaded) ---


def test_predictions_are_keyed_by_output_name_not_position():
    """A transposition of the two reordered heads would fail here.

    The spreadsheet puts `eSpCas9(1.1)` before `SpCas9-HF.1` and the
    checkpoints emit them the other way round, so the output layer names are
    swapped here and each output given its own value.
    """
    names = [output_name(head) for head in HEAD_NAMES]
    hf, esp = names.index("Output_SpCas9_HF_1"), names.index("Output_eSpCas9_1_1_")
    names[hf], names[esp] = names[esp], names[hf]
    outputs = {name: np.full((2, 1), float(i)) for i, name in enumerate(names)}

    preds = predictions_by_head(outputs)

    assert preds["SpCas9-HF.1"].tolist() == [float(esp)] * 2
    assert preds["eSpCas9(1.1)"].tolist() == [float(hf)] * 2
    assert list(preds) == list(HEAD_NAMES)


def test_predictions_by_head_flattens_each_output_to_one_dimension():
    outputs = {output_name(head): np.zeros((3, 1)) for head in HEAD_NAMES}

    preds = predictions_by_head(outputs)

    assert all(value.shape == (3,) for value in preds.values())


def test_predictions_by_head_rejects_a_missing_head():
    outputs = {output_name(head): np.zeros((1, 1)) for head in HEAD_NAMES[:-1]}

    with pytest.raises(KeyError, match="evoCas9"):
        predictions_by_head(outputs)
