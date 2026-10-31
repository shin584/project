"""Final NT Testset behavior through the agreed public workflow boundary."""

import csv
import hashlib
import json
import subprocess
import sys

import pytest

from cas12a.nt_workflow import FitFixture, ModelSource, SplitConfig
from cas12a.test.test_nt_preflight import artifact
from cas12a.test.test_nt_workflow import synthetic_table
from cas12a.test.test_nt_workflow import workflow as make_workflow


def selected_run(tmp_path, tiny_nt, *, constant_labels=False):
    table = synthetic_table(15)
    if constant_labels:
        from cas12a.data import TARGET_COLUMN

        table[TARGET_COLUMN] = 50
    workflow = make_workflow(tmp_path, table, split=SplitConfig.fixture(8, 2, 5))
    run = workflow.prepare(run_id="final-fixture")
    run = workflow.preflight(run, model=ModelSource.fixture(tiny_nt))
    run = workflow.fit(run, fixture=FitFixture(validation_losses=(0.1, 0.2, 0.2, 0.2)))
    run = workflow.export_selected(run)
    return workflow, workflow.persist_export(run)


def final_outputs(run, monkeypatch, values):
    """Control the external model's Testset outputs; validation reload stays real."""
    import torch
    from transformers import EsmForSequenceClassification, EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained(run.local_dir / "selected_export")
    partition = artifact(run, "partition.json")
    by_id = {row["source_row"]: row for row in artifact(run, "data.json")["rows"]}
    outputs = {
        tuple(
            tokenizer(by_id[i]["sequence"], padding="max_length", max_length=12)[
                "input_ids"
            ]
        ): value
        for i, value in zip(partition["partitions"]["test"], values, strict=True)
    }
    original = EsmForSequenceClassification.forward

    def forward(model, input_ids=None, **kwargs):
        result = original(model, input_ids=input_ids, **kwargs)
        for index, tokens in enumerate(input_ids.tolist()):
            if tuple(tokens) in outputs:
                result.logits[index, 0] = torch.tensor(outputs[tuple(tokens)])
        return result

    monkeypatch.setattr(EsmForSequenceClassification, "forward", forward)


def test_final_evaluation_preserves_targets_views_baseline_and_durable_identity(
    tmp_path, tiny_nt, monkeypatch
):
    workflow, run = selected_run(tmp_path, tiny_nt)
    final_outputs(run, monkeypatch, [-0.5, 0.0, 0.5, 1.0, 1.5])
    frozen = {
        name: (run.local_dir / name).read_bytes()
        for name in (
            "execution.json",
            "training.json",
            "history.json",
            "partition.json",
            "export.json",
        )
    }

    run = workflow.evaluate(run)

    assert run.status["state"] == "final_evaluation_validated"
    assert run.status["final_evaluation_completed"] is False
    assert run.status["model_completed"] is False
    directory = run.local_dir / "final_evaluation"
    metrics = json.loads((directory / "metrics.json").read_bytes())
    report = json.loads((directory / "evaluation.json").read_bytes())
    with (directory / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    partition = artifact(run, "partition.json")
    by_id = {row["source_row"]: row for row in artifact(run, "data.json")["rows"]}
    ids = partition["partitions"]["test"]
    assert [int(row["source_row"]) for row in predictions] == ids
    assert [float(row["raw_prediction"]) for row in predictions] == [
        -0.5,
        0,
        0.5,
        1,
        1.5,
    ]
    assert [float(row["bounded_efficiency"]) for row in predictions] == [
        0,
        0,
        0.5,
        1,
        1,
    ]
    for row, identity in zip(predictions, ids):
        assert row["sequence_sha256"] == by_id[identity]["sequence_sha256"]
        assert json.loads(row["source_identity"])["library_id"] == f"target-{identity}"
        assert float(row["label"]) == by_id[identity]["target"]
        assert (
            float(row["baseline_train_mean"])
            == partition["baseline_train_mean"]["value"]
        )
    assert metrics["baseline_train_mean"]["mae"] > 0
    assert metrics["baseline_train_mean"]["mse"] > 0
    assert metrics["raw"]["mae"] == pytest.approx(0.6)
    assert metrics["raw"]["mse"] == pytest.approx(0.492)
    assert metrics["clipped"]["mae"] == pytest.approx(0.4)
    assert metrics["clipped"]["mse"] == pytest.approx(0.272)
    assert metrics["baseline_train_mean"]["mae"] == pytest.approx(0.3225)
    assert metrics["baseline_train_mean"]["mse"] == pytest.approx(0.12715625)
    assert metrics["baseline_train_mean"]["spearman"] is None
    assert metrics["diagnostics"]["raw"]["out_of_range_count"] == 2
    assert metrics["diagnostics"]["raw"]["all_finite"] is True
    assert metrics["diagnostics"]["raw"]["constant"] is False
    assert {"spearman", "pearson", "mae", "mse"} <= metrics["raw"].keys()
    assert {"spearman", "pearson", "mae", "mse"} <= metrics["clipped"].keys()
    assert "not directly comparable" in report["comparison_policy"]
    assert "overlap" in report["comparison_policy"]
    assert report["validated"] is True
    for name, content in frozen.items():
        assert (run.local_dir / name).read_bytes() == content
    assert not (run.durable_dir / "final_evaluation").exists()

    run = workflow.persist_evaluation(run)

    assert run.status["final_evaluation_completed"] is True
    assert run.status["model_completed"] is False  # fixture, never a production result
    receipt = artifact(run, "durable_evaluation.json")
    assert receipt["selected_checkpoint_sha256"] == report["selected_checkpoint_sha256"]
    for name, expected in receipt["files"].items():
        assert (
            hashlib.sha256(
                (run.durable_dir / "final_evaluation" / name).read_bytes()
            ).hexdigest()
            == expected
        )
    assert workflow.verify_durable_evaluation(run).status == run.status
    with pytest.raises((ValueError, FileExistsError)):
        workflow.evaluate(run)


@pytest.mark.parametrize("values", [[2, 3, 4, 5, 6], [-2, -3, -4, -5, -6]])
def test_clipping_only_constant_is_reportable_without_retuning(
    tmp_path, tiny_nt, monkeypatch, values
):
    workflow, run = selected_run(tmp_path, tiny_nt)
    final_outputs(run, monkeypatch, values)
    run = workflow.evaluate(run)
    metrics = json.loads((run.local_dir / "final_evaluation/metrics.json").read_bytes())
    assert metrics["diagnostics"]["raw_collapse"] is False
    assert metrics["diagnostics"]["clipped_only_constant"] is True
    assert metrics["clipped"]["spearman"] is None
    assert metrics["clipped"]["spearman_undefined_reason"] == "constant predictions"
    assert metrics["clipped"]["mae"] == pytest.approx(0.34 if values[0] > 0 else 0.66)
    assert metrics["clipped"]["mse"] == pytest.approx(0.242 if values[0] > 0 else 0.562)
    assert workflow.persist_evaluation(run).status["final_evaluation_completed"] is True


def test_constant_raw_model_keeps_errors_but_cannot_complete(
    tmp_path, tiny_nt, monkeypatch
):
    workflow, run = selected_run(tmp_path, tiny_nt)
    final_outputs(run, monkeypatch, [0.25] * 5)
    with pytest.raises(ValueError, match="raw model collapse"):
        workflow.evaluate(run)
    metrics = json.loads((run.local_dir / "final_evaluation/metrics.json").read_bytes())
    assert metrics["raw"]["mae"] == pytest.approx(0.47)
    assert metrics["raw"]["mse"] == pytest.approx(0.2945)
    assert metrics["raw"]["spearman"] is None
    assert metrics["raw"]["pearson_undefined_reason"] == "constant predictions"
    assert metrics["diagnostics"]["raw_collapse"] is True
    assert metrics["diagnostics"]["clipped_only_constant"] is False
    assert metrics["validated"] is False
    status = artifact(run, "status.json")
    assert status["state"] == "final_evaluation_failed"
    assert status["model_completed"] is False
    assert status["final_evaluation_completed"] is False
    with pytest.raises(ValueError):
        workflow.persist_evaluation(run)


def test_constant_labels_keep_error_metrics_and_explicit_undefined_correlations(
    tmp_path, tiny_nt, monkeypatch
):
    workflow, run = selected_run(tmp_path, tiny_nt, constant_labels=True)
    final_outputs(run, monkeypatch, [-0.5, 0, 0.5, 1, 1.5])
    run = workflow.evaluate(run)
    metrics = json.loads((run.local_dir / "final_evaluation/metrics.json").read_bytes())
    for name in ("raw", "clipped", "baseline_train_mean"):
        assert metrics[name]["spearman"] is None
        assert metrics[name]["pearson"] is None
        assert metrics[name]["spearman_undefined_reason"] == "constant labels"
    assert metrics["raw"]["mae"] == pytest.approx(0.6)
    assert metrics["raw"]["mse"] == pytest.approx(0.5)
    assert metrics["clipped"]["mae"] == pytest.approx(0.4)
    assert metrics["clipped"]["mse"] == pytest.approx(0.2)
    assert metrics["baseline_train_mean"]["mae"] == 0
    assert metrics["validated"] is True


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_outputs_are_json_safe_failed_evidence(
    tmp_path, tiny_nt, monkeypatch, nonfinite
):
    workflow, run = selected_run(tmp_path, tiny_nt)
    final_outputs(run, monkeypatch, [0, 0.25, nonfinite, 0.75, 1])
    with pytest.raises(ValueError, match="nonfinite"):
        workflow.evaluate(run)
    directory = run.local_dir / "final_evaluation"

    def reject_nonstandard(value):
        pytest.fail(f"nonstandard JSON constant: {value}")

    metrics = json.loads(
        (directory / "metrics.json").read_bytes(), parse_constant=reject_nonstandard
    )
    report = json.loads(
        (directory / "evaluation.json").read_bytes(), parse_constant=reject_nonstandard
    )
    assert report["validated"] is False
    assert metrics["diagnostics"]["raw"]["nonfinite_count"] == 1
    assert metrics["diagnostics"]["raw"]["all_finite"] is False
    assert metrics["diagnostics"]["clipped"]["all_finite"] is False
    assert metrics["raw"]["mse"] is None
    assert (
        metrics["raw"]["spearman_undefined_reason"] == "nonfinite labels or predictions"
    )
    assert artifact(run, "status.json")["final_evaluation_completed"] is False


@pytest.mark.parametrize(
    "name",
    [
        "selected_export/model.safetensors",
        "selected_export/vocab.txt",
        "partition.json",
    ],
)
def test_final_evaluation_rejects_changed_input_identity(tmp_path, tiny_nt, name):
    workflow, run = selected_run(tmp_path, tiny_nt)
    path = run.local_dir / name
    path.write_bytes(b"changed identity")
    with pytest.raises((ValueError, OSError)):
        workflow.evaluate(run)
    assert not (run.local_dir / "final_evaluation").exists()
    assert artifact(run, "status.json")["model_completed"] is False


@pytest.mark.parametrize("name", ["metrics.json", "predictions.csv", "evaluation.json"])
def test_changed_final_evidence_cannot_publish(tmp_path, tiny_nt, monkeypatch, name):
    workflow, run = selected_run(tmp_path, tiny_nt)
    final_outputs(run, monkeypatch, [0, 0.25, 0.5, 0.75, 1])
    run = workflow.evaluate(run)
    (run.local_dir / "final_evaluation" / name).write_bytes(b"changed result")
    with pytest.raises((ValueError, OSError)):
        workflow.persist_evaluation(run)
    assert artifact(run, "status.json")["final_evaluation_completed"] is False
    assert not (run.durable_dir / "final_evaluation").exists()


def test_training_and_reload_never_infer_on_testset_and_final_never_fits(
    tmp_path, tiny_nt, monkeypatch
):
    import torch
    from transformers import EsmForSequenceClassification, EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained(tiny_nt)
    # Use the source column name explicitly; its spreadsheet name contains spaces.
    from cas12a.data import SEQUENCE_COLUMN

    input_identities = {
        tuple(
            tokenizer(sequence, padding="max_length", max_length=12)["input_ids"]
        ): identity
        for identity, sequence in enumerate(synthetic_table(15)[SEQUENCE_COLUMN])
    }
    seen, supervised = set(), set()
    original = EsmForSequenceClassification.forward

    def forward(model, input_ids=None, **kwargs):
        identities = {input_identities[tuple(tokens)] for tokens in input_ids.tolist()}
        seen.update(identities)
        if kwargs.get("labels") is not None:
            supervised.update(identities)
        return original(model, input_ids=input_ids, **kwargs)

    monkeypatch.setattr(EsmForSequenceClassification, "forward", forward)
    workflow, run = selected_run(tmp_path, tiny_nt)
    partition = artifact(run, "partition.json")
    assert seen == set(
        partition["partitions"]["train"] + partition["partitions"]["validation"]
    )
    assert supervised.isdisjoint(partition["partitions"]["test"])
    assert partition["baseline_train_mean"]["value"] == pytest.approx(0.6875)
    assert partition["baseline_train_mean"]["training_rows"] == 8
    seen.clear()
    supervised.clear()

    def forbid_fit(*args, **kwargs):
        pytest.fail("final evaluation must not create an optimizer or fit again")

    monkeypatch.setattr(torch.optim, "AdamW", forbid_fit)
    run = workflow.evaluate(run)
    assert seen == set(
        partition["partitions"]["test"] + partition["partitions"]["validation"]
    )
    assert not supervised
    assert run.status["model_completed"] is False


def test_pure_final_metrics_need_no_heavyweight_runtime():
    script = """
import json
import math
import sys
from cas12a.nt_evaluation import prediction_metrics
report = prediction_metrics([0, 0.5, 1], [0.25, 0.25, 0.25], 0.5)
assert report['raw']['mae'] == 5 / 12
assert report['raw']['mse'] == 11 / 48
assert report['raw']['spearman'] is None
ranked = prediction_metrics([0, 0.5, 1], [2, 1, 0], 0.5)
assert ranked['validated']  # negative correlation is reportable, with no gate
assert math.isclose(ranked['raw']['spearman'], -1)
assert math.isclose(ranked['raw']['pearson'], -1)
assert math.isclose(ranked['clipped']['spearman'], -math.sqrt(3) / 2)
assert math.isclose(ranked['clipped']['pearson'], -math.sqrt(3) / 2)
assert math.isclose(ranked['raw']['mse'], 7 / 4)
assert math.isclose(ranked['clipped']['mse'], 3 / 4)
json.dumps(report, allow_nan=False)
assert not {'torch', 'transformers', 'tensorflow', 'cas12a.nt_training', 'cas12a.nt_runtime'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
