import numpy as np
import pytest

from xai_test.cas12a_NT_analysis.analyze import run_validation
from xai_test.cas12a_NT_analysis.nt_inputs import verify_inputs
from xai_test.cas12a_NT_analysis.nt_predict import (
    NativePredictor,
    installed_environment,
)
from xai_test.cas12a_NT_analysis.nt_run import Study
from xai_test.cas12a_NT_analysis.test.test_rollout import POLICY


@pytest.fixture
def rollout_predictions(inputs, tmp_path):
    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    study = Study.create(tmp_path / "study", fixture=True)
    predictions = run_validation(verified, study, "a", reserved_seconds=300)
    return study, NativePredictor.load(inputs.export), predictions


def test_rollout_roundtrip_recomputes_mass_and_respects_requested_source_order(
    rollout_predictions,
):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout
    from xai_test.cas12a_NT_analysis.nt_rollout_artifacts import (
        plan_rollout,
        read_rollout,
        save_rollout,
        save_rollout_plan,
    )

    study, predictor, predictions = rollout_predictions
    plan = plan_rollout(predictions, run_id="a", policy=POLICY)
    assert plan["ids"] == predictions["coverage"]
    study.start("a", "rollout", 300, installed_environment("cpu"))
    save_rollout_plan(study, "a", plan)
    study.record_execution(installed_environment("cpu"), {"validated": True})
    for row in predictions["rows"]:
        save_rollout(study, "a", compute_rollout(row, predictor, policy=POLICY))
    active = read_rollout(study, "a")
    assert active["coverage_complete"] is True
    assert active["complete"] is False
    with pytest.raises(ValueError, match="unresolved"):
        read_rollout(study, "a", require_complete=True)
    study.finish()
    ids = list(reversed(predictions["coverage"]))
    final = read_rollout(study, "a", source_ids=ids, require_complete=True)
    assert final["complete"] is True
    assert final["gpu_executed"] is False
    assert [item["row"]["source_row"] for item in final["results"]] == ids
    result = final["results"][0]
    np.testing.assert_allclose(
        result["arrays"]["sequence"].sum(1) + result["summary"]["removed_cls_mass"],
        1,
        atol=1e-12,
    )
    assert result["array_schema"]["full"]["axes"] == [
        "valid_query_token",
        "valid_key_token",
    ]


@pytest.fixture
def planned_rollout(rollout_predictions):
    from xai_test.cas12a_NT_analysis.nt_rollout_artifacts import (
        plan_rollout,
        save_rollout_plan,
    )

    study, predictor, predictions = rollout_predictions
    plan = plan_rollout(predictions, run_id="a", policy=POLICY)
    study.start("a", "rollout", 300, installed_environment("cpu"))
    save_rollout_plan(study, "a", plan)
    study.record_execution(installed_environment("cpu"), {"validated": True})
    return study, predictor, predictions, plan


@pytest.mark.parametrize(
    "field",
    [
        "summary",
        "array",
        "policy",
        "source",
        "layers",
        "heads",
        "versions",
        "parity",
        "tokens",
    ],
)
def test_rollout_rejects_inconsistent_saved_quality_and_provenance(
    planned_rollout, field
):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout
    from xai_test.cas12a_NT_analysis.nt_rollout_artifacts import save_rollout

    study, predictor, predictions, _ = planned_rollout
    result = compute_rollout(predictions["rows"][0], predictor, policy=POLICY)
    if field == "summary":
        result["summary"]["removed_cls_mass"][0] += 0.1
    elif field == "array":
        result["arrays"]["full"][1, 0] += 0.1
    elif field == "policy":
        result["policy"]["query_l1_threshold"] = 0.5
    elif field == "source":
        result["row"]["prediction_sha256"] = "b" * 64
    elif field in ("layers", "heads"):
        result["execution"][field] = 999
    elif field == "versions":
        result["execution"]["packages"]["transformers"] = "unknown"
    elif field == "parity":
        result["outputs"]["attention_enabled"] = 100.0
    else:
        result["tokens"]["valid_positions"] = list(range(1, 10))
    with pytest.raises(ValueError):
        save_rollout(study, "a", result)
    assert study.artifacts("a", kind="rollout") == []


def test_failed_attention_diagnostic_survives_reload_and_blocks_completion(
    planned_rollout,
):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout
    from xai_test.cas12a_NT_analysis.nt_rollout_artifacts import (
        read_rollout,
        save_rollout,
    )

    study, predictor, predictions, _ = planned_rollout

    def omit_attention(module, arguments, output):
        if output.attentions is not None:
            output.attentions = None

    handle = predictor.model.register_forward_hook(omit_attention)
    try:
        result = compute_rollout(predictions["rows"][0], predictor, policy=POLICY)
    finally:
        handle.remove()
    save_rollout(study, "a", result)
    study.finish()
    report = read_rollout(Study.reopen(study.store.root, receipt=study.receipt), "a")
    assert report["failed"] == predictions["coverage"][:1]
    assert report["accepted_coverage"] == []
    assert report["complete"] is False
    assert report["results"][0]["failure"]["stage"] == "attention"
    assert report["results"][0]["outputs"]["native"] is not None
    assert report["results"][0]["arrays"] is None


@pytest.mark.parametrize("fault", ["corrupt", "missing", "write"])
def test_rollout_storage_faults_never_acknowledge_success(
    planned_rollout, fault, monkeypatch
):
    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout
    from xai_test.cas12a_NT_analysis.nt_rollout_artifacts import (
        read_rollout,
        save_rollout,
    )

    study, predictor, predictions, _ = planned_rollout
    result = compute_rollout(predictions["rows"][0], predictor, policy=POLICY)
    if fault == "write":
        original = study.store.put

        def broken_put(name, data):
            if name.endswith(".json") and "/artifacts/rollout-" in name:
                raise OSError("storage failed after NPZ write")
            return original(name, data)

        monkeypatch.setattr(study.store, "put", broken_put)
        with pytest.raises(OSError, match="storage failed"):
            save_rollout(study, "a", result)
        assert not any(
            record["kind"] == "rollout"
            for record in study.status()["runs"]["a"]["files"].values()
        )
    else:
        save_rollout(study, "a", result)
        path = next(study.store.root.glob("runs/a/artifacts/rollout-*.npz"))
        if fault == "corrupt":
            path.write_bytes(b"corrupt fixture")
        else:
            path.unlink()
        with pytest.raises((ValueError, OSError)):
            read_rollout(study, "a")
    assert study.poisoned is True


def test_model_free_rollout_reload_without_model_packages_or_source_assets(
    planned_rollout, inputs, tmp_path
):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from xai_test.cas12a_NT_analysis.nt_rollout import compute_rollout
    from xai_test.cas12a_NT_analysis.nt_rollout_artifacts import save_rollout

    study, predictor, predictions, _ = planned_rollout
    for row in predictions["rows"]:
        save_rollout(study, "a", compute_rollout(row, predictor, policy=POLICY))
    study.finish()
    inputs.export.rename(tmp_path / "unavailable-export")
    inputs.source.unlink()
    inputs.manifest.unlink()
    study.store.put("runs/a/artifacts/rollout-orphan.npz", b"uncommitted")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(study.receipt))
    code = """
import builtins, json, sys
original_import = builtins.__import__
def model_free(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'transformers'):
        raise AssertionError('model import attempted: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = model_free
from xai_test.cas12a_NT_analysis.nt_run import Study
from xai_test.cas12a_NT_analysis.nt_rollout_artifacts import read_rollout
study = Study.reopen(sys.argv[1], receipt=json.load(open(sys.argv[2])))
report = read_rollout(study, 'a', require_complete=True)
assert report['complete'] and not report['gpu_executed']
assert len(report['results']) == 4
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(study.store.root), str(receipt)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
