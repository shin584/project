import numpy as np
import pytest

from cas12a.nt_data import digest
from xai_test.cas12a_NT_analysis.nt_artifacts import prediction_rows, read_results
from xai_test.cas12a_NT_analysis.nt_inputs import encode_sequences, verify_inputs
from xai_test.cas12a_NT_analysis.nt_run import Study
from xai_test.cas12a_NT_analysis.test.test_ig import POLICY, scalar_predictor


@pytest.fixture
def ig_predictions(inputs, tmp_path):
    from xai_test.cas12a_NT_analysis.nt_ig import PROTOCOL

    predictor = scalar_predictor(inputs)
    verified = verify_inputs(
        inputs.source, inputs.manifest, inputs.export, inputs.receipt, fixture=True
    )
    sequences = [row["sequence"] for row in verified.rows]
    rows = prediction_rows(
        verified.rows,
        predictor.predict(sequences),
        encode_sequences(predictor.tokenizer, sequences, 4105),
    )
    identity = {
        **verified.identity,
        "protocol": {"ig": PROTOCOL},
        "code_sha256": "a" * 64,
    }
    study = Study.create(tmp_path / "study", fixture=True)
    study.register(
        "a",
        identity,
        snapshots={
            f"selected_export/{name}": (inputs.export / name).read_bytes()
            for name in ("vocab.txt", "config.json")
        },
    )
    study.start("a", "predictions", 300, {"device": "cpu"})
    study.save_artifact(
        "predictions",
        {"schema": 1, "identity": identity, "rows": rows},
        [row["source_row"] for row in rows],
    )
    study.finish()
    return study, predictor, rows


def test_ig_plan_and_shard_roundtrip_reconstruct_outputs_by_source_id(ig_predictions):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import (
        plan_ig,
        read_ig,
        save_ig,
        save_ig_plan,
    )

    study, predictor, rows = ig_predictions
    plan = plan_ig(
        read_results(study, "a"),
        run_id="a",
        policy=POLICY,
        probe_ids=[rows[1]["source_row"], rows[0]["source_row"]],
    )
    study.start("a", "ig", 300, {"device": "cpu"})
    save_ig_plan(study, "a", plan)
    study.record_execution(
        {"device": "cpu"}, {"validated": True, "injected_cpu_model": True}
    )
    result = compute_ig(rows[0], predictor, policy=POLICY)
    save_ig(study, "a", result)
    partial = read_ig(study, "a")
    assert partial["coverage"] == [rows[0]["source_row"]]
    assert partial["missing"] == [rows[1]["source_row"]]
    assert partial["complete"] is False
    result2 = compute_ig(rows[1], predictor, policy=POLICY)
    save_ig(study, "a", result2)
    assert read_ig(study, "a")["coverage_complete"] is True
    with pytest.raises(ValueError, match="unresolved"):
        read_ig(study, "a", require_complete=True)
    study.finish()
    final = read_ig(
        study,
        "a",
        source_ids=[rows[1]["source_row"], rows[0]["source_row"]],
        require_complete=True,
    )
    assert final["complete"] is True
    assert [item["row"]["source_row"] for item in final["results"]] == [
        rows[1]["source_row"],
        rows[0]["source_row"],
    ]
    np.testing.assert_array_equal(
        final["results"][0]["arrays"]["signed_mass"], result2["arrays"]["signed_mass"]
    )
    assert final["results"][0]["summary"]["reconstructed_output"] == pytest.approx(
        0.375, abs=1e-6
    )
    sidecar = study.artifacts("a", kind="ig")[0][1]
    assert sidecar["plan_sha256"] == digest(plan)
    assert sidecar["arrays"]["signed_mass"]["axes"] == ["attempt", "token"]


@pytest.fixture
def planned_ig(ig_predictions):
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import plan_ig, save_ig_plan

    study, predictor, rows = ig_predictions
    plan = plan_ig(
        read_results(study, "a"),
        run_id="a",
        policy=POLICY,
        probe_ids=[rows[0]["source_row"]],
    )
    study.start("a", "ig", 300, {"device": "cpu"})
    save_ig_plan(study, "a", plan)
    study.record_execution(
        {"device": "cpu"}, {"validated": True, "injected_cpu_model": True}
    )
    return study, predictor, rows, plan


@pytest.mark.parametrize(
    "field",
    ["summary", "quality", "policy", "source", "execution", "array", "reference"],
)
def test_ig_writer_rejects_inconsistent_derived_or_execution_fields(planned_ig, field):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import save_ig

    study, predictor, rows, _ = planned_ig
    result = compute_ig(rows[0], predictor, policy=POLICY)
    if field == "summary":
        result["summary"]["dna_signed_mass"][0] += 1
    elif field == "quality":
        result["attempts"][-1]["quality"]["signed_residual"] += 1
    elif field == "policy":
        result["policy"]["near_zero"] = 1
    elif field == "source":
        result["row"]["prediction_sha256"] = "b" * 64
    elif field == "execution":
        result["execution"]["autocast"] = True
    elif field == "reference":
        result["reference"]["shape"][1] = 999
    else:
        result["arrays"]["signed_mass"][0, 1] = np.nan
    with pytest.raises(ValueError):
        save_ig(study, "a", result)
    assert study.artifacts("a", kind="ig") == []


@pytest.mark.parametrize("mode", ["disconnected", "endpoint"])
def test_failed_ig_diagnostics_survive_reload_and_never_count_as_success(
    planned_ig, inputs, mode
):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import read_ig, save_ig

    study, predictor, rows, _ = planned_ig
    if mode == "disconnected":
        predictor = scalar_predictor(inputs, mode="disconnected")
    else:
        # Break saved/native parity without changing the frozen policy.
        import torch

        with torch.no_grad():
            predictor.model.embedding.weight.fill_(10)
        # The changed native endpoint must fail before integration.
    result = compute_ig(rows[0], predictor, policy=POLICY)
    assert result["accepted"] is False
    save_ig(study, "a", result)
    study.finish()
    report = read_ig(study, "a")
    assert report["coverage_complete"] is True
    assert report["complete"] is False
    assert report["accepted_coverage"] == []
    assert report["failed"] == [rows[0]["source_row"]]
    assert report["results"][0]["failure"] == result["failure"]
    with pytest.raises(ValueError, match="failed"):
        read_ig(study, "a", require_complete=True)


@pytest.mark.parametrize("failure", ["corrupt", "missing", "npz_write", "json_write"])
def test_ig_storage_failures_poison_manager_and_cannot_count_uncommitted_files(
    planned_ig, monkeypatch, failure
):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import read_ig, save_ig

    study, predictor, rows, _ = planned_ig
    result = compute_ig(rows[0], predictor, policy=POLICY)
    if failure.endswith("write"):
        extension = ".npz" if failure == "npz_write" else ".json"
        original = study.store.put

        def broken_put(name, data):
            if "/artifacts/ig-" in name and name.endswith(extension):
                raise OSError("external IG storage failed")
            return original(name, data)

        monkeypatch.setattr(study.store, "put", broken_put)
        with pytest.raises(OSError, match="storage failed"):
            save_ig(study, "a", result)
        assert not any(
            record["kind"] == "ig"
            for record in study.status()["runs"]["a"]["files"].values()
        )
    else:
        save_ig(study, "a", result)
        path = next(study.store.root.glob("runs/a/artifacts/ig-*.npz"))
        if failure == "corrupt":
            path.write_bytes(b"corrupt fixture")
        else:
            path.unlink()
        with pytest.raises((ValueError, OSError)):
            read_ig(study, "a")
    assert study.poisoned is True


def test_changed_disjoint_plan_and_duplicate_source_are_rejected(planned_ig):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import (
        plan_ig,
        save_ig,
        save_ig_plan,
    )

    study, predictor, rows, plan = planned_ig
    changed = plan_ig(
        read_results(study, "a"),
        run_id="a",
        policy=POLICY,
        probe_ids=[rows[1]["source_row"]],
    )
    with pytest.raises(ValueError, match="immutable"):
        save_ig_plan(study, "a", changed)
    save_ig_plan(study, "a", plan)
    result = compute_ig(rows[0], predictor, policy=POLICY)
    save_ig(study, "a", result)
    with pytest.raises(ValueError, match="overlapping"):
        save_ig(study, "a", result)


def test_model_free_ig_reader_uses_only_committed_study(planned_ig, inputs, tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import save_ig

    study, predictor, rows, _ = planned_ig
    save_ig(study, "a", compute_ig(rows[0], predictor, policy=POLICY))
    study.finish()
    study.store.put("runs/a/artifacts/ig-orphan.npz", b"uncommitted")
    inputs.export.rename(tmp_path / "unavailable-model")
    inputs.source.unlink()
    inputs.manifest.unlink()
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
from xai_test.cas12a_NT_analysis.nt_ig_artifacts import read_ig
study = Study.reopen(sys.argv[1], receipt=json.load(open(sys.argv[2])))
report = read_ig(study, 'a', require_complete=True)
assert report['complete']
assert abs(report['results'][0]['summary']['reconstructed_output'] - .375) < 1e-6
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


def test_retry_exhaustion_quality_is_reconstructed_from_saved_mass_vectors(
    ig_predictions,
):
    from xai_test.cas12a_NT_analysis.nt_ig import compute_ig
    from xai_test.cas12a_NT_analysis.nt_ig_artifacts import (
        plan_ig,
        read_ig,
        save_ig,
        save_ig_plan,
    )

    study, predictor, rows = ig_predictions
    strict = {key: (400 if key == "max_steps" else 0.0) for key in POLICY}
    plan = plan_ig(
        read_results(study, "a"),
        run_id="a",
        policy=strict,
        probe_ids=[rows[0]["source_row"]],
    )
    study.start("a", "ig", 300, {"device": "cpu"})
    save_ig_plan(study, "a", plan)
    study.record_execution(
        {"device": "cpu"}, {"validated": True, "injected_cpu_model": True}
    )
    save_ig(study, "a", compute_ig(rows[0], predictor, policy=strict))
    study.finish()
    report = read_ig(study, "a")
    result = report["results"][0]
    assert report["failed"] == [rows[0]["source_row"]]
    assert result["failure"] == {
        "stage": "quality",
        "reason": "retry limit exceeded",
        "steps": 400,
    }
    assert [attempt["steps"] for attempt in result["attempts"]] == [200, 400]
    assert result["summary"]["quality"]["completeness_pass"] is False
    assert result["summary"]["quality"]["convergence_pass"] is False
