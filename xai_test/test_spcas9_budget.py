"""Fixed-study requests to durable evidence, using disposable fake backends only."""

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_spcas9_study import inputs as _study_inputs

from SpCas9.spcas9_data import HEAD_NAMES, output_name
from SpCas9.spcas9_split import CHECKSUM_FILENAME, file_sha256, write_json
from SpCas9.spcas9_study import StudyProtocol
from SpCas9.spcas9_study_inputs import InputPins

inputs = _study_inputs


@pytest.fixture
def synthetic_ledger(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            {
                "optimizer_updates_charged": 6,
                "training_seconds_charged": 30.0,
                "sessions": [
                    {
                        "status": "finished",
                        "optimizer_updates_reserved": 6,
                        "duration_seconds": 30.0,
                    }
                ],
            }
        )
    )
    return path


@pytest.fixture
def controlled_inputs(inputs):
    path = inputs.split_dir / "val.csv"
    frame = pd.read_csv(path)
    frame["sequence"] = "ACGT" * 6 + "AGGAAA"
    for head in HEAD_NAMES:
        frame[head] = (np.arange(len(frame)) % 5) * 100.0
    frame.to_csv(path, index=False, lineterminator="\n")
    files = {**inputs.pins.files, "val.csv": file_sha256(path)}
    sidecar = inputs.split_dir / CHECKSUM_FILENAME
    record = json.loads(sidecar.read_text())
    record["files"] = files
    write_json(record, sidecar)
    return replace(
        inputs, pins=InputPins(inputs.pins.dataset_sha256, file_sha256(sidecar), files)
    )


class FixedBackend:
    """Picklable disposable backend; never loads a model or updates weights."""

    def fit(self, training, validation, *, seed, checkpoint_path):
        from SpCas9.spcas9_selection import NGG_MAE, NGG_SPEARMAN, ngg_stopping
        from SpCas9.spcas9_study_backend import FitResult

        checkpoint_path.write_bytes(b"disposable selected checkpoint")
        history = {
            "recipe_optimization_loss": [5.0],
            "val_recipe_optimization_loss": [5.0],
            "learning_rate": [0.001],
            NGG_MAE: [5.0],
            NGG_SPEARMAN: [1.0],
        }
        return FitResult(
            history,
            self.predict(checkpoint_path, validation.inputs),
            {"backend": "fake"},
            ngg_stopping(history),
        )

    def predict(self, checkpoint_path, inputs):
        return {
            output_name(h): (5.0 + np.arange(len(inputs)) % 5)[:, None]
            for h in HEAD_NAMES
        }


class FailingBackend(FixedBackend):
    def __init__(self, failure):
        self.failure = failure

    def fit(self, *args, **kwargs):
        if self.failure == "exit":
            os._exit(17)
        raise {
            "failure": RuntimeError,
            "interrupted": KeyboardInterrupt,
            "timeout": TimeoutError,
        }[self.failure]("disposable failure")


class OrphanBackend(FixedBackend):
    def __init__(self, *, bounded):
        self.bounded = bounded

    def fit(self, *args, checkpoint_path, **kwargs):
        (checkpoint_path.parent / "worker-ready").write_text(str(os.getpid()))
        time.sleep(2)
        if not self.bounded:
            os._exit(17)
        return super().fit(*args, checkpoint_path=checkpoint_path, **kwargs)


class HangingBackend(FixedBackend):
    def fit(self, *args, checkpoint_path, **kwargs):
        (checkpoint_path.parent / "worker-ready").write_text(str(os.getpid()))
        time.sleep(60)
        raise AssertionError("watchdog must terminate a hung backend")


class ChangingEnvironmentBackend(FixedBackend):
    def predict(self, checkpoint_path, inputs):
        result = super().predict(checkpoint_path, inputs)
        os.environ["OMP_NUM_THREADS"] = "changed-during-execution"
        return result


def test_fixed_study_records_protocol_identity_and_blocks_premature_full_training(
    inputs, synthetic_ledger, tmp_path
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        inputs,
        checkpoint_selection="ngg",
        execution_mode="full",
        synthetic_ledger=synthetic_ledger,
    )
    config = study.execution_configuration()
    assert config["environment"]["interpreter"] == sys.executable
    assert config["source"]["snapshot_sha256"]
    assert config["run_order"] == [
        {"recipe": recipe, "seed": seed}
        for recipe in ("baseline", "candidate")
        for seed in (42, 43, 44)
    ]
    assert config["representative_seed"] == 42
    assert config["budgets"] == {
        "full_fits": 6,
        "training_seconds": 21600,
        "synthetic_updates": 10,
        "synthetic_seconds": 300,
    }
    assert config["acceptance"]["relative_ngg_mae_reduction"] == 0.02
    assert config["ci"]["valid_draws"] == 2000
    assert study.budget_status()["training_seconds"] == 30.0
    with pytest.raises(PermissionError, match="readiness"):
        study.freeze_execution()
    run = study.plan_run(seed=42, recipe="baseline")
    with pytest.raises(PermissionError, match="readiness"):
        study.train_validation(run)
    assert study.budget_status()["attempts"] == []


def test_six_fits_follow_fixed_order_and_cannot_reset_after_reopening(
    controlled_inputs, synthetic_ledger, tmp_path
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    wrong = study.plan_run(seed=43, recipe="baseline")
    with pytest.raises(ValueError, match="order"):
        study.train_validation(wrong, backend=FixedBackend())
    assert study.budget_status()["attempts"] == []
    first = None
    for recipe, seed in [
        (r, s) for r in ("baseline", "candidate") for s in (42, 43, 44)
    ]:
        run = study.plan_run(seed=seed, recipe=recipe)
        result = study.train_validation(run, backend=FixedBackend())
        assert result["eligible"]
        first = first or run
        study = StudyProtocol(study.path)
    status = study.budget_status()
    assert len(status["attempts"]) == 6
    assert status["training_seconds"] > 30.0
    assert status["evaluation_seconds"] > 0
    extra = study.plan_run(seed=42, recipe="candidate")
    with pytest.raises(ValueError, match="six|6|exhausted"):
        study.train_validation(extra, backend=FixedBackend())
    assert study.report_only(first)["eligible"]
    recovered = study.recover_reports(tmp_path / "recovered")
    assert len(recovered["completed"]) == 6
    assert not recovered["training_permitted"]
    with pytest.raises(FileExistsError):
        study.recover_reports(tmp_path / "recovered")
    checkpoint = study.path / "runs" / first / "9hydra_v2.keras"
    checkpoint.write_bytes(b"damaged completed checkpoint")
    with pytest.raises(ValueError, match="corrupted"):
        study.recover_reports(tmp_path / "damaged")


@pytest.mark.parametrize(
    "failure, expected",
    [
        ("failure", "failed"),
        ("interrupted", "interrupted"),
        ("timeout", "timed_out"),
        ("exit", "failed"),
    ],
)
def test_failed_fit_consumes_attempt_and_blocks_all_later_fits(
    controlled_inputs, synthetic_ledger, tmp_path, failure, expected
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    run = study.plan_run(seed=42, recipe="baseline")
    with pytest.raises((RuntimeError, KeyboardInterrupt, TimeoutError)):
        study.train_validation(run, backend=FailingBackend(failure))
    study = StudyProtocol(study.path)
    status = study.budget_status()
    assert len(status["attempts"]) == 1
    assert status["attempts"][0]["status"] == expected
    assert status["training_seconds"] > 30
    assert status["blocked"]
    for next_run in (run, study.plan_run(seed=43, recipe="baseline")):
        with pytest.raises(ValueError, match="blocks"):
            study.train_validation(next_run, backend=FixedBackend())


@pytest.mark.parametrize("bounded", [True, False])
def test_parent_process_death_recovers_conservative_time_or_unknown_and_never_retries(
    controlled_inputs, synthetic_ledger, tmp_path, bounded
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    run = study.plan_run(seed=42, recipe="baseline")
    run_path = study.path / "runs" / run
    script = (
        "from pathlib import Path; from SpCas9.spcas9_study import StudyProtocol; "
        "from test_spcas9_budget import OrphanBackend; "
        f"StudyProtocol(Path({str(study.path)!r})).train_validation({run!r}, backend=OrphanBackend(bounded={bounded!r}))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script], cwd=Path(__file__).parent
    )
    try:
        deadline = time.monotonic() + 30
        while not (run_path / "worker-ready").exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        process.terminate()
        process.wait(timeout=5)
        # The disposable child returns or exits after two seconds. Never leave
        # a test worker or real training behind after killing its parent.
        time.sleep(3)
        recovered = StudyProtocol(study.path).recover_reports(tmp_path / "recovered")
        budget = recovered["budget"]
        assert budget["blocked"]
        assert budget["attempts"][0]["status"] == "interrupted"
        if bounded:
            assert budget["training_seconds"] >= 32
        else:
            assert budget["training_seconds"] is None
        with pytest.raises(ValueError, match="blocks"):
            StudyProtocol(study.path).train_validation(run, backend=FixedBackend())
        assert not recovered["training_permitted"]
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize(
    "change", ["config", "source", "environment", "synthetic", "inputs", "journal"]
)
def test_changed_execution_identity_or_missing_history_refuses_training(
    controlled_inputs, synthetic_ledger, tmp_path, monkeypatch, change
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    run = study.plan_run(seed=42, recipe="baseline")
    if change == "source":
        original = subprocess.check_output

        def changed_revision(command, **kwargs):
            if command[-2:] == ["rev-parse", "HEAD"]:
                return b"changed-revision\n"
            return original(command, **kwargs)

        monkeypatch.setattr(subprocess, "check_output", changed_revision)
    elif change == "environment":
        monkeypatch.setenv("OMP_NUM_THREADS", "changed-after-plan")
    else:
        path = {
            "config": study.path / "execution-config.json",
            "synthetic": synthetic_ledger,
            "inputs": controlled_inputs.split_dir / "train.csv",
            "journal": study.path / "budget.jsonl",
        }[change]
        path.write_bytes(b"" if change == "journal" else path.read_bytes() + b"\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        study.train_validation(run, backend=FixedBackend())


def test_synthetic_handoff_refuses_reset_and_additional_checks(
    controlled_inputs, synthetic_ledger, tmp_path
):
    from SpCas9.spcas9_ranking_checks import main

    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    with pytest.raises(FileExistsError, match="reset"):
        StudyProtocol.plan(
            tmp_path / "fresh",
            controlled_inputs,
            checkpoint_selection="ngg",
            execution_mode="verification",
            synthetic_ledger=synthetic_ledger,
        )
    with pytest.raises(ValueError, match="transferred"):
        main(["--ledger", str(synthetic_ledger)])
    with pytest.raises(FileExistsError):
        study.recover_reports(study.path)
    with pytest.raises(ValueError, match="overlap"):
        study.recover_reports(controlled_inputs.split_dir / "recovered")


def test_evaluation_and_bootstrap_do_not_consume_training_budget(
    controlled_inputs, synthetic_ledger, tmp_path
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    with study.evaluation_timing():
        time.sleep(0.01)
    with study.evaluation_timing(bootstrap=True):
        time.sleep(0.01)
    status = study.budget_status()
    assert status["training_seconds"] == 30
    assert status["evaluation_seconds"] > 0
    assert status["bootstrap_seconds"] > 0


def test_external_deadline_terminates_hung_backend_and_blocks_later_training(
    controlled_inputs, synthetic_ledger, tmp_path, monkeypatch
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    run = study.plan_run(seed=42, recipe="baseline")
    ready = study.path / "runs" / run / "worker-ready"
    real_clock = time.monotonic
    monkeypatch.setattr(
        "SpCas9.spcas9_budget.time.monotonic",
        lambda: real_clock() + (21600 if ready.exists() else 0),
    )
    with pytest.raises(TimeoutError):
        study.train_validation(run, backend=HangingBackend())
    status = study.budget_status()
    assert status["attempts"][0]["status"] == "timed_out"
    assert status["remaining_seconds"] == 0
    assert status["blocked"]


def test_child_refuses_to_fit_when_absolute_deadline_has_passed_during_startup(
    controlled_inputs, synthetic_ledger, tmp_path, monkeypatch
):
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    run = study.plan_run(seed=42, recipe="baseline")
    real_clock = time.monotonic
    # Parent's remaining allowance appears unspent; the child's unmodified
    # clock observes that the absolute deadline has already passed at startup.
    monkeypatch.setattr(
        "SpCas9.spcas9_budget.time.monotonic", lambda: real_clock() - 21600
    )
    with pytest.raises(TimeoutError):
        study.train_validation(run, backend=FixedBackend())
    assert not (study.path / "runs" / run / "9hydra_v2.keras").exists()
    assert study.budget_status()["attempts"][0]["status"] == "timed_out"


def test_runtime_identity_change_cannot_complete_a_run(
    controlled_inputs, synthetic_ledger, tmp_path, monkeypatch
):
    monkeypatch.setenv("OMP_NUM_THREADS", "frozen-test-setting")
    study = StudyProtocol.plan(
        tmp_path / "studies",
        controlled_inputs,
        checkpoint_selection="ngg",
        execution_mode="verification",
        synthetic_ledger=synthetic_ledger,
    )
    run = study.plan_run(seed=42, recipe="baseline")
    with pytest.raises(ValueError, match="changed during"):
        study.train_validation(run, backend=ChangingEnvironmentBackend())
    assert not study.report_only(run)["eligible"]
    assert study.budget_status()["blocked"]


def test_planning_refuses_synthetic_history_owned_by_check_runner(
    controlled_inputs, synthetic_ledger, tmp_path
):
    lock = synthetic_ledger.with_suffix(".json.lock")
    lock.write_text("active check runner")
    with pytest.raises(FileExistsError):
        StudyProtocol.plan(
            tmp_path / "studies",
            controlled_inputs,
            checkpoint_selection="ngg",
            execution_mode="verification",
            synthetic_ledger=synthetic_ledger,
        )
    assert not (tmp_path / "studies").exists()


@pytest.mark.parametrize("column", ["sequence", "protospacer", "row_index"])
def test_repinning_cannot_hide_split_overlap(inputs, tmp_path, column):
    path = inputs.split_dir / "val.csv"
    frame = pd.read_csv(path)
    train = pd.read_csv(inputs.split_dir / "train.csv")
    frame.loc[0, column] = train.loc[0, column]
    if column in ("sequence", "protospacer"):
        frame.loc[0, "sequence"] = train.loc[0, "sequence"]
        frame.loc[0, "protospacer"] = train.loc[0, "protospacer"]
    frame.to_csv(path, index=False, lineterminator="\n")
    files = {**inputs.pins.files, "val.csv": file_sha256(path)}
    sidecar = inputs.split_dir / CHECKSUM_FILENAME
    record = json.loads(sidecar.read_text())
    record["files"] = files
    write_json(record, sidecar)
    changed = replace(
        inputs, pins=InputPins(inputs.pins.dataset_sha256, file_sha256(sidecar), files)
    )
    with pytest.raises(ValueError, match="overlap"):
        StudyProtocol.plan(tmp_path / "studies", changed)
