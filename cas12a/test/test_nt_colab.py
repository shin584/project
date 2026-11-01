"""Execute the notebook's public workflow calls with offline, real tiny models."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from cas12a.nt_workflow import ModelSource, NTWorkflow, SplitConfig
from cas12a.test.test_nt_resume import interrupted
from cas12a.test.test_nt_workflow import synthetic_table

NOTEBOOK = Path(__file__).parents[1] / "model" / "NT_Cas12a_colab.ipynb"


def notebook_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def execute(cells, name, namespace):
    # Execute only the checked-in notebook, with environment setup excluded.
    exec(compile(cells[name], f"{NOTEBOOK.name}:{name}", "exec"), namespace)  # noqa: S102


def test_notebook_fresh_calls_complete_offline_lifecycle(
    tmp_path, tiny_nt, monkeypatch
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    cells = notebook_cells()
    source = tmp_path / "source.csv"
    table = synthetic_table(407)
    # Keep synthetic labels in range without changing the public loader contract.
    from cas12a.data import TARGET_COLUMN

    table[TARGET_COLUMN] = [float(i % 101) for i in range(407)]
    table.to_csv(source, index=False)
    namespace = {
        "NTWorkflow": NTWorkflow,
        "source": source,
        "partition_path": tmp_path / "partition.json",
        "local_work_root": tmp_path / "local",
        "drive_output_root": tmp_path / "durable",
        "drive_mount_root": tmp_path / "mount",
        "split": SplitConfig.fixture(400, 2, 5),
        "model_source": ModelSource.fixture(tiny_nt),
        "mode": "fresh",
        "run_id": "notebook-offline",
        "resume_run_id": "",
    }
    for stage in ("workflow", "prepare", "preflight", "train"):
        execute(cells, stage, namespace)
    run = namespace["run"]
    assert run.status["state"] == "training_completed"
    assert run.status["model_completed"] is False
    history = json.loads((run.local_dir / "history.json").read_bytes())
    assert history["selected_global_step"] == 125

    execute(cells, "local-export", namespace)
    assert namespace["run"].status["local_export_validated"] is True
    assert namespace["run"].status["durable_export_validated"] is False
    execute(cells, "durable-export", namespace)
    execute(cells, "evaluate", namespace)
    assert namespace["run"].status["final_evaluation_completed"] is False
    execute(cells, "publish-evaluation", namespace)
    run = namespace["run"]
    assert run.status["final_evaluation_completed"] is True
    assert run.status["durable_evaluation_validated"] is True
    assert run.status["run_kind"] == "fixture"
    assert run.status["model_completed"] is False
    execution = json.loads((run.durable_dir / "execution.json").read_bytes())
    assert execution["environment"]["device"] == "cpu"
    assert execution["environment"]["packages"]["torch"]
    report = json.loads((run.durable_dir / "export.json").read_bytes())
    receipt = json.loads((run.durable_dir / "durable_export.json").read_bytes())
    assert receipt["selected_artifact_sha256"] == report["selected_artifact_sha256"]
    final = run.durable_dir / "final_evaluation"
    evaluation = json.loads((final / "evaluation.json").read_bytes())
    assert evaluation["selected_artifact_sha256"] == report["selected_artifact_sha256"]
    with (final / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 5
    assert {
        "raw_prediction",
        "bounded_efficiency",
        "baseline_train_mean",
    } <= predictions[0].keys()
    for name in ("durable_export.json", "durable_evaluation.json"):
        receipt = json.loads((run.durable_dir / name).read_bytes())
        root = final if name == "durable_evaluation.json" else run.durable_dir
        for filename, expected in receipt["files"].items():
            assert (
                hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected
            )
    assert not (run.durable_dir / "checkpoints").exists()
    assert not list(run.durable_dir.rglob("training_state.pt"))


def test_notebook_resume_uses_original_id_without_preparing_or_preflighting(
    tmp_path, tiny_nt, monkeypatch
):
    workflow, original = interrupted(tmp_path, tiny_nt, monkeypatch)
    partition_before = workflow.partition_path.read_bytes()
    cells = notebook_cells()
    namespace = {
        "NTWorkflow": NTWorkflow,
        "source": workflow.source,
        "partition_path": workflow.partition_path,
        "local_work_root": workflow.local_work_root,
        "drive_output_root": workflow.drive_output_root,
        "drive_mount_root": workflow.drive_mount_root,
        "split": workflow.split,
        "mode": "resume",
        "resume_run_id": original.run_id,
    }
    for stage in ("workflow", "prepare", "preflight"):
        execute(cells, stage, namespace)
    assert "run" not in namespace
    execute(cells, "train", namespace)
    run = namespace["run"]
    assert run.run_id == original.run_id
    assert run.status["resumed_from_global_step"] == 2
    assert run.status["state"] == "training_completed"
    assert workflow.partition_path.read_bytes() == partition_before
    execute(cells, "local-export", namespace)
    assert namespace["run"].status["local_export_validated"] is True


def test_notebook_requires_an_explicit_execution_choice():
    cells = notebook_cells()
    with pytest.raises(ValueError, match="Explicitly choose fresh or resume"):
        execute(cells, "configuration", {"Path": Path})
    with pytest.raises(ValueError, match="Explicitly choose fresh or resume"):
        execute(cells, "train", {"mode": "choose"})


def code_cell_order():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [cell["id"] for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_the_restart_check_precedes_every_workflow_import():
    order = code_cell_order()
    cells = notebook_cells()
    checked = order.index("restart-check")
    assert checked < order.index("configuration")
    importing = [
        name for name, source in cells.items() if "cas12a.nt_workflow" in source
    ]
    assert importing and all(order.index(name) > checked for name in importing)
    # The added cell checks the environment only; no workflow logic is copied.
    assert "workflow" not in cells["restart-check"]
    assert len(cells["restart-check"].splitlines()) <= 6


def test_the_restart_check_cell_stops_a_session_that_installed_without_restarting(
    tmp_path,
):
    from cas12a.nt_session import SessionRestartRequired, record_installation

    marker = tmp_path / "nt-colab-install.json"
    record_installation(marker)
    with pytest.raises(SessionRestartRequired, match="Restart"):
        execute(notebook_cells(), "restart-check", {"INSTALL_MARKER": marker})


def test_the_restart_check_cell_passes_a_restarted_session(tmp_path):
    from cas12a.nt_session import kernel_identity, record_installation

    marker = tmp_path / "nt-colab-install.json"
    identity = dict(kernel_identity(), pid=kernel_identity()["pid"] + 1)
    record_installation(marker, identity=identity)
    namespace = {"INSTALL_MARKER": marker}
    execute(notebook_cells(), "restart-check", namespace)
    assert namespace["session_preconditions"]["restarted"] is True


def test_the_install_cell_refuses_a_repeat_before_touching_packages(tmp_path):
    from cas12a.nt_session import InstallationAlreadyCompleted, record_installation

    marker = tmp_path / "nt-colab-install.json"
    record_installation(marker)
    namespace = {"INSTALL_MARKER": marker, "subprocess": None, "sys": None}
    with pytest.raises(InstallationAlreadyCompleted):
        execute(notebook_cells(), "install", namespace)
