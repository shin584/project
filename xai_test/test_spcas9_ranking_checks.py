"""Exercise the synthetic check watchdog without importing a training backend."""

import json
import subprocess
from pathlib import Path

import pytest

from SpCas9.spcas9_ranking_checks import main


def test_check_runner_requires_exclusive_ledger_ownership(tmp_path):
    path = tmp_path / "budget.json"
    lock = tmp_path / "budget.json.lock"
    lock.write_text("another parent owns this ledger", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(["--ledger", str(path)])
    assert lock.read_text(encoding="utf-8") == "another parent owns this ledger"
    assert not path.exists()


def test_check_runner_rechecks_handoff_after_acquiring_reservation(
    tmp_path, monkeypatch
):
    path = tmp_path / "budget.json"
    owner = tmp_path / "budget.json.study-owner.json"
    original_open = Path.open

    def handoff_before_lock_acquisition(candidate, *args, **kwargs):
        if candidate == path.with_suffix(".json.lock"):
            owner.write_text("study claimed history before lock was acquired")
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", handoff_before_lock_acquisition)
    with pytest.raises(ValueError, match="transferred"):
        main(["--ledger", str(path)])
    assert not path.exists()
    assert not path.with_suffix(".json.lock").exists()


@pytest.mark.parametrize(
    "record",
    [
        {
            "training_seconds_charged": 300,
            "optimizer_updates_charged": 0,
            "sessions": [],
        },
        {"training_seconds_charged": 0, "optimizer_updates_charged": 6, "sessions": []},
        {
            "training_seconds_charged": 1,
            "optimizer_updates_charged": 0,
            "sessions": [{"status": "running"}],
        },
    ],
)
def test_check_runner_refuses_exhausted_or_unreconciled_evidence(tmp_path, record):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        main(["--ledger", str(path)])
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_timeout_preserves_child_update_reservation_and_charges_elapsed(
    tmp_path, monkeypatch
):
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {
                "training_seconds_charged": 250,
                "optimizer_updates_charged": 0,
                "sessions": [],
            }
        ),
        encoding="utf-8",
    )
    ticks = iter([100.0, 149.0])
    monkeypatch.setattr(
        "SpCas9.spcas9_ranking_checks.time.monotonic", lambda: next(ticks)
    )

    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 49
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["sessions"][-1]["status"] == "running"
        assert kwargs["env"]["SPCAS9_SYNTHETIC_SESSION"] == record["sessions"][-1]["id"]
        record["optimizer_updates_charged"] = 6
        record["sessions"][-1]["optimizer_updates_reserved"] = 6
        path.write_text(json.dumps(record), encoding="utf-8")
        raise subprocess.TimeoutExpired(command, 49)

    monkeypatch.setattr("SpCas9.spcas9_ranking_checks.subprocess.run", timed_out)
    assert main(["--ledger", str(path)]) == 124
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["training_seconds_charged"] == 299
    assert record["optimizer_updates_charged"] == 6
    assert record["sessions"][-1]["status"] == "timed_out"
    assert record["sessions"][-1]["duration_seconds"] == 49
