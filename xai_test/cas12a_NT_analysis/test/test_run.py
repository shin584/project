import pytest

from xai_test.cas12a_NT_analysis.nt_run import DirectoryStore, Study


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def monotonic(self):
        return self.seconds

    def utc(self):
        return 1_800_000_000.0 + self.seconds


def test_replacement_run_inherits_charge_and_requires_acknowledged_tip(tmp_path):
    clock = Clock()
    study = Study.create(tmp_path / "drive", fixture=True, clock=clock)
    study.register("run-a", {"model": "a"})
    study.start("run-a", "session-a", 100, {"device": "cpu"})
    clock.seconds += 3
    study.finish()
    receipt = study.receipt
    reopened = Study.reopen(tmp_path / "drive", receipt=receipt, clock=clock)
    reopened.register("run-b", {"model": "b"})
    assert reopened.status()["charged_seconds"] == 100
    reopened.start("run-b", "session-b", 21500, {"device": "cpu"})
    reopened.finish()
    with pytest.raises(ValueError, match="budget"):
        reopened.start("run-b", "session-c", 1, {"device": "cpu"})
    with pytest.raises(ValueError, match="tip"):
        Study.reopen(tmp_path / "drive", receipt=receipt, clock=clock)


def test_interruption_requires_termination_bound_and_settles_only_once(tmp_path):
    clock = Clock()
    root = tmp_path / "drive"
    study = Study.create(root, fixture=True, clock=clock)
    study.register("a", {"model": "a"})
    study.start("a", "vm1", 100, {"device": "cpu"})
    clock.seconds = 5
    study.checkpoint()
    reopened = Study.reopen(root, receipt=study.receipt, clock=clock)
    clock.seconds = 200
    with pytest.raises(ValueError, match="unresolved"):
        reopened.start("a", "vm2", 10, {})
    evidence = tmp_path / "external-termination.txt"
    evidence.write_text("Operator confirmed VM1 terminated at +120s")
    with pytest.raises(ValueError, match="upper bound"):
        reopened.settle(evidence, terminated_utc=1_800_000_120, upper_seconds=119)
    reopened.settle(evidence, terminated_utc=1_800_000_120, upper_seconds=120)
    assert reopened.status()["charged_seconds"] == 120
    with pytest.raises(ValueError, match="duplicate settlement"):
        reopened.settle(evidence, terminated_utc=1_800_000_120, upper_seconds=120)
    reopened.start("a", "vm2", 10, {"device": "different GPU"})


def test_newest_receipt_rejects_suffix_deletion_and_missing_ledger(tmp_path):
    root = tmp_path / "drive"
    study = Study.create(root, fixture=True)
    study.register("a", {"model": "a"})
    receipt = study.receipt
    (root / "ledger" / "00000001.json").unlink()
    with pytest.raises(ValueError, match="tip"):
        Study.reopen(root, receipt=receipt)
    with pytest.raises(ValueError, match="missing ledger"):
        Study.reopen(tmp_path / "lost-root", receipt=receipt)
    with pytest.raises(ValueError, match="tip"):
        Study.reopen(root, receipt=None)
    with pytest.raises(ValueError, match="empty study"):
        Study.create(root, fixture=True)


class FaultStore(DirectoryStore):
    fail_event = None
    delay_event = None

    def put(self, name, data):
        import json

        event = (
            json.loads(data).get("payload", {}).get("event")
            if name.startswith("ledger/")
            else None
        )
        if event == self.fail_event and self.fail_event:
            raise OSError("external storage unavailable")
        result = super().put(name, data)
        if event == self.delay_event and self.delay_event:
            self.clock.seconds += 200
        return result


def test_storage_failure_poisoning_keeps_job_unresolved(tmp_path):
    store = FaultStore(tmp_path / "drive")
    study = Study.create(store.root, fixture=True, store=store)
    study.register("a", {"model": "a"})
    study.start("a", "vm1", 100, {})
    store.fail_event = "heartbeat"
    with pytest.raises(OSError, match="unavailable"):
        study.checkpoint()
    with pytest.raises(ValueError, match="poisoned"):
        study.start("a", "vm2", 100, {})


def test_final_storage_acknowledgement_overrun_cannot_resume(tmp_path):
    clock = Clock()
    store = FaultStore(tmp_path / "drive")
    store.clock = clock
    study = Study.create(store.root, fixture=True, clock=clock, store=store)
    study.register("a", {})
    study.start("a", "vm1", 100, {})
    prior_receipt = study.receipt
    store.delay_event = "finish"
    with pytest.raises(ValueError, match="overrun"):
        study.finish()
    with pytest.raises(ValueError, match="cannot acknowledge"):
        _ = study.receipt
    with pytest.raises(ValueError, match="tip"):
        Study.reopen(store.root, receipt=prior_receipt, clock=clock)
    assert study.status()["charged_seconds"] == 200
    assert study.status()["blocked_reason"] == "reservation overrun"


def test_corrupt_shard_rejects_readback_before_further_work(tmp_path):
    root = tmp_path / "drive"
    study = Study.create(root, fixture=True)
    study.register("a", {})
    study.start("a", "cpu", 100, {})
    study.save_artifact("one", {"value": "original"}, [1])
    receipt = study.receipt
    (root / "runs/a/artifacts/one.json").write_bytes(b'{"value":"changed"}')
    with pytest.raises(ValueError, match="artifact hash"):
        Study.reopen(root, receipt=receipt)


def test_same_run_rejects_changed_identity(tmp_path):
    study = Study.create(tmp_path / "drive", fixture=True)
    study.register("a", {"packages": {"torch": "one"}, "model": "frozen"})
    with pytest.raises(ValueError, match="identity mismatch"):
        study.register("a", {"packages": {"torch": "two"}, "model": "frozen"})


def test_time_crossing_reservation_during_final_acknowledgement_blocks_work(tmp_path):
    class TickingClock(Clock):
        def monotonic(self):
            value = self.seconds
            self.seconds += 1
            return value

    study = Study.create(tmp_path / "drive", fixture=True, clock=TickingClock())
    study.register("a", {})
    study.start("a", "cpu", 4.5, {})
    with pytest.raises(ValueError, match="overrun"):
        study.finish()
    with pytest.raises(ValueError, match="poisoned"):
        study.start("a", "again", 100, {})


def test_transient_read_failure_requires_explicit_reopen(tmp_path):
    class ReadFailureStore(DirectoryStore):
        broken = False

        def read(self, name):
            if self.broken:
                raise OSError("external storage read unavailable")
            return super().read(name)

    store = ReadFailureStore(tmp_path / "drive")
    study = Study.create(store.root, fixture=True, store=store)
    study.register("a", {})
    study.start("a", "cpu", 100, {})
    study.save_artifact("one", {"value": "one"}, [1])
    store.broken = True
    with pytest.raises(OSError, match="read unavailable"):
        study.checkpoint()
    store.broken = False
    with pytest.raises(ValueError, match="poisoned"):
        study.checkpoint()
