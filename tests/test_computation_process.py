"""#194: exercise the renderer/API boundary across a real spawned process."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop.api import DesktopApi
from desktop.computation import ProcessDispatcher

DNA = "GATCCAGTCAGCATGCTTTACGATCGTACGATCGAATCGTAGGCTAGCATCGATCGATGCATCGTAGCTAGCTAGCATCGA"


def test_spawned_fake_serves_sites_views_charts_and_csv(tmp_path):
    events = []
    saved = tmp_path / "results.csv"
    dispatcher = ProcessDispatcher(events.append, preview=True)
    api = DesktopApi(dispatcher.renderer, dispatcher, lambda _: str(saved))
    try:
        for call in [api.prepare, lambda: api.scan_sites(DNA), lambda: api.predict(DNA),
                     lambda: api.explain(DNA, "Cas12a|+|16", "Cas12a", "ig"),
                     lambda: api.save_csv(DNA)]:
            rid = call()["request_id"]
            assert dispatcher.wait_idle(20)
            terminal = [e for e in events if e["request_id"] == rid][-1]
            assert terminal["type"] in {"ready", "result"}, terminal
        assert dispatcher.pid != os.getpid()
        assert all(e["generation"] == 1 for e in events)
        fragments = {k: v for e in events for k, v in e.get("fragments", {}).items()}
        assert {"sites", "best", "variants", "detail", "xai", "notice"} == set(fragments)
        assert 'data:image/svg+xml;base64,' in fragments["xai"]
        assert "SpCas9(WT)" in fragments["variants"]
        assert saved.read_bytes().startswith(b"\xef\xbb\xbf")
        json.dumps(events)
    finally:
        dispatcher.close()
    assert dispatcher.pid is None


def test_ui_imports_do_not_load_computation_libraries():
    script = """
import sys
from desktop.computation import ProcessDispatcher
from desktop.app import main
from desktop.screen import SCREEN_HTML
d = ProcessDispatcher(lambda e: None)
assert d.pid is None
assert not set(sys.modules).intersection({
    'torch', 'transformers', 'predict', 'numpy', 'matplotlib', 'shap',
    'desktop.renderer', 'desktop.fake', 'desktop.xai_view'})
d.close()
"""
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parent.parent,
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr


def test_cancel_and_replacement_remain_serial_and_reuse_the_child():
    events = []
    started = threading.Event()

    def emit(event):
        events.append(event)
        if event["type"] == "progress":
            started.set()

    dispatcher = ProcessDispatcher(emit, preview=True)
    api = DesktopApi(dispatcher.renderer, dispatcher)
    try:
        rid = api.explain(DNA, "Cas12a|+|16", "Cas12a", "shap",
                          {"nsamples": 1024, "n_background": 32})["request_id"]
        assert started.wait(10)
        pid = dispatcher.pid
        api.cancel()
        replacement = api.predict(DNA)["request_id"]
        assert dispatcher.wait_idle(20)
        assert dispatcher.pid == pid
        assert not any(e["type"] == "result" and e["request_id"] == rid for e in events)
        assert dispatcher.outcome(rid) == "cancelled"
        assert dispatcher.outcome(replacement) == "delivered"
        assert [e["request_id"] for e in events] == sorted(e["request_id"] for e in events)
    finally:
        dispatcher.close()


def test_dead_child_reports_error_and_next_request_retries_in_a_new_generation():
    events = []
    dispatcher = ProcessDispatcher(events.append, factory_name="process_fixtures:ControlledRenderer")
    api = DesktopApi(dispatcher.renderer, dispatcher)
    try:
        rid = api.scan_sites("crash")["request_id"]
        assert dispatcher.wait_idle(10)
        assert dispatcher.outcome(rid) == "failed"
        assert events[-1]["type"] == "error"
        old_pid = dispatcher.pid
        next_id = api.prepare()["request_id"]
        assert dispatcher.wait_idle(10)
        assert dispatcher.outcome(next_id) == "delivered"
        assert dispatcher.generation == 2
        assert dispatcher.pid != old_pid
        assert events[-1]["generation"] == 2
    finally:
        dispatcher.close()


def test_preparation_errors_are_escaped_and_can_be_retried():
    events = []
    dispatcher = ProcessDispatcher(events.append, factory_name="process_fixtures:FailsOnceRenderer")
    api = DesktopApi(dispatcher.renderer, dispatcher)
    try:
        rid = api.prepare()["request_id"]
        assert dispatcher.wait_idle(10)
        assert dispatcher.outcome(rid) == "failed"
        assert "&lt;retry preparation&gt;" in events[-1]["html"]
        rid = api.prepare()["request_id"]
        assert dispatcher.wait_idle(10)
        assert dispatcher.outcome(rid) == "delivered"
        assert events[-1]["type"] == "ready"
    finally:
        dispatcher.close()


@pytest.mark.parametrize("terminal", ["ready", "result", "error"])
def test_restart_discards_all_events_from_uninterruptible_old_jobs(terminal):
    events = []
    started, release = threading.Event(), threading.Event()
    dispatcher = ProcessDispatcher(events.append, preview=True)

    def old_job(on_batch):
        started.set()
        assert release.wait(10)
        on_batch(1, 1)
        if terminal == "error":
            raise ValueError("obsolete error")
        return "old ready" if terminal == "ready" else {"notice": "old result"}

    try:
        submit = dispatcher.submit_lifecycle if terminal in {"ready", "error"} else dispatcher.submit
        submit(old_job)
        assert started.wait(5)
        dispatcher.restart()
        release.set()
        rid = DesktopApi(dispatcher.renderer, dispatcher).prepare()["request_id"]
        assert dispatcher.wait_idle(10)
        assert events and all(e["generation"] == 2 for e in events)
        assert dispatcher.outcome(rid) == "delivered"
    finally:
        release.set()
        dispatcher.close()


def test_restart_and_close_confirm_actual_child_exit():
    import psutil

    events = []
    started = threading.Event()

    def emit(event):
        events.append(event)
        started.set()

    dispatcher = ProcessDispatcher(emit, factory_name="process_fixtures:SlowReadyRenderer")
    api = DesktopApi(dispatcher.renderer, dispatcher)
    try:
        api.prepare()
        assert started.wait(10)
        old = psutil.Process(dispatcher.pid)
        dispatcher.restart()
        assert not old.is_running()
        assert dispatcher.pid is None
        checkpoint = len(events)
        api.prepare()
        assert dispatcher.wait_idle(10)
        assert all(e["generation"] == 2 for e in events[checkpoint:])
        child = psutil.Process(dispatcher.pid)
    finally:
        dispatcher.close()
    assert not child.is_running()
    with pytest.raises(RuntimeError, match="closed"):
        api.prepare()


def test_close_stops_a_child_during_preparation_and_suppresses_late_ready():
    import psutil

    events = []
    started = threading.Event()

    def emit(event):
        events.append(event)
        started.set()

    dispatcher = ProcessDispatcher(emit, factory_name="process_fixtures:SlowReadyRenderer")
    api = DesktopApi(dispatcher.renderer, dispatcher)
    try:
        api.prepare()
        assert started.wait(10)
        child = psutil.Process(dispatcher.pid)
        dispatcher.close()
        assert not child.is_running()
        assert not any(e["type"] == "ready" for e in events)
        assert dispatcher.wait_idle(5)
    finally:
        dispatcher.close()


def _await_progress(events, rid, timeout=20):
    deadline = time.monotonic() + timeout
    while not any(e["type"] == "progress" and e["request_id"] == rid for e in events):
        assert time.monotonic() < deadline, events
        time.sleep(0.02)


def _hang(dispatcher, api, events):
    """Start an explanation in a child that reports once, then never answers."""
    rid = api.explain(DNA, "Cas12a|+|16", "Cas12a", "ig")["request_id"]
    _await_progress(events, rid)
    return rid


@pytest.mark.parametrize("then", ["cancel", "new-request"])
def test_unresponsive_child_is_force_restarted_in_the_same_family_after_the_grace(then):
    import psutil

    events = []
    dispatcher = ProcessDispatcher(events.append, cancel_grace=0.5,
                                   factory_name="process_fixtures:HangsRenderer")
    api = DesktopApi(dispatcher.renderer, dispatcher)
    try:
        assert api.prepare()["request_id"] and dispatcher.wait_idle(20)
        family = dispatcher.family
        stuck = _hang(dispatcher, api, events)
        old = psutil.Process(dispatcher.pid)
        checkpoint = len(events)
        if then == "cancel":
            api.cancel()
        else:
            superseded = api.predict(DNA)["request_id"]
        assert dispatcher.wait_idle(20)
        assert not old.is_running()
        assert dispatcher.outcome(stuck) == "cancelled"
        if then == "new-request":
            # The pending request belonged to the replaced generation; the
            # screen asks again once the new process is ready.
            assert dispatcher.outcome(superseded) == "superseded"
        after = events[checkpoint:]
        restarting = [e for e in after if e["type"] == "restarting"]
        assert len(restarting) == 1, after
        assert restarting[0] == {"type": "restarting", "reason": "cancel_grace",
                                 "request_id": restarting[0]["request_id"],
                                 "generation": 2, "family": family}
        assert after[-1]["type"] == "ready"
        assert after[-1]["request_id"] == restarting[0]["request_id"]
        assert all(e["generation"] == 2 for e in after)
        assert dispatcher.generation == 2 and dispatcher.family == family
        rid = api.predict(DNA)["request_id"]
        assert dispatcher.wait_idle(20)
        assert dispatcher.outcome(rid) == "delivered"
        child = psutil.Process(dispatcher.pid)
        assert child.pid != old.pid
    finally:
        dispatcher.close()
    assert not child.is_running()
    assert dispatcher.pid is None


def test_cooperative_cancel_within_the_grace_keeps_the_generation_and_child():
    events = []
    dispatcher = ProcessDispatcher(events.append, preview=True, cancel_grace=0.5)
    api = DesktopApi(dispatcher.renderer, dispatcher)
    calls = {
        "predict": lambda: api.predict(DNA),
        "ig": lambda: api.explain(DNA, "Cas12a|+|16", "Cas12a", "ig", {"steps": 2048}),
        "shap": lambda: api.explain(DNA, "Cas12a|+|16", "Cas12a", "shap",
                                    {"nsamples": 1024, "n_background": 32}),
    }
    try:
        assert api.prepare()["request_id"] and dispatcher.wait_idle(20)
        pid = dispatcher.pid
        for call in calls.values():
            rid = call()["request_id"]
            _await_progress(events, rid)
            api.cancel()
            assert dispatcher.wait_idle(20)
            assert dispatcher.outcome(rid) == "cancelled"
        time.sleep(1)  # longer than the grace: an answered cancel never restarts
        assert dispatcher.pid == pid and dispatcher.generation == 1
        assert not any(e["type"] == "restarting" for e in events)
    finally:
        dispatcher.close()
