"""#198 at the existing #194 public API/spawned-process seam."""

import threading

import psutil
import pytest

from desktop.api import DesktopApi
from desktop.computation import ProcessDispatcher

DNA = "GATCCAGTCAGCATGCTTTACGATCGTACGATCGAATCGTAGGCTAGCATCGATCGATGCATCGTAGCTAGCTAGCATCGA"


def test_family_switch_replaces_the_child_and_reselection_keeps_it():
    events = []
    dispatcher = ProcessDispatcher(events.append, preview=True)
    api = DesktopApi(dispatcher.renderer, dispatcher)
    try:
        assert dispatcher.family == "non_nt"
        api.prepare()
        assert dispatcher.wait_idle(10)
        original = psutil.Process(dispatcher.pid)
        reply = api.select_family("nt")
        assert not original.is_running(), "old child must exit before selection returns"
        assert dispatcher.wait_idle(10)
        assert reply["family"] == "nt" and reply["changed"]
        assert events[-1]["type"] == "ready"
        assert events[-1]["generation"] == reply["generation"]
        switch_events = [event for event in events if event["request_id"] == reply["request_id"]]
        assert [event["type"] for event in switch_events] == ["progress", "progress", "ready"]
        assert [(event["done"], event["total"]) for event in switch_events[:-1]] == [(1, 3), (2, 3)]
        nt = psutil.Process(dispatcher.pid)
        again = api.select_family("nt")
        assert dispatcher.wait_idle(10)
        assert not again["changed"] and again["generation"] == reply["generation"]
        assert dispatcher.pid == nt.pid
        assert not any(e["type"] == "progress" and e["request_id"] == again["request_id"] for e in events)
        with pytest.raises(ValueError, match="family"):
            api.select_family("unknown")
        assert dispatcher.pid == nt.pid
        api.select_family("non_nt")
        assert not nt.is_running()
        assert dispatcher.wait_idle(10)
        final = psutil.Process(dispatcher.pid)
    finally:
        dispatcher.close()
    assert not final.is_running()


@pytest.mark.parametrize("operation", ["prepare", "predict", "explain", "save_csv"])
def test_switch_during_work_discards_old_events_and_pending_csv(operation, tmp_path):
    events = []
    started = threading.Event()

    def emit(event):
        events.append(event)
        if event["type"] == "progress":
            started.set()

    dispatcher = ProcessDispatcher(emit, factory_name="process_fixtures:SlowFamilyRenderer")
    saved = tmp_path / "old.csv"
    api = DesktopApi(dispatcher.renderer, dispatcher, lambda _: str(saved))
    calls = {"prepare": api.prepare, "predict": lambda: api.predict(DNA),
             "explain": lambda: api.explain(DNA, "Cas12a|+|16", "Cas12a", "ig"),
             "save_csv": lambda: api.save_csv(DNA)}
    try:
        rid = calls[operation]()["request_id"]
        assert started.wait(10)
        child = psutil.Process(dispatcher.pid)
        cut = len(events)
        api.select_family("nt")
        assert not child.is_running()
        api.select_family("non_nt")
        reply = api.select_family("nt")
        assert dispatcher.wait_idle(15)
        assert events[-1]["type"] == "ready"
        assert events[-1]["generation"] == reply["generation"]
        assert all(e["generation"] > 1 for e in events[cut:])
        assert not any(e["request_id"] == rid for e in events[cut:])
        assert not saved.exists()
    finally:
        dispatcher.close()


@pytest.mark.parametrize("action", ["switch", "cancel"])
def test_obsolete_save_dialog_cannot_save_after_switch_or_cancel(tmp_path, action):
    opened, release = threading.Event(), threading.Event()
    saved = tmp_path / "obsolete.csv"
    replies = []

    def choose(_):
        opened.set()
        assert release.wait(10)
        return str(saved)

    dispatcher = ProcessDispatcher(lambda event: None, preview=True)
    api = DesktopApi(dispatcher.renderer, dispatcher, choose)
    saving = threading.Thread(target=lambda: replies.append(api.save_csv(DNA)))
    try:
        saving.start()
        assert opened.wait(5)
        api.select_family("nt") if action == "switch" else api.cancel()
        release.set()
        saving.join(10)
        assert dispatcher.wait_idle(10)
        if action == "switch":
            assert replies == [{"request_id": None, "reason": "restarted",
                                "generation": dispatcher.generation, "family": "nt"}]
        else:
            assert replies == [{"request_id": None}]
        assert not saved.exists()
    finally:
        release.set()
        saving.join(10)
        dispatcher.close()


def test_save_from_an_obsolete_view_does_not_even_open_a_dialog():
    dialogs = []
    dispatcher = ProcessDispatcher(lambda event: None, preview=True)
    api = DesktopApi(dispatcher.renderer, dispatcher, lambda name: dialogs.append(name))
    try:
        old_generation = dispatcher.generation
        api.select_family("nt")
        assert api.save_csv(DNA, None, old_generation) == {
            "request_id": None, "reason": "restarted",
            "generation": dispatcher.generation, "family": "nt"}
        assert dialogs == []
    finally:
        dispatcher.close()


@pytest.mark.parametrize("new_intent", ["cancel", "predict"])
def test_save_dispatched_late_cannot_adopt_a_newer_same_family_request(new_intent):
    dialogs = []
    dispatcher = ProcessDispatcher(lambda event: None, preview=True)
    api = DesktopApi(dispatcher.renderer, dispatcher, lambda name: dialogs.append(name))
    try:
        context = api.save_context()
        api.cancel() if new_intent == "cancel" else api.predict(DNA)
        assert api.save_csv(DNA, None, context["generation"], context["revision"]) == {"request_id": None}
        assert dialogs == []
    finally:
        dispatcher.close()


def test_save_after_the_idle_child_died_is_refused_then_succeeds_after_reanalysis(tmp_path):
    """#217: a replaced child refuses the save visibly; the reply carries the new generation."""
    events = []
    saved = tmp_path / "results.csv"
    dialogs = []

    def choose(name):
        dialogs.append(name)
        return str(saved)

    dispatcher = ProcessDispatcher(events.append, preview=True)
    api = DesktopApi(dispatcher.renderer, dispatcher, choose)
    try:
        api.prepare()
        assert dispatcher.wait_idle(10)
        child = psutil.Process(dispatcher.pid)
        child.kill()
        child.wait(10)
        # The screen sends the generation it last saw, plus save_context's revision.
        seen = dispatcher.generation
        context = api.save_context()
        reply = api.save_csv(DNA, None, seen, context["revision"])
        assert reply == {"request_id": None, "reason": "restarted",
                         "generation": seen + 1, "family": "non_nt"}
        assert dialogs == [], "a refused save must not ask for a path"
        assert dispatcher.wait_idle(10)
        assert not saved.exists()
        # Reanalysis in the new generation, then saving with the refreshed one.
        rid = api.predict(DNA)["request_id"]
        assert dispatcher.wait_idle(20)
        assert dispatcher.outcome(rid) == "delivered"
        context = api.save_context()
        assert context["generation"] == reply["generation"]
        rid = api.save_csv(DNA, None, context["generation"], context["revision"])["request_id"]
        assert dispatcher.wait_idle(20)
        assert dispatcher.outcome(rid) == "delivered"
        assert saved.read_bytes().startswith(b"\xef\xbb\xbf")
    finally:
        dispatcher.close()
