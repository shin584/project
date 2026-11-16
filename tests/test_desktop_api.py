"""#178 호출 경계 테스트. 가중치·torch·pywebview 없이 돈다.

확인하는 것
  1) 가짜 구현이 모델 가중치 없이 네 함수(탐색·예측·XAI·CSV) 모두 조각을 돌려준다
  2) 요청 2개를 연달아 보내면 앞의 것이 취소되고 뒤의 것만 결과를 돌려준다
  3) 지난 요청의 진행률·결과는 요청 ID로 폐기된다 (중간 취소가 안 되는 구간 포함)
  4) 계산은 언제나 작업자 1개에서만 돈다 (여러 스레드에서 동시에 불러도)
  5) 조각에 외부 자원·스크립트가 없고, 이벤트는 JSON으로 직렬화된다
"""

import json
import re
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop.api import (
    SLOTS,
    Cancelled,
    DesktopApi,
    Dispatcher,
    webview_emitter,
)
from desktop.fake import DETAIL_COLUMNS, FakeRenderer

# 81bp. + 가닥 16에 Cas12a TTTA, 40에 SpCas9 AGG가 있다 (#179 게이트 입력과 같은 배치).
DNA = (
    "GATCCAGTCAGCATGC"
    + "TTTA"
    + "CGATCGTACGATCGAATCGT"
    + "AGG"
    + "CTAGCATCGATCGATGCATCGTAGCTAGCTAGCATCGA"
)
assert len(DNA) == 81


class Recorder:
    """emit 싱크. 화면(JS)이 받았을 이벤트를 순서대로 모은다."""

    def __init__(self):
        self.events = []
        self._cv = threading.Condition()

    def __call__(self, event):
        json.dumps(event)  # evaluate_js로 넘어가므로 언제나 직렬화돼야 한다
        with self._cv:
            self.events.append(event)
            self._cv.notify_all()

    def wait_for(self, pred, timeout=10.0):
        with self._cv:
            ok = self._cv.wait_for(lambda: any(pred(e) for e in self.events), timeout)
        assert ok, f"이벤트가 오지 않았다. 받은 것: {self.events}"

    def of(self, rid):
        with self._cv:
            return [e for e in self.events if e["request_id"] == rid]

    def results(self):
        with self._cv:
            return [e for e in self.events if e["type"] == "result"]


def _is(kind, rid):
    return lambda e: e["type"] == kind and e["request_id"] == rid


@pytest.fixture
def rig(tmp_path):
    rec = Recorder()
    dispatcher = Dispatcher(rec)
    renderer = FakeRenderer(step_delay=0.002)
    saved = tmp_path / "out.csv"
    api = DesktopApi(renderer, dispatcher, ask_save_path=lambda name: str(saved))
    yield api, rec, dispatcher, saved
    dispatcher.close()


# ══════════════════════════════════════════════════════
# 1) 가짜 구현
# ══════════════════════════════════════════════════════


def test_fake_needs_no_model_code():
    assert "torch" not in sys.modules
    assert "predict" not in sys.modules
    assert "webview" not in sys.modules


def test_scan_sites_returns_sites_fragment(rig):
    api, rec, _, _ = rig
    rid = api.scan_sites(DNA)["request_id"]
    rec.wait_for(_is("result", rid))
    (result,) = [e for e in rec.of(rid) if e["type"] == "result"]
    html = result["fragments"]["sites"]
    assert "Cas12a|+|16" in html
    assert "SpCas9|+|40" in html


def test_predict_fills_three_views_and_reports_progress(rig):
    api, rec, _, _ = rig
    rid = api.predict(DNA)["request_id"]
    rec.wait_for(_is("result", rid))
    events = rec.of(rid)
    progress = [(e["done"], e["total"]) for e in events if e["type"] == "progress"]
    assert progress, "예측이 진행률을 보내지 않았다"
    assert all(0 <= d <= t for d, t in progress)
    assert [d for d, _ in progress] == sorted(d for d, _ in progress)
    assert progress[-1][0] == progress[-1][1]
    fragments = events[-1]["fragments"]
    assert set(fragments) == {"best", "variants", "detail"}
    assert "SpCas9(WT)" in fragments["variants"]


@pytest.mark.parametrize(
    "method,params",
    [
        ("ig", {"steps": 8}),
        ("shap", {"nsamples": 16, "n_background": 2}),
        ("rollout", {}),
    ],
)
def test_explain_returns_chart_fragment(rig, method, params):
    api, rec, _, _ = rig
    rid = api.explain(DNA, "Cas12a|+|16", "Cas12a", method, params)["request_id"]
    rec.wait_for(_is("result", rid))
    html = rec.of(rid)[-1]["fragments"]["xai"]
    assert "<svg" in html
    if method != "rollout":  # Rollout은 순전파 1회라 진행률이 없다
        assert any(e["type"] == "progress" for e in rec.of(rid))


def test_save_csv_writes_the_chosen_file(rig):
    api, rec, _, saved = rig
    rid = api.save_csv(DNA, "variants")["request_id"]
    rec.wait_for(_is("result", rid))
    header = saved.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.split(",") == DETAIL_COLUMNS  # app.py의 전체 상세 열과 같아야 한다
    assert str(saved) in rec.of(rid)[-1]["fragments"]["notice"]


def test_save_csv_dialog_closed_submits_nothing(tmp_path):
    rec = Recorder()
    dispatcher = Dispatcher(rec)
    try:
        api = DesktopApi(
            FakeRenderer(step_delay=0), dispatcher, ask_save_path=lambda n: None
        )
        assert api.save_csv(DNA, "variants") == {"request_id": None}
        assert rec.events == []
    finally:
        dispatcher.close()


def test_bad_input_becomes_an_error_event_and_worker_survives(rig):
    api, rec, _, _ = rig
    bad = api.predict("<b>ACGT</b>")["request_id"]
    rec.wait_for(_is("error", bad))
    (err,) = [e for e in rec.of(bad) if e["type"] == "error"]
    assert "<b>" not in err["html"], "사용자 입력이 이스케이프되지 않았다"
    ok = api.predict(DNA)["request_id"]
    rec.wait_for(_is("result", ok))


def test_unknown_fragment_slot_is_an_error_not_a_dead_worker(rig):
    """Renderer가 없는 자리에 조각을 보내면 화면에 오류로 보이고, 작업자는 산다."""
    api, rec, dispatcher, _ = rig
    bad = dispatcher.submit(lambda on_batch: {"nowhere": "<p>?</p>"})
    rec.wait_for(_is("error", bad))
    ok = api.predict(DNA)["request_id"]
    rec.wait_for(_is("result", ok))


def test_worker_survives_a_broken_event_sink():
    """창이 닫혀 evaluate_js가 터져도 작업자가 죽으면 안 된다.

    죽으면 이후 모든 요청이 영영 멈추고, 화면에는 아무 신호도 가지 않는다.
    """
    rec = Recorder()
    broken = {"on": True}

    def emit(event):
        if broken["on"] and event["type"] == "result":
            raise RuntimeError("window is gone")
        rec(event)

    dispatcher = Dispatcher(emit)
    try:
        first = dispatcher.submit(lambda on_batch: {"xai": "<p>1</p>"})
        assert dispatcher.wait_idle()
        assert dispatcher.outcome(first) == "failed"

        broken["on"] = False
        second = dispatcher.submit(lambda on_batch: {"xai": "<p>2</p>"})
        rec.wait_for(_is("result", second))
        assert dispatcher.outcome(second) == "delivered"
    finally:
        dispatcher.close()


def test_unknown_xai_method_is_an_error(rig):
    api, rec, _, _ = rig
    rid = api.explain(DNA, "Cas12a|+|16", "Cas12a", "lime", {})["request_id"]
    rec.wait_for(_is("error", rid))


def test_fragments_are_self_contained(rig):
    api, rec, _, _ = rig
    rids = [
        api.scan_sites(DNA)["request_id"],
    ]
    rec.wait_for(_is("result", rids[-1]))
    for call in (
        lambda: api.predict(DNA),
        lambda: api.explain(DNA, "SpCas9|+|40", "SpCas9(WT)", "ig", {"steps": 4}),
        lambda: api.explain(DNA, "SpCas9|+|40", "SpCas9(WT)", "shap", {"nsamples": 8}),
        lambda: api.explain(DNA, "SpCas9|+|40", "SpCas9(WT)", "rollout", {}),
    ):
        rid = call()["request_id"]
        rec.wait_for(_is("result", rid))
        rids.append(rid)
    for event in rec.results():
        for slot, html in event["fragments"].items():
            assert slot in SLOTS
            assert not re.search(r"<script|https?://|src=\"(?!data:)", html), slot


# ══════════════════════════════════════════════════════
# 2) 연달아 보낸 요청: 앞의 것은 취소, 뒤의 것만 결과
# ══════════════════════════════════════════════════════


def test_second_request_cancels_the_first(rig):
    api, rec, dispatcher, _ = rig
    first = api.explain(DNA, "Cas12a|+|16", "Cas12a", "ig", {"steps": 5000})[
        "request_id"
    ]
    rec.wait_for(_is("progress", first))  # 앞의 것이 실제로 계산 중일 때
    second = api.predict(DNA)["request_id"]
    rec.wait_for(_is("result", second))
    dispatcher.wait_idle()

    assert second > first
    assert dispatcher.outcome(first) == "cancelled"
    assert [e["request_id"] for e in rec.results()] == [second]


def test_back_to_back_without_waiting(rig):
    api, rec, dispatcher, _ = rig
    first = api.explain(DNA, "Cas12a|+|16", "Cas12a", "shap", {"nsamples": 4096})[
        "request_id"
    ]
    second = api.explain(DNA, "Cas12a|+|16", "Cas12a", "ig", {"steps": 8})["request_id"]
    rec.wait_for(_is("result", second))
    dispatcher.wait_idle()

    assert dispatcher.outcome(first) in {"cancelled", "superseded"}
    assert [e["request_id"] for e in rec.results()] == [second]


def test_cancel_stops_the_running_request(rig):
    api, rec, dispatcher, _ = rig
    rid = api.explain(DNA, "Cas12a|+|16", "Cas12a", "ig", {"steps": 5000})["request_id"]
    rec.wait_for(_is("progress", rid))
    api.cancel()
    dispatcher.wait_idle()
    assert dispatcher.outcome(rid) == "cancelled"
    seen = len(rec.events)
    assert rec.results() == []
    assert len(rec.events) == seen


# ══════════════════════════════════════════════════════
# 3) 지난 요청의 진행률·결과 폐기
# ══════════════════════════════════════════════════════


def test_uninterruptible_stale_work_is_discarded():
    """모델 로딩·Rollout처럼 중간 취소가 안 되는 구간이 끝까지 돌아도 버려진다."""
    rec = Recorder()
    dispatcher = Dispatcher(rec)
    started, release = threading.Event(), threading.Event()

    def stubborn(on_batch):
        on_batch(1, 3)
        started.set()
        release.wait(5)
        try:  # 취소 신호를 무시하고 끝까지 간다
            on_batch(2, 3)
        except Cancelled:
            pass
        return {"xai": "<p>late</p>"}

    try:
        old = dispatcher.submit(stubborn)
        assert started.wait(5)
        new = dispatcher.submit(lambda on_batch: {"xai": "<p>fresh</p>"})
        boundary = len(rec.events)
        release.set()
        rec.wait_for(_is("result", new))
        dispatcher.wait_idle()

        assert dispatcher.outcome(old) == "stale"
        assert all(e["request_id"] == new for e in rec.events[boundary:])
        assert [e for e in rec.of(old) if e["type"] == "result"] == []
        assert [(e["done"], e["total"]) for e in rec.of(old)] == [(1, 3)]
    finally:
        dispatcher.close()


def test_delivered_request_ids_never_go_backwards(rig):
    api, rec, dispatcher, _ = rig
    for steps in (2000, 1000, 500, 8):
        api.explain(DNA, "Cas12a|+|16", "Cas12a", "ig", {"steps": steps})
    dispatcher.wait_idle()
    ids = [e["request_id"] for e in rec.events]
    assert ids == sorted(ids)


# ══════════════════════════════════════════════════════
# 4) 단일 작업자
# ══════════════════════════════════════════════════════


def test_only_one_job_runs_at_a_time_across_threads():
    """pywebview는 노출 함수를 호출마다 다른 스레드에서 돌린다. 그래도 계산은 겹치지 않는다."""
    rec = Recorder()
    dispatcher = Dispatcher(rec)
    lock = threading.Lock()
    state = {"running": 0, "peak": 0, "threads": set()}

    def job(on_batch):
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
            state["threads"].add(threading.get_ident())
        try:
            for i in range(1, 6):
                threading.Event().wait(0.001)
                on_batch(i, 5)
        finally:
            with lock:
                state["running"] -= 1
        return {"xai": "<p>ok</p>"}

    ids, ids_lock = [], threading.Lock()

    def caller():
        for _ in range(10):
            rid = dispatcher.submit(job)
            with ids_lock:
                ids.append(rid)

    try:
        threads = [threading.Thread(target=caller) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dispatcher.wait_idle()

        assert state["peak"] == 1
        assert len(state["threads"]) == 1, "계산이 작업자 스레드 밖에서 돌았다"
        assert len(set(ids)) == len(ids), "요청 ID가 중복됐다"
        assert [e["request_id"] for e in rec.results()][-1] == max(ids)
    finally:
        dispatcher.close()


# ══════════════════════════════════════════════════════
# 5) pywebview 연결
# ══════════════════════════════════════════════════════


def test_webview_emitter_calls_the_js_entry_point():
    calls = []

    class _Window:
        def evaluate_js(self, script):
            calls.append(script)

    emit = webview_emitter(_Window())
    emit({"type": "result", "request_id": 3, "fragments": {"xai": '<p>"</p>'}})
    (script,) = calls
    assert script.startswith("window.xaiDesktop.onEvent(")
    payload = script[len("window.xaiDesktop.onEvent(") : -2]
    assert json.loads(payload)["fragments"]["xai"] == '<p>"</p>'
