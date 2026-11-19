"""#197 actual child-process predictions, selection and filtered CSV."""

import csv
import io
import json
import os
from pathlib import Path
from html.parser import HTMLParser

import psutil

from desktop.api import DesktopApi
from desktop.computation import ProcessDispatcher

DNA = "A" * 16 + "TTTA" + "A" * 61
ROOT = Path(__file__).resolve().parents[1]


def test_real_cas12a_only_child_loads_lazily_and_connects_views_selection_and_csv(tmp_path, monkeypatch):
    access_log = tmp_path / "model-access.log"
    monkeypatch.setenv("ISSUE197_MODEL_ACCESS_LOG", str(access_log))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    events = []
    saved = tmp_path / "predictions.csv"
    dispatcher = ProcessDispatcher(events.append, factory_name="cas12a_process_fixtures:Cas12aOnlyRenderer")
    api = DesktopApi(dispatcher.renderer, dispatcher, lambda _: str(saved))

    def run(call, kind="result"):
        rid = call()["request_id"]
        assert dispatcher.wait_idle(120)
        received = [e for e in events if e["request_id"] == rid]
        assert received[-1]["type"] == kind, received
        return received[-1], received

    try:
        run(api.prepare, "ready")
        child = psutil.Process(dispatcher.pid)
        assert child.pid != os.getpid()
        assert not access_log.exists(), "preparation must not open model weights"
        # A no-site input must not trigger warmup as a side effect of a view.
        run(lambda: api.predict("A" * 81))
        assert not access_log.exists()
        scan, _ = run(lambda: api.scan_sites(DNA))
        assert "Cas12a|+|16" in scan["fragments"]["sites"]
        event, progress = run(lambda: api.predict(DNA))
        fragments = event["fragments"]
        assert set(fragments) == {"best", "variants", "detail"}
        assert "Cas12a CNN+BiLSTM manifest" in fragments["best"]
        assert 'data-site-key="Cas12a|+|16"' in fragments["detail"]
        assert 'data-heads="[&quot;Cas12a&quot;]"' in fragments["detail"]
        assert 'data-head="Cas12a"' in fragments["best"]
        assert any(e["type"] == "progress" for e in progress)
        assert set(Path(p).name for p in access_log.read_text().splitlines()) == {"cas12a_cnn_manifest.keras"}
        accesses = access_log.read_text()
        _, csv_events = run(lambda: api.save_csv(DNA, {"cas_names": ["Cas12a"]}))
        assert not any(e["type"] == "progress" for e in csv_events), "CSV should reuse predictions"
        table = list(csv.reader(io.StringIO(saved.read_text(encoding="utf-8-sig"))))
        assert table[0] == ["순위", "Cas 변종", "PAM", "가이드 서열 (RNA)", "가닥", "거리 (bp)",
                            "원출력", "예측 점수 (%)", "점수 기준", "모델", "모델 버전"]
        assert len(table) == 2
        assert table[1][:6] == ["1", "Cas12a", "TTTA", "A" * 23, "+", "0"]
        assert table[1][8:] == ["절단 효율", "Cas12a CNN+BiLSTM manifest",
                               "cas12a-cnn-20261102 · sha256 94ac977865ef"]
        reference = json.loads((ROOT / "docs/benchmarks/issue197-cas12a-reference.json").read_text())
        value = next(c["raw_score"] for c in reference["cases"] if c["window"] == "AAAATTTA" + "A" * 26)
        assert abs(float(table[1][6]) - value) <= 1e-4
        assert abs(float(table[1][7]) - value * 100) <= .01
        run(lambda: api.save_csv(DNA, {"cas_names": []}))
        assert len(list(csv.reader(io.StringIO(saved.read_text(encoding="utf-8-sig"))))) == 1
        explanation, ig_events = run(lambda: api.explain(DNA, "Cas12a|+|16", "Cas12a", "ig"))
        assert "data:image/png;base64," in explanation["fragments"]["xai"]
        assert "합성 경로" in explanation["fragments"]["xai"]
        assert any(e["type"] == "progress" for e in ig_events)
        assert access_log.read_text() == accesses, "view/CSV/XAI selection reloaded the checkpoint"
    finally:
        dispatcher.close()
    assert not child.is_running()


def test_mixed_cas_input_keeps_cas12a_scores_filters_ranks_and_head_selection():
    from desktop.worker_renderer import NonNTComputationRenderer

    class Options(HTMLParser):
        def __init__(self, fragment):
            super().__init__()
            self.sites = {}
            self.feed(fragment)

        def handle_starttag(self, tag, attrs):
            if tag == "option":
                data = dict(attrs)
                self.sites[data["value"]] = data

    dna = json.loads((ROOT / "docs/equivalence-baseline/capture.json").read_text())["input_seq"]
    renderer = NonNTComputationRenderer()
    fragments = renderer.prediction_fragments(dna, lambda d, t: None)
    assert "Cas12a는 변종이 없어" in fragments["variants"]
    selectors = Options(fragments["detail"]).sites
    table = renderer.csv_table(dna, lambda d, t: None, {"cas_names": ["Cas12a"]})
    assert [row[0] for row in table[1:]] == [1, 2]
    assert [row[7] for row in table[1:]] == [77.5, 63.05]
    for key, value in [("Cas12a|+|16", 77.50158905982971), ("Cas12a|-|28", 63.04842233657837)]:
        option = selectors[key]
        assert json.loads(option["data-heads"]) == ["Cas12a"]
        score, kind = json.loads(option["data-scores"])["Cas12a"]
        assert abs(score - value) < .001
        assert kind == "efficiency"
    assert all(row[-2] == "Cas12a CNN+BiLSTM manifest" for row in table[1:])
    filtered = renderer.csv_table(dna, lambda d, t: None,
                                 {"cas_names": ["Cas12a"], "eff_range": [70, 100]})
    assert filtered == [table[0], table[1]]
