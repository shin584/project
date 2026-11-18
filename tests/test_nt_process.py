"""Explicit real-NT #194 acceptance test. Requires the two local checkpoints.

Run this file separately from lightweight regressions. The UI-side test process
never imports model runtimes; all predictions, XAI and PNG generation use IPC.
"""

import base64
import json
import re
import struct
import sys
import threading
import time
from functools import partial
from html.parser import HTMLParser
from pathlib import Path

import psutil
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from desktop.api import DesktopApi
from desktop.computation import ProcessDispatcher


class Options(HTMLParser):
    def __init__(self, fragment):
        super().__init__()
        self.options = {}
        self.feed(fragment)

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == "option":
            self.options[data["value"]] = json.loads(data["data-heads"])


def test_real_nt_prediction_parity_and_process_owned_xai(tmp_path, monkeypatch):
    for cas in ("Cas12a", "SpCas9"):
        if not (ROOT / "models" / f"{cas}.safetensors").is_file():
            pytest.skip("Real NT checkpoints are not present")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    baseline = ROOT / "docs/equivalence-baseline"
    meta = json.loads((baseline / "capture.json").read_text(encoding="utf-8"))
    dna = meta["input_seq"]
    events = []
    dispatcher = ProcessDispatcher(events.append)
    saved = tmp_path / "predictions.csv"
    api = DesktopApi(dispatcher.renderer, dispatcher, lambda _: str(saved))
    stop = threading.Event()
    peaks = {}
    stage = "startup"
    ui = psutil.Process()

    def sample():
        while not stop.wait(0.1):
            total = 0
            for process in [ui, *ui.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peaks[stage] = max(peaks.get(stage, 0), total)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    evidence = {"issue": 194, "baseline": "docs/equivalence-baseline/capture.json",
                "prediction_tolerance": "CSV byte equality; baseline already rounds raw to 4 and score to 2 decimals",
                "xai_quality": "Execution/PNG/cache only; units and numerical quality are #201",
                "operations": []}

    def run(call):
        start = time.monotonic()
        rid = call()["request_id"]
        assert dispatcher.wait_idle(600), f"Timed out: {stage}"
        received = [e for e in events if e["request_id"] == rid]
        assert dispatcher.outcome(rid) == "delivered", received
        evidence["operations"].append({"stage": stage, "seconds": round(time.monotonic() - start, 3),
                                       "progress_events": sum(e["type"] == "progress" for e in received)})
        print(f"{stage}: {evidence['operations'][-1]}", flush=True)
        return received[-1], received

    try:
        run(api.prepare)
        child = psutil.Process(dispatcher.pid)
        stage = "scan"
        event, _ = run(lambda: api.scan_sites(dna))
        assert "Cas12a|+|16" in event["fragments"]["sites"]
        stage = "prediction"
        event, _ = run(lambda: api.predict(dna))
        assert set(event["fragments"]) == {"best", "variants", "detail"}
        options = Options(event["fragments"]["detail"]).options
        # Independent order from the preserved #181 fixed fixture.
        assert options["SpCas9|+|32"] == ["SpCas9-NG", "xCas9"]
        assert options["Cas12a|+|16"] == ["Cas12a"]
        stage = "csv"
        _, csv_events = run(lambda: api.save_csv(dna))
        assert saved.read_bytes() == (baseline / "crispr_prediction_results.csv").read_bytes()
        assert not any(e["type"] == "progress" for e in csv_events), "CSV recomputed predictions"
        for target in meta["xai"]:
            for method in ("ig", "shap", "rollout"):
                stage = target["cas_type"] + "/" + method
                event, _ = run(partial(api.explain, dna, target["site_key"], target["head"], method))
                fragment = event["fragments"]["xai"]
                pngs = re.findall('src="data:image/png;base64,([^"]+)"', fragment)
                assert len(pngs) == (1 if method == "ig" else 2)
                for encoded in pngs:
                    data = base64.b64decode(encoded)
                    assert data[:8] == b"\x89PNG\r\n\x1a\n"
                    width, height = struct.unpack(">II", data[16:24])
                    assert width >= 500 and height >= 200
                stage += "/resize"
                again, received = run(partial(api.explain, dna, target["site_key"], target["head"], method, size=40))
                assert again["fragments"]["xai"] == fragment.replace('width:80%', 'width:40%')
                assert not any(e["type"] == "progress" for e in received)
        assert not set(sys.modules).intersection({"torch", "transformers", "predict", "numpy", "matplotlib", "shap"})
        evidence.update(status="pass", child_pid=child.pid, generation=dispatcher.generation,
                        csv_bytes=len(saved.read_bytes()), csv_byte_equal=True,
                        ui_has_no_computation_imports=True)
    finally:
        dispatcher.close()
        stop.set()
        monitor.join(2)
        evidence["process_tree_peak_rss_bytes"] = peaks
        evidence["after_close_ui_rss_bytes"] = ui.memory_info().rss
        evidence["child_exited"] = dispatcher.pid is None
        output = ROOT / ".packaging/evidence/issue194-nt-process.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert not child.is_running()
