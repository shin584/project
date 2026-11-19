"""Explicit real-model #198 window acceptance; run separately from regressions.

python -X utf8 tests/family_window_probe.py OUTPUT_DIRECTORY
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(output):
    import psutil
    import webview

    from bundle_runtime import configure_offline
    from desktop import computation
    from desktop.app import main

    configure_offline()
    output.mkdir(parents=True, exist_ok=True)
    dna = json.loads((ROOT / "docs/equivalence-baseline/capture.json").read_text(encoding="utf-8"))["input_seq"]
    report = {"issue": 198, "status": "fail", "stages": [], "peak_tree_rss_bytes": {}}
    owners = []
    events = []
    stage = "startup"
    finished = threading.Event()
    original_owner = computation.ProcessDispatcher

    def observe_owner(emit, **kwargs):
        def record(event):
            events.append(event)
            emit(event)
        owner = original_owner(record, **kwargs)
        owners.append(owner)
        return owner

    computation.ProcessDispatcher = observe_owner
    original_window = webview.create_window

    def observe_window(*args, **kwargs):
        window = original_window(*args, **kwargs)
        window.create_file_dialog = lambda *a, **k: str(output / (stage + ".csv"))

        def wait(expression):
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if window.evaluate_js(expression):
                    return
                errors = [e for e in events if e["type"] == "error"]
                if errors:
                    raise AssertionError(errors[-1])
                time.sleep(.1)
            raise TimeoutError(expression)

        def drive():
            nonlocal stage
            try:
                wait("!document.getElementById('analyze').disabled")
                assert window.evaluate_js("document.getElementById('model-family').value") == "non_nt"
                assert owners[0].family == "non_nt"
                previous = None
                for family in ("non_nt", "nt", "non_nt"):
                    stage = f"{len(report['stages']) + 1}-{family}"
                    began = time.monotonic()
                    if previous is None:
                        window.evaluate_js("document.getElementById('dna').value = " + json.dumps(dna)
                                           + "; document.getElementById('analyze').click()")
                    else:
                        window.evaluate_js("""(() => {
                          const select = document.getElementById('model-family');
                          select.value = """ + json.dumps(family) + """;
                          select.dispatchEvent(new Event('change'));
                        })()""")
                        assert window.evaluate_js("document.getElementById('analyze').disabled")
                    wait("""!document.getElementById('analyze').disabled &&
                      document.getElementById('task-state').hidden &&
                      document.querySelector('#detail-results tbody tr') !== null""")
                    owner = owners[0]
                    child = psutil.Process(owner.pid)
                    if previous is not None:
                        assert not previous.is_running(), "previous family still owns memory"
                    assert window.evaluate_js("document.getElementById('dna').value") == dna
                    for tab in ("best", "variants", "detail", "xai"):
                        window.evaluate_js(f"document.getElementById('tab-{tab}').click()")
                        content = window.evaluate_js(f"document.getElementById('view-{tab}').innerHTML")
                        (output / (stage + "-" + tab + ".html")).write_text(content, encoding="utf-8")
                        assert content.strip()
                    if family == "non_nt":
                        assert not window.evaluate_js("Boolean(document.querySelector('[data-slot=xai] img'))")
                        assert "non-NT" in window.evaluate_js("document.querySelector('[data-slot=xai]').textContent")
                    window.evaluate_js("document.getElementById('detail-search').click(); document.getElementById('detail-csv').click()")
                    wait("document.getElementById('notice').textContent.includes('저장했습니다:')")
                    csv = (output / (stage + ".csv")).read_text(encoding="utf-8-sig")
                    assert ("NT SpCas9" in csv) == (family == "nt")
                    assert ("9-Hydra v2" in csv) == (family == "non_nt")
                    if family == "nt":
                        from desktop.detail_view import csv_against_baseline
                        comparison = csv_against_baseline(
                            (output / (stage + ".csv")).read_bytes(),
                            (ROOT / "docs/equivalence-baseline/crispr_prediction_results.csv").read_bytes())
                        assert comparison["csv_base_columns_byte_equal"], comparison
                        report["nt_csv_base_columns_byte_equal"] = True
                    elif previous is not None:
                        assert (output / (stage + ".csv")).read_bytes() == (output / "1-non_nt.csv").read_bytes()
                        report["non_nt_return_csv_byte_equal"] = True
                    checkpoint = len(events)
                    generation, pid = owner.generation, owner.pid
                    window.evaluate_js("""
                      document.getElementById('model-family').dispatchEvent(new Event('change'));
                      document.querySelectorAll('[role=tab]').forEach(tab => tab.click());
                      const slider = document.getElementById('graph-size');
                      slider.value = '40'; slider.dispatchEvent(new Event('input'));
                    """)
                    time.sleep(.25)
                    assert len(events) == checkpoint and owner.generation == generation and owner.pid == pid
                    report["stages"].append({"family": family, "pid": child.pid,
                                             "generation": generation, "seconds": round(time.monotonic() - began, 3)})
                    previous = child
                assert not {"torch", "keras", "transformers", "predict", "numpy", "matplotlib"}.intersection(sys.modules)
                report.update(status="pass", ui_has_no_model_imports=True)
            except Exception:
                report["error"] = traceback.format_exc()
            finally:
                finished.set()
                window.destroy()

        window.events.loaded += lambda: threading.Thread(target=drive, daemon=True).start()
        return window

    def sample():
        ui = psutil.Process()
        while not finished.wait(.1):
            total = 0
            for process in [ui, *ui.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peaks = report["peak_tree_rss_bytes"]
            peaks[stage] = max(peaks.get(stage, 0), total)

    webview.create_window = observe_window
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        main()
    finally:
        finished.set()
        monitor.join(2)
        report["events"] = [{k: v for k, v in event.items() if k not in {"fragments", "html"}} for event in events]
        report["child_closed"] = bool(owners) and owners[0].pid is None
        report["after_close_tree_rss_bytes"] = psutil.Process().memory_info().rss
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]).resolve()))
