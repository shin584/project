"""Dashboard v3 window evidence through its public JS/events (#194).

The UI imports no model libraries. Prediction parity uses the preserved CSV;
XAI checks execution, actual image display and reuse. Numerical quality is #201.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import threading
import time
import traceback
from pathlib import Path
from unittest.mock import patch


def validate(output: Path, *, preview: bool = False) -> int:
    from bundle_runtime import configure_offline, is_packaged, resource_root

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("검증 출력 폴더는 비어 있어야 합니다: " + str(output))
    configure_offline()
    baseline = resource_root() / ("equivalence-baseline" if is_packaged() else "docs/equivalence-baseline")
    meta = json.loads((baseline / "capture.json").read_text(encoding="utf-8"))
    report = {"issue": 194, "smoke_status": "fail", "preview": preview,
              "frozen": is_packaged(), "host": platform.node(),
              "python": platform.python_version(), "platform": platform.platform(),
              "gate_status": "pending_clean_environment_and_remaining_v3_tickets",
              "xai_quality": "pending #201; execution/display only",
              # #196 appended the model/version columns, so the preserved baseline
              # is compared on the existing columns only; the cut and the
              # comparison live in detail_view (docs/desktop-boundary.md).
              "prediction_tolerance": {"rtol": 1e-5, "atol": 1e-8,
                                       "csv": "byte equality on the base columns"},
              "other_pc_validation": "pending", "visual_review": "pending"}
    started = time.monotonic()
    try:
        _run(output, baseline, meta, report, preview=preview)
    except Exception:  # noqa: BLE001 -- persist failures at the executable boundary
        report["error"] = traceback.format_exc()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (output / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["smoke_status"] == "pass" else 1


def _run(output, baseline, meta, report, *, preview):
    import psutil
    import webview
    import webview.http

    from desktop import api, detail_view
    from desktop.app import main

    counts = {"events": 0, "progress": 0, "requests": 0}
    errors = []
    child_pids = set()
    peaks = []
    finished = threading.Event()
    original_create, original_emitter = webview.create_window, api.webview_emitter

    def emitter(window):
        deliver = original_emitter(window)

        def emit(event):
            counts["events"] += 1
            if event["type"] == "progress":
                counts["progress"] += 1
            if event["type"] == "error":
                errors.append(event["html"])
            deliver(event)

        return emit

    def create(*args, **kwargs):
        window = original_create(*args, **kwargs)
        if window is None:
            raise RuntimeError("No desktop window")
        window.create_file_dialog = lambda *a, **kw: str(output / "crispr_prediction_results.csv")
        original_expose = window.expose

        def expose(*functions):
            def observe(function):
                def call(*args):
                    counts["requests"] += 1
                    return function(*args)
                call.__name__ = function.__name__
                return call
            original_expose(*(observe(f) for f in functions))

        window.expose = expose

        def wait(expression):
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if errors:
                    raise RuntimeError(errors[-1])
                if window.evaluate_js(expression):
                    return
                time.sleep(0.1)
            raise TimeoutError(expression)

        def idle():
            wait("document.getElementById('task-state').hidden && !document.getElementById('analyze').disabled")

        def snapshot(name):
            markup = window.evaluate_js("document.documentElement.outerHTML")
            (output / (name + ".html")).write_text(markup, encoding="utf-8")

        def choose(method):
            window.evaluate_js(f"document.querySelector('[name=xai-method][value={method}]').click()")
            idle()
            wait("document.querySelector('[data-slot=xai] img')?.naturalWidth > 0")

        def drive():
            try:
                wait("!document.getElementById('analyze').disabled")
                if not preview:
                    # This preserved-baseline validator explicitly checks NT;
                    # the application itself now always starts with non-NT.
                    window.evaluate_js("""
                      const family = document.getElementById('model-family');
                      family.value = 'nt'; family.dispatchEvent(new Event('change'));
                    """)
                    wait("!document.getElementById('analyze').disabled")
                window.evaluate_js("document.getElementById('dna').value = " + json.dumps(meta["input_seq"])
                                   + "; document.getElementById('analyze').click()")
                wait("document.querySelector('[data-slot=xai] img')?.naturalWidth > 0")
                idle()
                for tab in ("best", "variants", "detail"):
                    window.evaluate_js(f"document.getElementById('tab-{tab}').click()")
                    if tab == "detail":
                        window.evaluate_js("document.getElementById('detail-search').click()")
                    snapshot(tab)
                window.evaluate_js("document.getElementById('detail-csv').click()")
                wait("document.getElementById('notice').textContent.includes('저장했습니다:')")
                idle()
                if not preview:
                    saved_csv = (output / "crispr_prediction_results.csv").read_bytes()
                    expected_csv = (baseline / "crispr_prediction_results.csv").read_bytes()
                    report.update(detail_view.csv_against_baseline(saved_csv, expected_csv))
                    if not report["csv_base_columns_byte_equal"]:
                        raise RuntimeError("NT prediction CSV differs from the fixed baseline")
                    reference = json.loads((baseline / "candidates.json").read_text(encoding="utf-8"))
                    actual = window.evaluate_js("""Array.from(document.querySelectorAll('#detail-results tbody tr')).map(r => ({
                      site: r.dataset.siteKey, head: r.dataset.head, raw: Number(r.dataset.raw), score: Number(r.dataset.eff)
                    }))""")
                    actual = {(row["site"], row["head"]): row for row in actual}
                    checked = 0
                    for group in reference["best_candidates"]:
                        for row in group["rows"]:
                            found = actual[(row["_site_key"], row["최적 Cas 변종"])]
                            assert math.isclose(found["raw"], row["_raw_score_full"], rel_tol=1e-5, abs_tol=1e-8)
                            assert math.isclose(found["score"], row["_efficiency_pct_full"], rel_tol=1e-5, abs_tol=1e-8)
                            checked += 1
                    report["unrounded_prediction_rows_checked"] = checked
                window.evaluate_js("document.getElementById('tab-xai').click()")
                for target in meta["xai"]:
                    window.evaluate_js("document.getElementById('xai-site').value = " + json.dumps(target["site_key"])
                                       + "; document.getElementById('xai-site').dispatchEvent(new Event('change', {bubbles:true}))")
                    idle()
                    for method in ("ig", "shap", "rollout"):
                        choose(method)
                        snapshot(target["cas_type"] + "-" + method)
                    progress_before = counts["progress"]
                    for method in ("ig", "shap", "rollout"):
                        choose(method)
                    if not preview:
                        assert counts["progress"] == progress_before, "Returning to an explanation recomputed it"
                before = dict(counts)
                window.evaluate_js("""
                  document.querySelectorAll('[role=tab]').forEach(t => t.click());
                  const slider = document.getElementById('graph-size');
                  slider.value = 40; slider.dispatchEvent(new Event('input'));
                """)
                time.sleep(0.3)
                assert before == counts, "Display changes sent calculation requests"
                assert window.evaluate_js("Array.from(document.querySelectorAll('[data-slot=xai] figure')).every(f => f.style.width === '40%')")
                report["ui_has_no_computation_imports"] = not set(sys.modules).intersection(
                    {"torch", "transformers", "predict", "matplotlib", "shap", "desktop.renderer"})
                assert report["ui_has_no_computation_imports"]
                report["http_server_created"] = webview.http.global_server is not None
                assert not report["http_server_created"]
                report.update(smoke_status="pass", method_return_uses_cache=True if not preview else "not_checked_for_fake",
                              display_changes_no_requests_or_compute=True, counts=dict(counts))
            except Exception:  # noqa: BLE001 -- save GUI/worker evidence before closing
                report["error"] = traceback.format_exc()
            finally:
                finished.set()
                window.destroy()

        window.events.loaded += lambda: threading.Thread(target=drive, daemon=True).start()
        return window

    def sample():
        ui = psutil.Process()
        while not finished.wait(0.1):
            total = 0
            for process in [ui, *ui.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                    if "--multiprocessing-fork" in process.cmdline():
                        child_pids.add(process.pid)
                except psutil.NoSuchProcess:
                    pass
            peaks.append(total)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        with patch.object(webview, "create_window", create), patch.object(api, "webview_emitter", emitter):
            main(preview=preview)
        if not finished.is_set():
            raise RuntimeError("Validation closed before completion or another desktop is running")
    finally:
        finished.set()
        monitor.join(2)
    report["process_tree_peak_rss_bytes"] = max(peaks, default=0)
    report["computation_child_pids"] = sorted(child_pids)
    report["computation_children_exited"] = bool(child_pids) and all(not psutil.pid_exists(pid) for pid in child_pids)
    if not report["computation_children_exited"]:
        report["smoke_status"] = "fail"
