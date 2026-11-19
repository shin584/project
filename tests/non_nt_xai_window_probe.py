"""#199 real WebView2/child-process acceptance, without showing a desktop window.

py -3.13 -X utf8 tests/non_nt_xai_window_probe.py OUTPUT_DIRECTORY
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DNA = "GATCCAGTCAGCATGCTTTACGATCGTACGATCGAATCGTAGGCTAGCATCGATCGATGCATCGTAGCTAGCTAGCATCGA"


def run(output):
    import webview
    from bundle_runtime import configure_offline
    from desktop.api import DesktopApi, webview_emitter
    from desktop.computation import ProcessDispatcher
    from desktop.screen import SCREEN_HTML

    configure_offline()
    output.mkdir(parents=True, exist_ok=False)
    events = []
    report = {"issue": 199, "status": "fail", "checks": []}
    window = webview.create_window("#199 acceptance", html=SCREEN_HTML, hidden=True,
                                   width=1280, height=840)
    if window is None:
        raise RuntimeError('Could not create the acceptance window')
    emitter = webview_emitter(window)

    def emit(event):
        events.append(event)
        emitter(event)

    dispatcher = ProcessDispatcher(emit, family="non_nt")
    api = DesktopApi(dispatcher.renderer, dispatcher)
    window.expose(api.prepare, api.select_family, api.scan_sites, api.predict, api.explain,
                  api.save_context, api.save_csv, api.cancel)

    def wait(expression):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if window.evaluate_js(expression):
                return
            errors = [e for e in events if e["type"] == "error"]
            if errors:
                raise AssertionError(errors[-1])
            time.sleep(.1)
        raise TimeoutError(expression)

    def check(name, passed):
        report["checks"].append({"check": name, "passed": bool(passed)})
        assert passed, name

    def capture(name):
        images = window.evaluate_js("Array.from(document.querySelectorAll('[data-slot=xai] img'), i => i.src)")
        for i, source in enumerate(images):
            (output / f"{name}-{i}.png").write_bytes(base64.b64decode(source.split(",", 1)[1]))
        return images

    def drive():
        try:
            window.evaluate_js("window.xaiDesktop.start()")
            wait("!document.getElementById('analyze').disabled")
            check("non_nt_rollout_disabled_with_reason", window.evaluate_js("""
              document.querySelector('[name=xai-method][value=rollout]').disabled &&
              !document.getElementById('rollout-reason').hidden &&
              document.querySelector('[name=xai-method]:checked').value === 'ig'
            """))
            check("defaults_unchanged", window.evaluate_js("""
              ['ig-steps','shap-n','shap-bg'].map(id => document.getElementById(id).value).join('/') === '32/64/4'
            """))
            window.evaluate_js("document.getElementById('dna').value=" + json.dumps(DNA) +
                               ";document.getElementById('analyze').click()")
            wait("document.querySelector('#xai-site option[value=\"SpCas9|+|40\"]') !== null")
            window.evaluate_js("""
              document.getElementById('tab-xai').click();
              const site = document.getElementById('xai-site'); site.value = 'SpCas9|+|40';
              site.dispatchEvent(new Event('change', {bubbles:true}));
              const head = document.getElementById('xai-head'); head.value = 'SpCas9-NG';
              head.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            wait("document.querySelectorAll('[data-slot=xai] img').length === 1 && document.getElementById('task-state').hidden")
            check("ig_logo_rendered", len(capture("ig")) == 1)
            before = len(events)
            first = capture("ig-before-resize")
            window.evaluate_js("""
              const slider = document.getElementById('graph-size'); slider.value = 40;
              slider.dispatchEvent(new Event('input'));
            """)
            check("resize_reuses_png_without_calculation", capture("ig-resized") == first and len(events) == before)
            window.evaluate_js("""
              const method = document.querySelector('[name=xai-method][value=shap]');
              method.checked = true; method.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            wait("document.querySelectorAll('[data-slot=xai] img').length === 2 && document.getElementById('task-state').hidden")
            check("shap_bars_and_waterfall_rendered", len(capture("shap")) == 2)
            text = window.evaluate_js("document.querySelector('[data-slot=xai]').textContent")
            check("units_seeds_and_causality_reference", all(s in text for s in ("제한 전", "%p", "배경 seed 0", "인과관계")))
            check("no_separate_quality_warning", window.evaluate_js("!document.querySelector('[data-slot=xai] .warning, [data-slot=xai] [role=alert]')"))
            window.evaluate_js("""
              document.getElementById('shap-n').value = 1024;
              document.getElementById('shap-bg').value = 32;
              document.getElementById('xai-controls').requestSubmit();
            """)
            wait("!document.getElementById('task-state').hidden && Number(document.getElementById('task-progress').value) > 0")
            old = events[-1]["request_id"]
            window.evaluate_js("""
              const site = document.getElementById('xai-site'); site.value = 'Cas12a|+|16';
              site.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            wait("document.querySelector('[data-slot=xai]').textContent.includes('#200') && document.getElementById('task-state').hidden")
            check("unsupported_site_cancels_previous_calculation", dispatcher.wait_idle(30) and dispatcher.outcome(old) == 'cancelled')
            check("unsupported_site_clears_previous_window_metrics_and_chart", window.evaluate_js("""
              !document.getElementById('xai-window').children.length &&
              !document.getElementById('xai-metrics').children.length &&
              !document.querySelector('[data-slot=xai] img') &&
              document.querySelector('[data-slot=xai]').textContent.includes('#200')
            """))
            window.evaluate_js("""
              document.getElementById('shap-n').value = 64;
              document.getElementById('shap-bg').value = 4;
              const site = document.getElementById('xai-site'); site.value = 'SpCas9|+|40';
              site.dispatchEvent(new Event('change', {bubbles:true}));
              const head = document.getElementById('xai-head'); head.value = 'SpCas9-NG';
              head.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            wait("document.querySelectorAll('[data-slot=xai] img').length === 2 && document.getElementById('task-state').hidden")
            progress_start = len(events)
            window.evaluate_js("""
              document.getElementById('shap-n').value = 1024;
              document.getElementById('shap-bg').value = 32;
              document.getElementById('xai-controls').requestSubmit();
            """)
            wait("!document.getElementById('task-state').hidden && Number(document.getElementById('task-progress').value) > 0")
            window.evaluate_js("document.getElementById('cancel-task').click()")
            wait("document.getElementById('task-state').hidden && document.getElementById('notice').textContent.includes('취소')")
            check("precision_progress_and_cancel", any(e["type"] == "progress" for e in events[progress_start:]))
            window.evaluate_js("""
              document.getElementById('shap-n').value = 64;
              document.getElementById('shap-bg').value = 4;
              document.getElementById('xai-controls').requestSubmit();
            """)
            wait("document.querySelectorAll('[data-slot=xai] img').length === 2 && document.getElementById('task-state').hidden")
            check("can_reselect_cached_default_after_cancel", len(capture("shap-after-cancel")) == 2)
            check("ui_has_no_model_libraries", not {"torch", "keras", "transformers", "numpy"}.intersection(sys.modules))
            report["status"] = "pass"
        except Exception:
            report["error"] = traceback.format_exc()
        finally:
            window.destroy()

    window.events.loaded += lambda: threading.Thread(target=drive, daemon=True).start()
    try:
        webview.start(gui="edgechromium", http_server=False, debug=False)
    finally:
        dispatcher.close()
        report["child_closed"] = dispatcher.pid is None
        report["events"] = [{k: v for k, v in event.items() if k not in ("fragments", "html")} for event in events]
        report["passed"] = sum(c["passed"] for c in report["checks"])
        report["failed"] = len(report["checks"]) - report["passed"]
        report["skipped"] = 0
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report["status"], report["passed"], "passed", report["failed"], "failed", flush=True)
    if "error" in report:
        print(report["error"], flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]).resolve()))
