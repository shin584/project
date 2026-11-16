"""Gate-2 evidence from the real desktop window, models and save-CSV path.

Only used with --validate. Observers forward public calls unchanged; they never
substitute predictions or copy numerical outputs from the Streamlit baseline.
A developer run cannot certify another PC or approve the visual checklist.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import tempfile
import threading
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def validate(output: Path) -> int:
    from bundle_runtime import configure_offline, is_packaged

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # No stale success artifacts may survive a failed invocation.
    if any(output.iterdir()):
        raise ValueError("검증 출력 폴더는 비어 있어야 합니다: " + str(output))
    configure_offline()
    report: dict = {
        "issue": 188, "smoke_status": "fail", "frozen": is_packaged(),
        "host": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "executable": sys.executable,
        "gate_status": "not_passed", "decision": "Streamlit 시연",
        "visual_review": "pending", "other_pc_validation": "pending",
    }
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="xai-gate2-hf-") as cache:
        if is_packaged():
            os.environ["HF_HOME"] = cache
            os.environ["HF_HUB_CACHE"] = str(Path(cache) / "hub")
        report["empty_hf_cache"] = is_packaged()
        try:
            _run(output, report)
        except Exception:  # noqa: BLE001 -- persist executable-boundary failures
            report["smoke_status"] = "fail"
            report["error"] = traceback.format_exc()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (output / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["smoke_status"] == "pass" else 1


def _run(output: Path, report: dict) -> None:
    import core
    import numpy as np

    # Keep numerical settings identical to the recorded CPU baseline.
    import torch
    import webview
    from bundle_runtime import is_packaged, resource_root, validate_bundle
    from xai import explain

    from desktop.app import main

    torch.set_num_threads(4)
    baseline = (resource_root() / "equivalence-baseline" if is_packaged()
                else resource_root() / "docs/equivalence-baseline")
    meta = json.loads((baseline / "capture.json").read_text(encoding="utf-8"))
    scans = []
    results: dict = {}
    calls = {"prediction": 0, "ig": 0, "shap": 0, "rollout": 0}
    bridge_calls: list[str] = []

    def observe_prediction(original):
        def invoke(*args, **kwargs):
            calls["prediction"] += 1
            result = original(*args, **kwargs)
            scans.append(result)
            return result
        return invoke

    def observe_explanation(original, method):
        def invoke(predictor, model_key, head, window, *args, **kwargs):
            result = original(predictor, model_key, head, window, *args, **kwargs)
            results[(model_key, head, window, method)] = result
            return result
        return invoke

    def count_compute(original, method):
        def invoke(*args, **kwargs):
            calls[method] += 1
            return original(*args, **kwargs)
        return invoke

    network_attempts = []

    def audit(event, args):
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo"}:
            network_attempts.append(event)
            raise RuntimeError("Offline desktop attempted " + event)

    sys.addaudithook(audit)
    if is_packaged():
        validate_bundle()

    original_create = webview.create_window
    finished = threading.Event()

    def create(*args, **kwargs):
        kwargs.update(width=1600, height=1200)
        window = original_create(*args, **kwargs)
        if window is None:
            raise RuntimeError("No desktop window")
        # Answer the OS dialog; the actual DesktopApi.save_csv still runs.
        window.create_file_dialog = lambda *a, **kw: str(output / "crispr_prediction_results.csv")
        expose = window.expose

        def expose_observed(*functions):
            def track(function):
                def invoke(*args):
                    bridge_calls.append(function.__name__)
                    return function(*args)
                invoke.__name__ = function.__name__
                return invoke
            expose(*(track(f) for f in functions))

        window.expose = expose_observed

        def wait(expression, timeout=600):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                error = window.evaluate_js("document.querySelector('#notice .error')?.textContent")
                if error:
                    raise RuntimeError(error)
                value = window.evaluate_js(expression)
                if value:
                    return value
                time.sleep(0.1)
            raise TimeoutError(expression)

        def idle():
            wait("document.getElementById('task-state').hidden && !document.getElementById('analyze').disabled")

        def snapshot(name):
            # Serialize *live* controls, including the selected tab and filters.
            html = window.evaluate_js("""(() => {
              const copy = document.documentElement.cloneNode(true);
              const inputs = document.querySelectorAll('input,select');
              copy.querySelectorAll('input,select').forEach((el, i) => {
                el.setAttribute('value', inputs[i].value);
                if (inputs[i].checked) el.setAttribute('checked','');
                else el.removeAttribute('checked');
                if (el.tagName === 'SELECT') Array.from(el.options).forEach(o => {
                  o.toggleAttribute('selected', o.value === inputs[i].value);
                });
              });
              copy.querySelectorAll('script').forEach(e => e.remove());
              return '<!doctype html>' + copy.outerHTML;
            })()""")
            (output / (name + ".html")).write_text(html, encoding="utf-8")

        def drive():
            try:
                wait("!document.getElementById('analyze').disabled")
                window.evaluate_js("document.getElementById('dna').value = "
                                   + json.dumps(meta["input_seq"])
                                   + "; document.getElementById('analyze').click()")
                wait("document.querySelector('[data-slot=xai] img')?.naturalWidth > 0")
                idle()
                report["controls"] = window.evaluate_js("""({
                  graph: ['min','max','step','value'].map(k => document.getElementById('graph-size')[k]),
                  settings: ['ig-steps','shap-n','shap-bg'].map(id =>
                    ['min','max','step','value'].map(k => document.getElementById(id)[k])),
                  selected_site: document.getElementById('xai-site').value
                })""")
                if report["controls"] != {
                    "graph": ["40", "100", "5", "80"],
                    "settings": [["8", "256", "8", "32"], ["16", "1024", "16", "64"], ["2", "32", "2", "4"]],
                    "selected_site": meta["xai"][0]["site_key"],
                }:
                    raise RuntimeError("Desktop controls differ from Streamlit defaults/ranges")
                for tab, name in [("best", "01-best-candidates"), ("variants", "02-variant-heatmap"),
                                  ("detail", "03-detail")]:
                    window.evaluate_js(f"document.getElementById('tab-{tab}').click()")
                    if tab == "detail":
                        window.evaluate_js("document.getElementById('detail-search').click()")
                    snapshot(name)
                _capture_candidates(output, window, scans[0], meta["input_seq"])
                window.evaluate_js("document.getElementById('detail-csv').click()")
                wait("document.getElementById('notice').textContent.includes('저장했습니다:')")
                idle()
                window.evaluate_js("document.getElementById('tab-xai').click(); document.querySelector('#xai-controls details').open = true")
                for target in meta["xai"]:
                    window.evaluate_js("""(() => {
                      const select = document.getElementById('xai-site');
                      select.value = """ + json.dumps(target["site_key"]) + "; select.dispatchEvent(new Event('change', {bubbles:true})); })()")
                    idle()
                    if window.evaluate_js("document.getElementById('xai-head').value") != target["head"]:
                        raise RuntimeError("Desktop selected a different explanation head")
                    for method, name in [("rollout", "04-xai"), ("ig", "05-xai-ig"), ("shap", "06-xai-shap")]:
                        window.evaluate_js(f"document.querySelector('[name=xai-method][value={method}]').click()")
                        idle()
                        wait("document.querySelector('[data-slot=xai] img')?.naturalWidth > 0")
                        snapshot(name + ("-Cas12a" if target["cas_type"] == "Cas12a" else ""))
                    site = next(s for s in scans[0].sites if s.site_key == target["site_key"])
                    key = ("A/" + site.cas_type, target["head"], site.model_input_seq)
                    ig, shap, rollout = (results[(*key, m)] for m in ("ig", "shap", "rollout"))
                    np.savez(output / target["file"], window_seq=np.array(site.model_input_seq),
                             ig_attribution=ig.attribution, shap_values=shap.values,
                             rollout_position_profile=rollout.position_profile,
                             rollout_token_matrix=rollout.token_matrix,
                             rollout_token_labels=np.array(rollout.token_labels))
                before = (dict(calls), len(bridge_calls))
                window.evaluate_js("""
                  document.querySelectorAll('[role=tab]').forEach(t => t.click());
                  const slider = document.getElementById('graph-size');
                  slider.value = 40; slider.dispatchEvent(new Event('input'));
                """)
                time.sleep(0.5)
                report["display_changes_no_requests_or_compute"] = before == (calls, len(bridge_calls))
                report["resize"] = window.evaluate_js("""({
                  label: document.getElementById('graph-size-value').textContent,
                  widths: Array.from(document.querySelectorAll('[data-slot=xai] figure')).map(f => f.style.width)
                })""")
                if report["resize"]["label"] != "40%" or not report["resize"]["widths"] or any(w != "40%" for w in report["resize"]["widths"]):
                    raise RuntimeError("Graph size did not update")
                before = dict(calls)
                for method in ("ig", "rollout", "shap"):
                    window.evaluate_js(f"document.querySelector('[name=xai-method][value={method}]').click()")
                    idle()
                report["method_return_uses_cache"] = before == calls
                report["compute_calls"] = dict(calls)
                report["blocked_python_network_attempts"] = network_attempts
                import psutil
                import webview.http

                process = psutil.Process()
                report["tcp_listeners_at_completion"] = [
                    connection.laddr.port
                    for proc in [process, *process.children(recursive=True)]
                    for connection in proc.net_connections(kind="tcp")
                    if connection.status == psutil.CONN_LISTEN
                ]
                report["http_server_created"] = webview.http.global_server is not None
                if report["tcp_listeners_at_completion"] or report["http_server_created"]:
                    raise RuntimeError("Desktop opened a server")
                module_spec = importlib.util.spec_from_file_location("gate2_compare", baseline / "compare_baseline.py")
                assert module_spec and module_spec.loader
                comparator = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(comparator)
                report["numeric_comparison_exit"] = comparator.main(["compare_baseline", str(output)])
                if report["numeric_comparison_exit"] != 0:
                    raise RuntimeError("Streamlit numerical baseline differs")
                if not report["display_changes_no_requests_or_compute"] or not report["method_return_uses_cache"]:
                    raise RuntimeError("Display changes recomputed model results")
                if network_attempts:
                    raise RuntimeError("Python network attempts recorded")
                report["smoke_status"] = "pass"
            except Exception:  # noqa: BLE001 -- persist GUI/worker failures
                report["error"] = traceback.format_exc()
            finally:
                finished.set()
                window.destroy()

        window.events.loaded += lambda: threading.Thread(target=drive, daemon=True).start()
        return window

    with ExitStack() as stack:
        stack.enter_context(patch.object(core, "run_prediction", observe_prediction(core.run_prediction)))
        for method, name in [("ig", "integrated_gradients"), ("shap", "kernel_shap"), ("rollout", "attention_rollout")]:
            stack.enter_context(patch.object(core.EXPLAIN_CACHE, name, observe_explanation(getattr(core.EXPLAIN_CACHE, name), method)))
            stack.enter_context(patch.object(explain, name, count_compute(getattr(explain, name), method)))
        stack.enter_context(patch.object(webview, "create_window", create))
        main()
    if not finished.is_set():
        raise RuntimeError("Validation window closed before checks completed (or desktop already running)")


def _capture_candidates(output: Path, window, scan, dna: str) -> None:
    """Use rendered table cells/order, retaining full precision from their rows."""
    raw = window.evaluate_js("""(() => {
      const tables = selector => Array.from(document.querySelectorAll(selector)).map(t => ({
        columns: Array.from(t.querySelectorAll('thead th')).map(c => c.textContent),
        rows: Array.from(t.querySelectorAll('tbody tr')).filter(r => !r.hidden).map(r => ({
          key: r.dataset.siteKey, head: r.dataset.head, selected: r.classList.contains('selected'),
          cells: Array.from(r.children).map(c => c.textContent)
        }))
      }));
      return {selected: document.getElementById('xai-site').value,
        best: tables('#best table'), variants: tables('#variants table'), detail: tables('#detail-results table')};
    })()""")
    rows_by_key = {(r.site_key, r.cas_name): r for r in scan.variant_results}
    best = []
    for table in raw["best"]:
        label = table["columns"][6]
        items = []
        for row in table["rows"]:
            cells = row["cells"]
            vr = rows_by_key[(row["key"], row["head"])]
            values = [int(cells[0]), *cells[1:5], int(cells[5]), float(cells[6].rstrip("%")), float(cells[7])]
            items.append(dict(zip(table["columns"], values)) | {
                "_site_key": vr.site_key, "_efficiency_pct_full": vr.efficiency_pct,
                "_raw_score_full": vr.raw_score,
            })
        best.append({"cas": "", "score_label": label, "total_sites": len(items), "shown": len(items), "rows": items})
    heatmap = raw["variants"][0]
    heat_rows = [{"guide_label": r["cells"][0], **{
        col: None if value == "None" else float(value.rstrip("%"))
        for col, value in zip(heatmap["columns"][1:], r["cells"][1:])
    }} for r in heatmap["rows"]]
    # Heatmap text has one decimal; full-precision pivot values are independently
    # checked via the prediction rows while screenshot text is recorded in HTML.
    for item, rendered in zip(heat_rows, heatmap["rows"]):
        for col in heatmap["columns"][1:]:
            vr = rows_by_key.get((rendered["key"], col))
            if vr is not None:
                if item[col] != float(f"{round(vr.efficiency_pct, 2):.1f}"):
                    raise ValueError("Rendered heatmap score differs from prediction")
                item[col] = round(vr.efficiency_pct, 2)
    detail = raw["detail"][0]
    detail_rows = []
    for r in detail["rows"]:
        c = r["cells"]
        detail_rows.append(dict(zip(detail["columns"], [int(c[0]), *c[1:5], int(c[5]), float(c[6]), float(c[7].rstrip("%")), c[8]])))
    data = {
        "input_seq": dna, "selected_site_key": raw["selected"],
        "sites": [{name: getattr(site, name) for name in (
            "site_key", "cas_type", "strand", "pam_seq", "pam_start", "distance", "guide_seq",
            "model_input_seq", "raw_score", "efficiency_pct", "score_kind"
        )} for site in scan.sorted_by_score()],
        "best_candidates": best,
        "variant_heatmap": {"columns": heatmap["columns"][1:], "total": len(heat_rows), "shown": len(heat_rows), "rows": heat_rows},
        "detail": {"rows": detail_rows, "highlighted_rows": [i for i, r in enumerate(detail["rows"], 1) if r["selected"]], "dropped_by_default_filter": len(scan.variant_results) - len(detail_rows)},
    }
    (output / "candidates.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
