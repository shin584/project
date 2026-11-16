"""Drive the real Windows shell in its own process, without model libraries.

Only the external WebView boundary is instrumented: window creation and exposed
prepare are forwarded unchanged after recording the initial visible state.
"""

from __future__ import annotations

import importlib.abc
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

report: dict = {}
release_prepare = threading.Event()
prepare_called = threading.Event()


class RejectModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "transformers", "safetensors", "streamlit", "predict"
        }:
            raise AssertionError("Shell imported a model/UI dependency: " + fullname)


sys.meta_path.insert(0, RejectModelImports())


def reject_network(event, args):
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo"}:
        report.setdefault("network_attempts", []).append(event)
        raise AssertionError("Desktop shell attempted network access: " + event)


sys.addaudithook(reject_network)

import psutil
import webview
import webview.http as webview_http

create_window = webview.create_window


def observe_window(*args, **kwargs):
    report["html_string"] = isinstance(kwargs.get("html"), str)
    report["url"] = kwargs.get("url")
    window = create_window(*args, **kwargs)
    assert window is not None
    expose = window.expose

    # The save dialog is a blocking OS window. Answer it with a fixed path so
    # the CSV path is driven end to end without a human at the keyboard.
    csv_target = Path(sys.argv[1]).parent / "saved.csv"

    def stub_file_dialog(*dialog_args, **dialog_kwargs):
        report["save_dialog"] = dialog_kwargs.get("save_filename")
        return str(csv_target)

    window.create_file_dialog = stub_file_dialog

    def observe_exposed(*functions):
        def track(function):
            def invoke(*args):
                reply = function(*args)
                report.setdefault("bridge_calls", []).append({
                    "name": function.__name__, "args": args, "reply": reply,
                })
                return reply

            invoke.__name__ = function.__name__
            return invoke

        def prepare():
            # JS has requested preparation, but no renderer has started yet.
            report["initial"] = window.evaluate_js("""({
              status: document.getElementById('model-status').textContent,
              disabled: document.getElementById('analyze').disabled
            })""")
            report["shown_before_prepare"] = window.events.shown.is_set()
            prepare_called.set()
            if not release_prepare.wait(20):
                raise TimeoutError("Test did not release preparation")
            return next(f for f in functions if f.__name__ == "prepare")()

        expose(*(prepare if f.__name__ == "prepare" else track(f) for f in functions))

    window.expose = observe_exposed

    def wait_for(expression):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            value = window.evaluate_js(expression)
            if value:
                return value
            time.sleep(0.05)
        raise TimeoutError(expression)

    def drive():
        try:
            report["loaded"] = window.evaluate_js("""({
              state: document.readyState, shell: typeof window.xaiDesktop,
              api: typeof window.pywebview.api.prepare,
              status: document.getElementById('model-status')?.textContent,
              script: document.scripts.length
            })""")
            report["shown_at_load"] = window.events.shown.is_set()
            assert prepare_called.wait(20), "Shell did not request preparation"
            # Re-launch from another working directory. A losing process must
            # exit before importing WebView or starting a model worker.
            duplicate = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", """
import importlib.abc, runpy, sys
sys.path.insert(0, sys.argv[1])
class RejectUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'webview' or fullname == 'desktop.fake':
            raise AssertionError('Duplicate launched UI: ' + fullname)
sys.meta_path.insert(0, RejectUI())
sys.argv = ['desktop']
runpy.run_module('desktop', run_name='__main__')
""", str(ROOT)],
                cwd=Path(sys.argv[1]).parent, capture_output=True, text=True,
                encoding="utf-8", timeout=15, check=False,
            )
            report["duplicate_exit"] = duplicate.returncode
            report["duplicate_error"] = duplicate.stderr
            release_prepare.set()
            wait_for("!document.getElementById('analyze').disabled")
            report["ready_status"] = window.evaluate_js(
                "document.getElementById('model-status').textContent"
            )
            report["sidebar"] = window.evaluate_js("""({
              labels: Array.from(document.querySelectorAll('aside label')).map(e => e.textContent),
              button: document.getElementById('analyze').textContent,
              graph: ['min','max','step','value'].map(k => document.getElementById('graph-size')[k])
            })""")

            window.evaluate_js("""
              document.getElementById('dna').value = 'INVALID';
              document.getElementById('analyze').click();
            """)
            wait_for("document.querySelector('#notice .error') && !document.getElementById('analyze').disabled")
            report["validation_error"] = window.evaluate_js(
                "document.getElementById('notice').textContent"
            )
            dna = (
                "GATCCAGTCAGCATGCTTTACGATCGTACGATCGAATCGTAGG"
                "CTAGCATCGATCGATGCATCGTAGCTAGCTAGCATCGA"
            )
            window.evaluate_js(
                "document.getElementById('dna').value = " + json.dumps(dna)
                + "; document.getElementById('analyze').click();"
            )
            wait_for("""!document.getElementById('analyze').disabled &&
              document.querySelector('[data-slot="xai"] img')?.naturalWidth > 0""")
            requests_before_display_changes = len(report["bridge_calls"])
            report["views"] = window.evaluate_js("""(() => {
              return Array.from(document.querySelectorAll('[role="tab"]')).map(tab => {
                tab.click();
                const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));
                const visible = panels.filter(p => !p.hidden);
                return {label: tab.textContent, selected: tab.getAttribute('aria-selected'),
                  visible: visible.map(p => p.id), content: visible[0].innerHTML};
              });
            })()""")
            report["resize"] = window.evaluate_js("""(() => {
              const image = document.querySelector('[data-slot="xai"] img');
              const before = image.src;
              const slider = document.getElementById('graph-size');
              slider.value = 40;
              slider.dispatchEvent(new Event('input'));
              return {sameImage: image.src === before, width: image.closest('figure').style.width,
                label: document.getElementById('graph-size-value').textContent};
            })()""")
            time.sleep(0.2)
            report["display_changes_sent_requests"] = (
                len(report["bridge_calls"]) != requests_before_display_changes
            )
            report["xai_controls"] = window.evaluate_js("""({
              siteLabel: document.querySelector('label[for="xai-site"]').textContent,
              headLabel: document.querySelector('label[for="xai-head"]').textContent,
              methods: Array.from(document.querySelectorAll('[name="xai-method"]')).map(e => e.parentElement.textContent),
              settings: ['ig-steps','shap-n','shap-bg'].map(id => {
                const e = document.getElementById(id);
                return ['min','max','step','value'].map(k => e[k]);
              }),
              sites: Array.from(document.getElementById('xai-site').options).map(e => e.value)
            })""")
            # Changing site/head/settings sends a new calculation while leaving
            # the controls usable during the running request.
            window.evaluate_js("""
              const site = document.getElementById('xai-site');
              site.value = 'SpCas9|+|40';
              site.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            wait_for("document.getElementById('task-state').hidden && document.querySelector('[data-slot=\"xai\"] img')?.naturalWidth > 0")
            report["sp_heads"] = window.evaluate_js("Array.from(document.getElementById('xai-head').options).map(e => e.value)")
            window.evaluate_js("""
              const head = document.getElementById('xai-head');
              head.selectedIndex = 1;
              head.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            wait_for("document.getElementById('task-state').hidden && document.querySelector('[data-slot=\"xai\"] figcaption')?.textContent.includes('SpCas9-NG')")
            window.evaluate_js("""
              document.getElementById('ig-steps').value = 40;
              document.getElementById('shap-n').value = 80;
              document.getElementById('shap-bg').value = 6;
              const shap = document.querySelector('[name="xai-method"][value="shap"]');
              shap.checked = true;
              shap.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            report["shap_progress"] = wait_for("""(() => {
              const progress = document.getElementById('task-progress');
              if (!document.getElementById('task-state').hidden && progress.value > 0 && progress.value < 100)
                return {value:progress.value, label:document.getElementById('task-label').textContent};
              return null;
            })()""")
            window.evaluate_js("""
              const ig = document.querySelector('[name="xai-method"][value="ig"]');
              ig.checked = true;
              ig.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            wait_for("document.getElementById('task-state').hidden && document.querySelector('[data-slot=\"xai\"] figcaption')?.textContent.includes('IG')")
            report["replacement_chart"] = window.evaluate_js("document.querySelector('[data-slot=\"xai\"] figcaption').textContent")
            # Explicit cancellation must also leave the form ready to retry.
            window.evaluate_js("""
              const shap = document.querySelector('[name="xai-method"][value="shap"]');
              shap.checked = true;
              shap.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            wait_for("!document.getElementById('task-state').hidden && document.getElementById('task-progress').value > 0")
            window.evaluate_js("document.getElementById('cancel-task').click()")
            report["cancelled_ui"] = window.evaluate_js("""({
              hidden: document.getElementById('task-state').hidden,
              notice: document.getElementById('notice').textContent,
              canAnalyze: !document.getElementById('analyze').disabled,
              canRetry: !document.querySelector('#xai-controls button[type=submit]').disabled
            })""")
            # ── 전체 상세 보기: 필터 · 검색 · CSV 저장 (#185) ──
            window.evaluate_js("document.getElementById('tab-detail').click()")
            report["detail_before_search"] = window.evaluate_js("""({
              hidden: document.getElementById('detail-results').hidden,
              csvDisabled: document.getElementById('detail-csv').disabled,
              variants: Array.from(document.querySelectorAll('[name="detail-cas"]')).map(b => b.value),
              total: Number(document.getElementById('detail-results').dataset.total)
            })""")
            report["detail_unfiltered"] = window.evaluate_js("""(() => {
              document.getElementById('detail-search').click();
              const rows = Array.from(document.querySelectorAll('#detail-results tbody tr'));
              return {count: document.getElementById('detail-count').textContent,
                shown: rows.filter(r => !r.hidden).length,
                ranks: rows.filter(r => !r.hidden).map(r => r.cells[0].textContent),
                names: rows.filter(r => !r.hidden).map(r => r.cells[1].textContent),
                keys: rows.filter(r => !r.hidden).map(r => r.dataset.siteKey),
                selected: rows.filter(r => r.classList.contains('selected'))
                  .map(r => r.dataset.siteKey),
                highlightCount: document.getElementById('detail-selected').textContent,
                csvDisabled: document.getElementById('detail-csv').disabled};
            })()""")
            # Deselect every variant but one and narrow both ranges.
            report["detail_filtered"] = window.evaluate_js("""(() => {
              document.getElementById('detail-none').click();
              const boxes = Array.from(document.querySelectorAll('[name="detail-cas"]'));
              const keep = boxes.find(b => b.value === 'Cas12a') || boxes[0];
              keep.checked = true;
              const distHigh = document.getElementById('detail-dist-high');
              distHigh.value = 10;
              distHigh.dispatchEvent(new Event('input', {bubbles: true}));
              document.getElementById('detail-search').click();
              const rows = Array.from(document.querySelectorAll('#detail-results tbody tr'));
              const shown = rows.filter(r => !r.hidden);
              return {variant: keep.value, count: document.getElementById('detail-count').textContent,
                ranks: shown.map(r => r.cells[0].textContent),
                names: shown.map(r => r.cells[1].textContent),
                distances: shown.map(r => Number(r.dataset.dist)),
                distCaption: document.getElementById('detail-dist-caption').textContent,
                effCaption: document.getElementById('detail-eff-caption').textContent,
                // The upper handle now bounds the lower one, as a slider would.
                lowMax: document.getElementById('detail-dist-low').max};
            })()""")
            window.evaluate_js("document.getElementById('detail-csv').click()")
            # "저장하는 중..." also contains 저장; wait for the saved-path notice.
            wait_for("""document.getElementById('task-state').hidden &&
              document.getElementById('notice').textContent.includes('저장했습니다')""")
            report["csv_notice"] = window.evaluate_js(
                "document.getElementById('notice').textContent"
            )
            # Hex, not text: reading as text would translate the CRLF the
            # Streamlit download writes, and that is part of what gate 2 diffs.
            report["csv_hex"] = csv_target.read_bytes().hex()
            # An empty result offers no download, exactly as Streamlit does.
            report["detail_empty"] = window.evaluate_js("""(() => {
              document.getElementById('detail-none').click();
              document.getElementById('detail-search').click();
              return {csvDisabled: document.getElementById('detail-csv').disabled,
                noMatch: !document.getElementById('detail-no-match').hidden,
                // Streamlit's 노란 줄 caption lives inside `if filtered:`.
                highlightHidden: document.getElementById('detail-highlight').hidden,
                tableHidden: document.querySelector('#detail-results table').hidden,
                count: document.getElementById('detail-count').textContent};
            })()""")
            window.evaluate_js("""
              document.getElementById('detail-all').click();
              document.getElementById('detail-dist-high').value = 15;
              document.getElementById('detail-search').click();
            """)
            # Check the accessible tab keyboard interaction through DOM events.
            report["keyboard_view"] = window.evaluate_js("""(() => {
              document.getElementById('tab-xai').dispatchEvent(new KeyboardEvent('keydown', {key:'Home'}));
              return document.querySelector('[role="tab"][aria-selected="true"]').textContent;
            })()""")
            process = psutil.Process()
            report["listeners"] = [
                conn.laddr.port
                for proc in [process, *process.children(recursive=True)]
                for conn in proc.net_connections(kind="tcp")
                if conn.status == psutil.CONN_LISTEN
            ]
            report["server_created"] = webview_http.global_server is not None
        except Exception:  # noqa: BLE001 -- persist failures before closing the WebView
            report["error"] = traceback.format_exc()
        finally:
            release_prepare.set()
            Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            window.destroy()

    # pywebview invokes handlers for one event in sequence. Drive on its own
    # thread so the application's loaded handler can start preparation.
    def start_driver():
        threading.Thread(target=drive, daemon=True).start()

    window.events.loaded += start_driver
    return window


webview.create_window = observe_window

from desktop.app import main

raise SystemExit(main(preview=True))
