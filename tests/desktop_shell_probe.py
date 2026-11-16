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

    def observe_exposed(*functions):
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

        expose(*(prepare if f.__name__ == "prepare" else f for f in functions))

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
runpy.run_module('desktop', run_name='__main__')
""", str(ROOT)],
                cwd=Path(sys.argv[1]).parent, capture_output=True, text=True,
                encoding="utf-8", timeout=15,
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
        except Exception:
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

raise SystemExit(main())
