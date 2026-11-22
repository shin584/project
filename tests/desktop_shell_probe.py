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
    hold_save, save_opened, release_save, save_returned = (threading.Event() for _ in range(4))

    def stub_file_dialog(*dialog_args, **dialog_kwargs):
        report["save_dialog"] = dialog_kwargs.get("save_filename")
        report["save_dialogs"] = report.get("save_dialogs", 0) + 1
        if hold_save.is_set():
            save_opened.set()
            assert release_save.wait(30), "Save dialog was not released"
        return None if report.get("dialog_closes") else str(csv_target)

    window.create_file_dialog = stub_file_dialog

    def observe_exposed(*functions):
        def track(function):
            def invoke(*args):
                reply = function(*args)
                report.setdefault("bridge_calls", []).append({
                    "name": function.__name__, "args": args, "reply": reply,
                })
                if function.__name__ == "save_csv":
                    save_returned.set()
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
              methods: Array.from(document.querySelectorAll('[name="xai-method"]'))
                .filter(e => !e.parentElement.hidden).map(e => e.parentElement.textContent),
              retryHidden: document.getElementById('retry-explain').hidden,
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
              canRetry: !document.getElementById('retry-explain').hidden
            })""")
            window.evaluate_js("document.getElementById('retry-explain').click()")
            wait_for("document.getElementById('retry-explain').hidden && document.querySelector('[data-slot=xai] img')?.naturalWidth > 0")
            window.evaluate_js("""
              const head = document.getElementById('xai-head');
              head.dataset.previous = head.value;
              head.add(new Option('invalid head', 'invalid'));
              head.value = 'invalid';
              head.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            report["retry_visible_on_explain_error"] = wait_for("""!document.getElementById('retry-explain').hidden &&
              document.querySelector('#notice .error') !== null""")
            window.evaluate_js("""
              const head = document.getElementById('xai-head');
              head.lastElementChild.remove(); head.value = head.dataset.previous;
              document.getElementById('retry-explain').click();
            """)
            wait_for("document.getElementById('retry-explain').hidden && document.querySelector('[data-slot=xai] img')?.naturalWidth > 0")
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
            # Deselect every variant but one and adjust both thresholds.
            report["detail_filtered"] = window.evaluate_js("""(() => {
              document.getElementById('detail-none').click();
              const boxes = Array.from(document.querySelectorAll('[name="detail-cas"]'));
              const keep = boxes.find(b => b.value === 'Cas12a') || boxes[0];
              keep.checked = true;
              const distHigh = document.getElementById('detail-dist-high');
              distHigh.value = 10;
              distHigh.dispatchEvent(new Event('input', {bubbles: true}));
              const effLow = document.getElementById('detail-eff-low');
              effLow.value = 30;
              effLow.dispatchEvent(new Event('input', {bubbles: true}));
              document.getElementById('detail-search').click();
              const rows = Array.from(document.querySelectorAll('#detail-results tbody tr'));
              const shown = rows.filter(r => !r.hidden);
              return {variant: keep.value, count: document.getElementById('detail-count').textContent,
                ranks: shown.map(r => r.cells[0].textContent),
                names: shown.map(r => r.cells[1].textContent),
                distances: shown.map(r => Number(r.dataset.dist)),
                scores: shown.map(r => Number(r.dataset.eff)),
                distCaption: document.getElementById('detail-dist-caption').textContent,
                effCaption: document.getElementById('detail-eff-caption').textContent};
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
              document.getElementById('detail-eff-low').value = 0;
              document.getElementById('detail-search').click();
            """)
            # Check the accessible tab keyboard interaction through DOM events.
            report["keyboard_view"] = window.evaluate_js("""(() => {
              document.getElementById('tab-xai').dispatchEvent(new KeyboardEvent('keydown', {key:'Home'}));
              return document.querySelector('[role="tab"][aria-selected="true"]').textContent;
            })()""")
            report["stale_generation_ignored"] = window.evaluate_js("""(() => {
              const before = document.getElementById('model-status').innerHTML +
                document.getElementById('notice').innerHTML + document.querySelector('[data-slot=xai]').innerHTML;
              for (const type of ['ready', 'progress', 'result', 'error']) {
                window.xaiDesktop.onEvent({type, generation: 0, request_id: -1,
                  done: 1, total: 2, html: 'STALE', fragments: {xai: 'STALE'}});
              }
              return before === document.getElementById('model-status').innerHTML +
                document.getElementById('notice').innerHTML + document.querySelector('[data-slot=xai]').innerHTML;
            })()""")
            window.evaluate_js("""
              window.xaiDesktop.onEvent({type: 'error', generation: 1, request_id: -1,
                html: '<p class="error">temporary preparation failure</p>'});
            """)
            report["retry_visible_on_error"] = window.evaluate_js("!document.getElementById('retry-prepare').hidden")
            window.evaluate_js("document.getElementById('retry-prepare').click()")
            wait_for("!document.getElementById('analyze').disabled && document.getElementById('retry-prepare').hidden")
            report["retry_ready"] = window.evaluate_js("document.getElementById('model-status').textContent")
            report["default_family"] = window.evaluate_js("document.getElementById('model-family').value")
            previous = report["bridge_calls"][-1]["reply"]
            window.evaluate_js("""
              window.switchStatuses = [];
              const original = window.xaiDesktop.onEvent;
              window.xaiDesktop.onEvent = function(event) {
                const status = document.getElementById('model-status');
                const before = status.textContent;
                original.call(this, event);
                if (before !== status.textContent) window.switchStatuses.push(status.textContent);
              };
            """)
            report["switch_cleared"] = window.evaluate_js("""(() => {
              const family = document.getElementById('model-family');
              family.value = 'nt'; family.dispatchEvent(new Event('change'));
              return document.getElementById('analyze').disabled &&
                document.getElementById('model-status').textContent.endsWith('0%') &&
                document.getElementById('xai-controls').hidden &&
                ['sites', 'best', 'variants', 'detail', 'xai'].every(name =>
                  document.querySelector(`[data-slot="${name}"]`).textContent === '');
            })()""")
            # Deliver all kinds of old events while the new selection is still
            # waiting for its bridge reply (including the old ready fast path).
            window.evaluate_js("""for (const type of ['ready', 'progress', 'result', 'error']) {
              window.xaiDesktop.onEvent({type, generation: 1, request_id: """
                + str(previous["request_id"]) + """, done: 1, total: 2, html: 'STALE',
                fragments: {best: 'STALE', xai: 'STALE'}});
            }""")
            window.evaluate_js("""
              const family = document.getElementById('model-family');
              for (const value of ['non_nt', 'nt']) {
                family.value = value; family.dispatchEvent(new Event('change'));
              }
            """)
            wait_for("""!document.getElementById('analyze').disabled &&
              document.getElementById('task-state').hidden &&
              document.querySelector('[data-slot="xai"] img')?.naturalWidth > 0""")
            report["switch_input_preserved"] = window.evaluate_js("document.getElementById('dna').value") == dna
            report["switch_no_stale"] = window.evaluate_js("!document.body.textContent.includes('STALE')")
            report["switch_statuses"] = window.evaluate_js("window.switchStatuses")
            report["nt_rollout_visible"] = window.evaluate_js("!document.getElementById('rollout-option').hidden")
            before_repeat = len(report["bridge_calls"])
            window.evaluate_js("document.getElementById('model-family').dispatchEvent(new Event('change'))")
            time.sleep(.2)
            report["same_family_no_request"] = len(report["bridge_calls"]) == before_repeat
            saved_mtime = csv_target.stat().st_mtime_ns
            hold_save.set()
            save_returned.clear()
            window.evaluate_js("document.getElementById('detail-search').click(); document.getElementById('detail-csv').click()")
            assert save_opened.wait(10), "CSV dialog did not open"
            window.evaluate_js("""
              const family = document.getElementById('model-family');
              family.value = 'non_nt'; family.dispatchEvent(new Event('change'));
            """)
            wait_for("""!document.getElementById('analyze').disabled &&
              document.getElementById('task-state').hidden &&
              document.querySelector('[data-slot="xai"] img')?.naturalWidth > 0""")
            report["switch_during_save_dialog"] = True
            report["rollout_hidden_again"] = window.evaluate_js("document.getElementById('rollout-option').hidden")
            release_save.set()
            assert save_returned.wait(10)
            report["obsolete_dialog_did_not_write"] = csv_target.stat().st_mtime_ns == saved_mtime
            assert [call for call in report["bridge_calls"] if call["name"] == "save_csv"][-1]["reply"]["request_id"] is None
            # #217: closing the dialog ends quietly; a save into a replaced
            # computation process is refused visibly and refreshes the generation.
            hold_save.clear()
            report["dialog_closes"] = True
            save_returned.clear()
            window.evaluate_js("document.getElementById('detail-search').click(); document.getElementById('detail-csv').click()")
            assert save_returned.wait(10)
            wait_for("document.getElementById('task-state').hidden")
            report["dialog_closed_ui"] = window.evaluate_js("""({
              notice: document.getElementById('notice').textContent,
              canAnalyze: !document.getElementById('analyze').disabled})""")
            report["dialog_closed_error"] = window.evaluate_js("document.querySelector('#notice .error') !== null")
            report["dialog_closes"] = False
            computation = [child for child in psutil.Process().children(recursive=True)
                           if "multiprocessing" in " ".join(child.cmdline())]
            assert len(computation) == 1, computation
            computation[0].kill()
            computation[0].wait(10)
            saved_mtime = csv_target.stat().st_mtime_ns
            dialogs_before = report["save_dialogs"]
            save_returned.clear()
            window.evaluate_js("document.getElementById('detail-csv').click()")
            assert save_returned.wait(10)
            wait_for("document.querySelector('#notice .error')")
            refusal = [call for call in report["bridge_calls"] if call["name"] == "save_csv"][-1]["reply"]
            report["restarted_refusal"] = refusal
            report["restarted_ui"] = window.evaluate_js("""({
              notice: document.getElementById('notice').textContent,
              taskHidden: document.getElementById('task-state').hidden,
              canAnalyze: !document.getElementById('analyze').disabled})""")
            report["restarted_did_not_write"] = csv_target.stat().st_mtime_ns == saved_mtime
            report["restarted_dialog_opened"] = report["save_dialogs"] != dialogs_before
            window.evaluate_js("document.getElementById('analyze').click()")
            wait_for("""!document.getElementById('analyze').disabled &&
              document.getElementById('task-state').hidden &&
              document.querySelector('[data-slot="xai"] img')?.naturalWidth > 0""")
            window.evaluate_js("document.getElementById('detail-search').click(); document.getElementById('detail-csv').click()")
            wait_for("""document.getElementById('task-state').hidden &&
              document.getElementById('notice').textContent.includes('저장했습니다')""")
            report["saved_after_reanalysis"] = csv_target.stat().st_mtime_ns != saved_mtime
            report["save_generation_after_reanalysis"] = [
                call for call in report["bridge_calls"] if call["name"] == "save_csv"][-1]["args"][2]
            # #218: a child that stops answering is replaced in the same family
            # after the cancel grace, both after a bare cancel and when a new
            # analysis was requested behind the cancel.
            def computation_child():
                found = [child for child in psutil.Process().children(recursive=True)
                         if "multiprocessing" in " ".join(child.cmdline())]
                assert len(found) == 1, found
                return found[0]

            def freeze_long_shap():
                window.evaluate_js("""
                  document.getElementById('shap-n').value = '1024';
                  document.getElementById('shap-bg').value = '32';
                  document.querySelector('[name="xai-method"][value="shap"]').checked = true;
                  document.getElementById('xai-controls').dispatchEvent(new Event('change'));
                """)
                wait_for("document.getElementById('task-progress').value > 0")
                child = computation_child()
                child.suspend()
                return child

            def restart_ui():
                return window.evaluate_js("""({
                  status: document.getElementById('model-status').textContent,
                  notice: document.getElementById('notice').textContent,
                  disabled: document.getElementById('analyze').disabled})""")

            frozen = freeze_long_shap()
            window.evaluate_js("document.getElementById('cancel-task').click()")
            wait_for("document.getElementById('model-status').textContent.includes('다시 시작하는')")
            report["restart_after_cancel_ui"] = restart_ui()
            saves_before = sum(call["name"] == "save_csv" for call in report["bridge_calls"])
            window.evaluate_js("document.getElementById('detail-csv').click()")
            wait_for("!document.getElementById('analyze').disabled")
            report["restart_after_cancel_ready"] = restart_ui()
            report["restart_ignored_csv"] = saves_before == sum(
                call["name"] == "save_csv" for call in report["bridge_calls"])
            report["restart_after_cancel_old_child_gone"] = not frozen.is_running()
            report["restart_after_cancel_family"] = window.evaluate_js(
                "document.getElementById('model-family').value")
            frozen = freeze_long_shap()
            # The re-run analysis explains again; keep that explanation short.
            window.evaluate_js("""document.getElementById('cancel-task').click();
              document.getElementById('shap-n').value = '16';
              document.getElementById('shap-bg').value = '2';
              document.getElementById('analyze').click()""")
            wait_for("document.getElementById('model-status').textContent.includes('다시 시작하는')")
            report["restart_after_new_request_locked"] = restart_ui()["disabled"]
            wait_for("""!document.getElementById('analyze').disabled &&
              document.getElementById('task-state').hidden &&
              document.querySelector('[data-slot="xai"] img')?.naturalWidth > 0""")
            report["restart_after_new_request_completed"] = window.evaluate_js(
                "document.querySelector('#detail-results tbody tr') !== null")
            report["restart_after_new_request_old_child_gone"] = not frozen.is_running()
            report["restart_children"] = len([computation_child()])
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
            release_save.set()
            Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            window.destroy()

    # pywebview invokes handlers for one event in sequence. Drive on its own
    # thread so the application's loaded handler can start preparation.
    def start_driver():
        threading.Thread(target=drive, daemon=True).start()

    window.events.loaded += start_driver
    return window


if __name__ == "__main__":
    webview.create_window = observe_window
    from desktop import computation
    from desktop.app import main

    computation.CANCEL_GRACE = 1.0  # #218: keep the probe's forced restarts short

    raise SystemExit(main(preview=True))
