"""Re-check the #215 exhibition-audit scenarios on source and the frozen EXE (#221).

Each `run` launches one real window (source checkout or packaged EXE) with real
models, offline, and drives it only through its WebView2 UI, the native save
dialog and the computation process's OS state (suspend/kill). Scenarios, in order:

1. load limit: the computation process is frozen during its first model load;
   past LOAD_LIMIT the screen shows the error and "모델 준비 다시 시도", whose
   click recovers in the same family.
2. CSV dialog closed: closing the native save dialog ends silently and clears
   "CSV를 저장하는 중...".
3. CSV after an idle exit: the computation process is killed while idle; the
   next save is refused with a visible error, "CSV를 저장하는 중..." does not
   remain, and a save after reanalysis succeeds.
4. request records: 100,000 requests through the JS bridge; the UI process
   memory is sampled before and after.
5. stall without cancel: a frozen process during analysis is force-restarted
   after NO_PROGRESS_LIMIT with an error, and the analysis is not re-run.
6. stalled cancel and new analysis: cancel then analyze again on a frozen
   process; after CANCEL_GRACE it is force-restarted in the same family and the
   new analysis completes.
7. native close after the forced restarts: exit code 0 and no child left.

`history` runs 100,000 requests through `Dispatcher` from the source tree or,
with `--exe`, from the `desktop.api` bytecode frozen inside the EXE, and checks
that the request records stay at OUTCOME_LIMIT or below.
"""

import argparse
import csv
import ctypes
import json
import os
import platform
import site
import socket
import subprocess
import sys
import time
import traceback
import tracemalloc
import types
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import psutil
from measure_ram import ROOT, digest, sample_tree, save

TITLE = "CRISPR PAM Scanner + XAI"
EXE_NAME = "XaiDashboardV3.exe"
ROWS = "#detail-results tbody tr"
ENVIRONMENT = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "KERAS_BACKEND": "torch",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
LIMITS = {"cancel_grace": 53.0, "no_progress_limit": 105.0, "load_limit": 165.0}
# Wrap the bridge and the event sink so the run can tell which requests the
# screen itself made (an automatic re-run would show up as a new call).
OBSERVE = """() => {
  const trace = window.recheckTrace = [];
  const api = window.pywebview.api;
  window.recheckRaw = {};
  for (const name of ['prepare', 'select_family', 'scan_sites', 'predict', 'explain', 'save_context', 'save_csv', 'cancel']) {
    const send = api[name];
    window.recheckRaw[name] = send;
    api[name] = (...args) => { trace.push({call: name, time: Date.now() / 1000}); return send(...args); };
  }
  const receive = window.xaiDesktop.onEvent;
  window.xaiDesktop.onEvent = event => {
    if (event.type !== 'progress') {
      trace.push({event: event.type, reason: event.reason, generation: event.generation,
                  family: event.family, time: Date.now() / 1000});
    }
    return receive(event);
  };
}"""
BURST = """async ([total, batch, dna]) => {
  const predict = window.recheckRaw.predict;
  let replies = 0;
  for (let sent = 0; sent < total; sent += batch) {
    const calls = [];
    for (let i = 0; i < Math.min(batch, total - sent); i++) calls.push(predict(dna));
    replies += (await Promise.all(calls)).filter(r => r && r.request_id).length;
  }
  return replies;
}"""


def _title_windows(pids):
    """(hwnd, owner pid) of the app's top-level windows owned by `pids`."""
    user32 = ctypes.windll.user32
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    found = []

    @callback
    def visit(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value in pids:
            title = ctypes.create_unicode_buffer(160)
            user32.GetWindowTextW(hwnd, title, len(title))
            if title.value == TITLE:
                found.append((hwnd, owner.value))
        return True

    user32.EnumWindows(visit, 0)
    return found


def window_owner(process, timeout=60):
    """The process that owns the app window. A venv `python.exe` is a redirector
    that starts the real interpreter as its child, so this is not always `process`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Desktop exited before showing its window: {process.returncode}")
        try:
            tree = {process.pid, *(c.pid for c in psutil.Process(process.pid).children(recursive=True))}
        except psutil.NoSuchProcess:
            tree = {process.pid}
        owners = {pid for _, pid in _title_windows(tree)}
        if len(owners) == 1:
            return owners.pop()
        time.sleep(.2)
    raise TimeoutError("The app window did not appear")


def freeze_first_worker(process, timeout=60):
    """Suspend the first computation process as soon as it exists, before its
    model load can finish; return it with the time it was frozen."""
    from measure_package import worker
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            child = worker(process)
        except (RuntimeError, psutil.Error):
            time.sleep(.01)
            continue
        child.suspend()
        return child, time.monotonic()
    raise TimeoutError("The computation process did not start")


def close_window(pid):
    """Post WM_CLOSE to the app's top-level window, as the title-bar X does."""
    windows = _title_windows({pid})
    for hwnd, _ in windows:
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)
    return len(windows)


def identity(mode, exe, source_python):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=ROOT, text=True).strip()
    found = {"checkout_commit": commit, "checkout_tracked_changes": bool(dirty),
             "host": platform.node(), "platform": platform.platform(),
             "ram_bytes": psutil.virtual_memory().total, "logical_cpus": psutil.cpu_count()}
    if mode == "exe":
        manifest = exe.parent / "release-manifest.json"
        release = json.loads(manifest.read_text(encoding="utf-8"))
        found.update(executable=str(exe), executable_sha256=digest(exe),
                     manifest_sha256=digest(manifest), source_commit=release["source_commit"],
                     built_at=release["built_at"], file_count=release["file_count"],
                     total_bytes=release["total_bytes"])
    else:
        found.update(source_python=str(source_python), source_commit=commit)
    return found


def drive(page, process, ui_pid, desktop, output, report, frozen):
    from measure_package import IDLE, worker
    fixture = json.loads((ROOT / "docs/equivalence-baseline/capture.json").read_text(encoding="utf-8"))
    dna = fixture["input_seq"]
    started = time.monotonic()

    def record(name, passed, **data):
        report["checks"].append({"check": name, "passed": bool(passed),
                                 "elapsed_s": round(time.monotonic() - started, 2), **data})
        print(f"{'PASS' if passed else 'FAIL'} {name} {json.dumps(data, ensure_ascii=False)}", flush=True)

    def idle(timeout=600):
        page.wait_for_function(IDLE, timeout=timeout * 1000)

    def notice():
        return page.locator("#notice").inner_text()

    def errors():
        return [e.inner_text() for e in page.locator("#notice .error").all()]

    def trace():
        return page.evaluate("window.recheckTrace")

    def analyze(expect_rows=38):
        page.locator("#tab-best").click()
        page.locator("#dna").fill(dna)
        page.locator("#analyze").click()
        idle()
        page.locator("#tab-detail").click()
        return page.locator(ROWS).count() == expect_rows

    def save_dialog(timeout):
        dialog = desktop.window(class_name="#32770", title_re=".*(Save|저장).*")
        try:
            dialog.wait("ready", timeout=timeout)
        except Exception:  # noqa: BLE001 -- pywinauto raises its own timeout types
            return None
        return dialog

    def dismiss(dialog):
        # The dialog's Cancel button (IDCANCEL), as a visitor closing it would.
        dialog.child_window(control_id=2, class_name="Button").click()
        dialog.wait_not("visible", timeout=10)

    def ui_private():
        return psutil.Process(ui_pid).memory_info().private

    def ready_bridge():
        page.wait_for_function("window.xaiDesktop && typeof window.pywebview?.api?.prepare === 'function'",
                               timeout=60000)
        page.evaluate(OBSERVE)

    if frozen is None:
        # Probe development only: the run then cannot pass.
        ready_bridge()
        idle()
        record("load_limit_scenario_run", False)
    else:
        # 1. Load limit: the first computation process was frozen at launch.
        child, frozen_at = frozen
        ready_bridge()
        record("frozen_before_ready", "준비" in page.locator("#model-status").inner_text()
               and page.locator("#analyze").is_disabled(),
               status=page.locator("#model-status").inner_text())
        page.wait_for_selector("#notice .error", timeout=(LIMITS["load_limit"] + 60) * 1000)
        seconds = time.monotonic() - frozen_at
        record("load_limit_error_and_retry", page.locator("#retry-prepare").is_visible()
               and any("적재 한도" in e for e in errors()) and page.locator("#analyze").is_disabled()
               and not child.is_running(), seconds_after_freeze=round(seconds, 1), errors=errors())
        page.locator("#retry-prepare").click()
        idle()
        replacement = worker(process)
        record("load_limit_retry_recovers", replacement.pid != child.pid
               and page.locator("#model-family").input_value() == "non_nt", worker_pid=replacement.pid)

    record("non_nt_prediction", analyze(), rows=page.locator(ROWS).count())

    # 2. Closing the save dialog ends silently.
    target = output / "export.csv"
    page.locator("#detail-search").click()
    page.locator("#detail-csv").click()
    dialog = save_dialog(20)
    if dialog is not None:
        dismiss(dialog)
    idle(30)
    # The task ends with no error and no file, and the saving notice is cleared.
    record("csv_dialog_closed_is_silent", dialog is not None and not errors() and not target.exists()
           and "저장하는 중" not in notice(), notice=notice())

    # 3. The computation process exits while idle, then the user saves.
    old = worker(process)
    old.kill()
    old.wait(10)
    page.locator("#detail-csv").click()
    dialog = save_dialog(5)
    if dialog is not None:
        dismiss(dialog)
    idle(120)
    record("csv_after_idle_exit_refused_visibly", dialog is None and not target.exists()
           and any("저장하지 못했습니다" in e for e in errors()) and "저장하는 중" not in notice(),
           notice=notice(), dialog_opened=dialog is not None)
    record("csv_reanalysis", analyze(), rows=page.locator(ROWS).count())
    page.locator("#detail-search").click()
    expected = page.locator(ROWS + ":not([hidden])").count()
    page.locator("#detail-csv").click()
    dialog = save_dialog(20)
    if dialog is not None:
        dialog.child_window(class_name="Edit").set_edit_text(str(target))
        dialog.child_window(control_id=1).click()
    page.wait_for_function("document.querySelector('#notice').textContent.includes('저장했습니다:')",
                           timeout=120000)
    with target.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    record("csv_saved_after_reanalysis", len(rows) == expected + 1 and rows[0][-2:] == ["모델", "모델 버전"],
           rows=len(rows) - 1)
    target.unlink()

    # 4. 100,000 requests through the bridge, twice. The records are not
    # readable from outside, so the UI process memory is the observable: a
    # second burst that does not grow it again means nothing accumulates per
    # request (the first may raise the allocator's high-water mark).
    bursts = []
    for _ in range(2):
        before = ui_private()
        burst_started = time.monotonic()
        replies = page.evaluate(BURST, [100_000, 1000, "A" * 81])
        burst_seconds = time.monotonic() - burst_started
        time.sleep(10)
        after = ui_private()
        bursts.append({"replies": replies, "seconds": round(burst_seconds, 1),
                       "ui_private_before_mib": round(before / 2**20, 1),
                       "ui_private_after_mib": round(after / 2**20, 1),
                       "ui_private_delta_mib": round((after - before) / 2**20, 1)})
    record("request_burst_100k", all(b["replies"] == 100_000 for b in bursts) and analyze(),
           bursts=bursts)

    # 5. Stall without a cancel: no-progress limit, error, no re-run.
    old = worker(process)
    old.suspend()
    page.locator("#tab-best").click()
    page.locator("#analyze").click()
    frozen_at = time.monotonic()
    marker = len(trace())
    page.wait_for_function("[...document.querySelectorAll('#notice .error')].some(e => e.textContent.includes('응답하지 않아'))",
                           timeout=(LIMITS["no_progress_limit"] + 60) * 1000)
    seconds = time.monotonic() - frozen_at
    status = page.locator("#model-status").inner_text()
    locked = page.locator("#analyze").is_disabled()
    idle()
    time.sleep(5)
    later = trace()[marker:]
    reruns = [e["call"] for e in later if e.get("call") in ("scan_sites", "predict", "explain")]
    new = worker(process)
    record("stall_no_progress_restart", any(e.get("event") == "restarting" and e.get("reason") == "no_progress"
                                            for e in later)
           and not old.is_running() and new.pid != old.pid and not reruns
           and page.locator("#model-family").input_value() == "non_nt"
           and any("응답하지 않아" in e for e in errors()),
           seconds_after_freeze=round(seconds, 1), status_at_error=status, locked_at_error=locked,
           reruns=reruns, worker_pid=new.pid)

    # 6. Stalled cancel, then a new analysis. The cancel button shows once a
    # computation is under way (an explanation at once, an analysis only after
    # the PAM scan answers), so the frozen computation is an explanation.
    record("prediction_before_cancel_scenario", analyze(), rows=page.locator(ROWS).count())
    old = worker(process)
    old.suspend()
    page.locator("#tab-xai").click()
    case = fixture["xai"][0]
    page.evaluate("""([key, name]) => {
        const site = document.querySelector('#xai-site');
        site.value = key;
        const heads = JSON.parse(site.selectedOptions[0]?.dataset.heads || '[]');
        if (!heads.includes(name)) throw Error('Fixture XAI site/head unavailable');
        const head = document.querySelector('#xai-head');
        head.replaceChildren(...heads.map(x => new Option(x, x)));
        head.value = name;
        document.querySelector('#ig-steps').value = 40;
        const radio = document.querySelector('[name=xai-method][value=ig]');
        radio.checked = true;
        radio.dispatchEvent(new Event('change', {bubbles: true}));
    }""", [case["site_key"], case["head"]])
    page.wait_for_selector("#task-state:not([hidden])", timeout=10000)
    time.sleep(1)
    page.locator("#cancel-task").click()
    cancelled_at = time.monotonic()
    cancel_notice = notice()
    marker = len(trace())
    page.locator("#analyze").click()
    page.wait_for_function("document.querySelector('#model-status').textContent.includes('다시 시작하는 중')",
                           timeout=(LIMITS["cancel_grace"] + 60) * 1000)
    seconds = time.monotonic() - cancelled_at
    status = page.locator("#model-status").inner_text()
    locked = page.locator("#analyze").is_disabled()
    notice_at_restart = notice()
    idle()
    later = trace()[marker:]
    new = worker(process)
    page.locator("#tab-detail").click()
    rows = page.locator(ROWS).count()
    # The new analysis replaces the cancel notice with its own progress text,
    # so the notice at restart is only recorded.
    record("stalled_cancel_then_new_analysis", cancel_notice == "계산을 취소했습니다." and locked
           and any(e.get("event") == "restarting" and e.get("reason") == "cancel_grace" for e in later)
           and not old.is_running() and new.pid != old.pid and rows == 38
           and page.locator("#model-family").input_value() == "non_nt"
           and not any(e.get("call") == "select_family" for e in later),
           cancel_notice=cancel_notice, seconds_after_cancel=round(seconds, 1), status_at_restart=status,
           notice_at_restart=notice_at_restart, rows=rows, worker_pid=new.pid)


def run(output, mode, exe, source_python, automation_site, skip_load=False):
    # Window-only dependencies; `history` also runs in the build venv.
    from measure_package import webview_endpoint
    from playwright.sync_api import sync_playwright

    output.mkdir(parents=True, exist_ok=False)
    if mode == "exe":
        command = [str(exe)]
    else:
        command = [str(source_python), "-X", "utf8", str(ROOT / "desktop_entry.py")]
    cwd = output / "unrelated-cwd"
    cwd.mkdir()
    site.addsitedir(str(automation_site))
    application = import_module("pywinauto").Application
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    browser_args = (f"--remote-debugging-address=127.0.0.1 --remote-debugging-port={port} "
                    '--disable-background-networking --proxy-server=http://127.0.0.1:9 '
                    '--host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE localhost"')
    env = {**os.environ, **ENVIRONMENT, "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS": browser_args,
           "WEBVIEW2_USER_DATA_FOLDER": str(output / "webview-profile"),
           "MPLCONFIGDIR": str(output / "matplotlib")}
    if mode == "exe":
        env.update({name: str(output / "empty-hf") for name in
                    ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")})
        env.pop("PYTHONPATH", None)
    report = {"issue": 221, "mode": mode, "started_at": datetime.now(timezone.utc).isoformat(),
              "identity": identity(mode, exe, source_python), "command": command, "cwd": str(cwd),
              "limits_expected": LIMITS, "checks": [], "status": "fail"}
    with (output / "window.log").open("w", encoding="utf-8") as log, ThreadPoolExecutor(1) as executor:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        sampler = executor.submit(sample_tree, process, output, 1.0, 3600)
        try:
            frozen = None if skip_load else freeze_first_worker(process)
            ui_pid = window_owner(process)
            report["window_owner_pid"] = ui_pid
            desktop = application(backend="win32").connect(process=ui_pid, timeout=30)
            with sync_playwright() as driver:
                browser = driver.chromium.connect_over_cdp(webview_endpoint(port, process, 60))
                context = browser.contexts[0]
                page = context.pages[0] if context.pages else context.wait_for_event("page")
                page.set_default_timeout(30000)
                context.new_cdp_session(page).send("Network.emulateNetworkConditions", {
                    "offline": True, "latency": 0, "downloadThroughput": 0, "uploadThroughput": 0})
                page_errors = []
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                drive(page, process, ui_pid, desktop, output, report, frozen)
                report["page_errors"] = page_errors
                report["trace"] = page.evaluate("window.recheckTrace || []")
                tree = psutil.Process(process.pid).children(recursive=True)
            # 7. Native close after the forced restarts.
            report["close_windows_posted"] = close_window(ui_pid)
            report["app_exit_code"] = process.wait(30)
            _, alive = psutil.wait_procs(tree, timeout=10)
            report["survivors"] = [p.pid for p in alive]
            report["checks"].append({"check": "native_close_after_forced_restarts",
                                     "passed": report["app_exit_code"] == 0 and not alive
                                     and report["close_windows_posted"] == 1,
                                     "exit_code": report["app_exit_code"], "survivors": report["survivors"]})
        except Exception:  # noqa: BLE001 -- keep the evidence of a failed window run
            report["error"] = traceback.format_exc()
            print(report["error"], flush=True)
        finally:
            if process.poll() is None:
                close_window(report.get("window_owner_pid", process.pid))
                try:
                    process.wait(15)
                except subprocess.TimeoutExpired:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        child.kill()
                    process.kill()
            result = sampler.result()
            report["sampler"] = {k: v for k, v in result.items() if k != "process_inventory"}
            failed = [c["check"] for c in report["checks"] if not c["passed"]]
            report["summary"] = {"passed": len(report["checks"]) - len(failed), "failed": len(failed),
                                 "failed_checks": failed}
            if (not failed and "error" not in report and result["exit_code"] == 0
                    and not result["survivors"] and not result["timed_out"]):
                report["status"] = "pass"
            save(output / "report.json", report)
    print(json.dumps({"status": report["status"], **report["summary"]}, ensure_ascii=False))
    return int(report["status"] != "pass")


def frozen_api(exe):
    """Load `desktop.api` from the PYZ archive frozen inside the EXE."""
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(exe))
    name = next(n for n in archive.toc if n.endswith(".pyz"))
    pyz = archive.open_embedded_archive(name)
    code = pyz.extract("desktop.api")
    if isinstance(code, tuple):
        code = code[-1]
    module = types.ModuleType("desktop.api")
    module.__file__ = f"{exe}::desktop.api"
    sys.modules[module.__name__] = module
    exec(code, module.__dict__)
    return module


def history(output, exe):
    if exe is None:
        sys.path.insert(0, str(ROOT))
        api = import_module("desktop.api")
        origin = {"source": "checkout", "commit": identity("source", None, None)["checkout_commit"]}
    else:
        api = frozen_api(exe)
        origin = {"source": "frozen EXE bytecode", **identity("exe", exe, None)}
    dispatcher = api.Dispatcher(lambda event: None)
    tracemalloc.start()
    samples = []
    try:
        for index in range(100):
            for _ in range(1000):
                dispatcher.submit(lambda batch: {})
            assert dispatcher.wait_idle(10)
            if index in {0, 9, 99}:
                samples.append({"requests": (index + 1) * 1000, "records": len(dispatcher._outcomes),
                                "traced_bytes": tracemalloc.get_traced_memory()[0]})
    finally:
        dispatcher.close()
        tracemalloc.stop()
    report = {"issue": 221, "origin": origin, "outcome_limit": api.OUTCOME_LIMIT, "samples": samples,
              "passed": api.OUTCOME_LIMIT == 256 and all(s["records"] <= 256 for s in samples)}
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    print(json.dumps(report, ensure_ascii=False))
    return int(not report["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    window = commands.add_parser("run")
    window.add_argument("output", type=Path)
    window.add_argument("mode", choices=("source", "exe"))
    window.add_argument("--exe", type=Path)
    window.add_argument("--source-python", type=Path, default=ROOT / ".packaging/issue204-inputs/venv/Scripts/python.exe")
    window.add_argument("--skip-load-limit", action="store_true",
                        help="probe development only; the run is reported as failed")
    window.add_argument("--automation-site", type=Path, default=ROOT / ".packaging/issue204-inputs/ui-automation")
    records = commands.add_parser("history")
    records.add_argument("output", type=Path)
    records.add_argument("--exe", type=Path)
    args = parser.parse_args()
    if args.command == "history":
        return history(args.output.resolve(), args.exe.resolve() if args.exe else None)
    if args.mode == "exe" and args.exe is None:
        parser.error("--exe is required for an EXE run")
    return run(args.output.resolve(), args.mode, args.exe.resolve() if args.exe else None,
               args.source_python.resolve(), args.automation_site.resolve(), args.skip_load_limit)


if __name__ == "__main__":
    raise SystemExit(main())
