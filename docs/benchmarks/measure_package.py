"""Measure both real Windows release folders through their WebView2 UI (#204)."""

import argparse
import csv
import json
import math
import os
import platform
import site
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import psutil
from measure_ram import ROOT, digest, sample_tree, save, source_identity, summarize
from playwright.sync_api import sync_playwright

BASELINE = ROOT / "docs/equivalence-baseline"
EXE = {"legacy": "XaiDashboard.exe", "v3": "XaiDashboardV3.exe"}
IDLE = "document.querySelector('#task-state').hidden && !document.querySelector('#analyze').disabled"
IMAGES = "[data-slot=xai] img"
OBSERVE = """() => {
  const trace = window.releaseTrace = [];
  const api = window.pywebview.api;
  for (const operation of ['scan_sites', 'predict', 'explain', 'select_family', 'prepare', 'cancel']) {
    if (!api[operation]) continue;
    const send = api[operation];
    api[operation] = (...args) => {
      const details = operation === 'explain' ? {site: args[1], head: args[2], method: args[3], settings: args[4]} :
                      operation === 'select_family' ? {selected: args[0]} : {};
      trace.push({label: operation, time: Date.now()/1000, ...details});
      return send(...args);
    };
  }
  const receive = window.xaiDesktop.onEvent;
  window.xaiDesktop.onEvent = event => {
    const {type, generation, request_id, family} = event;
    trace.push({label: 'event', type, generation, request_id, family, time: Date.now()/1000});
    return receive(event);
  };
}"""


def identify(folder, mode):
    internal = folder / "_internal"
    manifest = folder / "release-manifest.json"
    release = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "root": str(folder), "executable_sha256": digest(folder / EXE[mode]),
        "manifest_sha256": digest(manifest), "source_commit": release["source_commit"],
        "dependencies": release.get("dependencies", release.get("versions")),
        "webview2_sha256": digest(internal / "webview2/msedgewebview2.exe"),
        "assets": {path.relative_to(internal).as_posix(): digest(path) for path in sorted([
            *internal.glob("models/*.safetensors"), *internal.glob("models/non_nt/**/*.keras"),
            *internal.glob("tokenizer/*"),
        ]) if path.is_file()},
    }


def register(output, legacy, v3):
    capture = json.loads((BASELINE / "capture.json").read_text(encoding="utf-8"))
    releases = {"legacy": legacy}
    if v3 is not None:
        releases["v3"] = v3
    identity = {mode: identify(path, mode) for mode, path in releases.items()}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "contract.json", {
        "issue": 204, "registered_at": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen release comparison on the development PC" if v3 else "frozen legacy repeat",
        "releases": identity, "measurement_modes": list(identity),
        "pc": {"host": platform.node(), "platform": platform.platform(),
               "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total},
        "observer_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("measure_ram.py"))},
        "dna": capture["input_seq"], "settings": capture["settings"],
        "cases": [{"site": case["site_key"], "head": case["head"]} for case in capture["xai"]],
        "environment": {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "KERAS_BACKEND": "torch",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
        "sample_interval_seconds": .1, "settle_seconds": 5, "stable_seconds": 10,
        "operation_timeout_seconds": 900, "run_timeout_seconds": 3600,
        "sequence": ["legacy:startup,predict,automatic-XAI,both-Cas-IG/SHAP/Rollout,close",
                     "v3:default-nonNT,startup,predict,automatic-XAI,both-Cas-IG/SHAP",
                     "v3:switch-NT,automatic-analysis,both-Cas-IG/SHAP/Rollout",
                     "v3:return-nonNT,automatic-analysis,10-second-stable,acceptance checks,close"],
        "metrics": {"rss": "simultaneous process working-set sum (may count shared pages twice)",
                    "private": "simultaneous private commit sum, including paged-out memory",
                    "peak": "maximum simultaneous sum per phase", "stable": "final 2 seconds median/min/max"},
        "offline_scope": "fresh WebView2 profile and frozen HF cache; local dead proxy, DNS overrides and "
                         "CDP offline mode; Windows host adapter remains enabled",
        "legacy_disk_shap": "production policy; packaged cache may be used",
    })


def webview_endpoint(port, process, timeout):
    deadline = time.monotonic() + timeout
    address = f"http://127.0.0.1:{port}"
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Desktop exited before opening WebView2: {process.returncode}")
        try:
            with opener.open(address + "/json/version", timeout=1) as response:
                return json.load(response)["webSocketDebuggerUrl"]
        except (URLError, TimeoutError, KeyError):
            time.sleep(.2)
    raise TimeoutError("WebView2 debugging endpoint did not open")


def request_script(case, method, settings=None):
    return """([siteKey, headName, method, settings]) => {
      const site = document.querySelector('#xai-site');
      site.value = siteKey;
      const heads = JSON.parse(site.selectedOptions[0]?.dataset.heads || '[]');
      if (!heads.includes(headName)) throw Error(`Unavailable head ${headName} at ${siteKey}`);
      const head = document.querySelector('#xai-head');
      head.replaceChildren(...heads.map(name => new Option(name, name)));
      head.value = headName;
      document.querySelector(`[name=xai-method][value=${method}]`).checked = true;
      for (const [id, value] of Object.entries(settings || {})) document.querySelector('#' + id).value = value;
      document.querySelector('#xai-controls').requestSubmit();
    }""", [case["site"], case["head"], method, settings]


def worker(process):
    children = [p for p in psutil.Process(process.pid).children(recursive=True)
                if "--multiprocessing-fork" in p.cmdline()]
    if len(children) != 1:
        raise RuntimeError(f"Expected one compute process, found {[p.pid for p in children]}")
    return children[0]


def csv_export(page, desktop, target):
    page.locator("#tab-detail").click()
    page.locator("#detail-search").click()
    page.locator("#detail-csv").click()
    dialog = desktop.window(class_name="#32770", title_re=".*(Save|저장).*")
    dialog.wait("ready", timeout=20)
    dialog.child_window(class_name="Edit").set_edit_text(str(target))
    dialog.child_window(control_id=1).click()
    page.wait_for_function("document.querySelector('#notice').textContent.includes('저장했습니다:')")
    with target.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


def stress(page, process, case, wait_idle, record):
    # Force a fresh explanation, then supersede it with a different method.
    page.evaluate(*request_script(case, "shap", {"shap-n": 128}))
    page.evaluate(*request_script(case, "ig", {"ig-steps": 40, "shap-n": 64}))
    wait_idle(1)
    record("consecutive_requests_latest_ig", page.locator(f"{IMAGES}[alt*='IG']").count() == 1)

    page.evaluate(*request_script(case, "shap", {"shap-n": 1024, "shap-bg": 32}))
    page.wait_for_function("document.querySelector('#task-progress').hasAttribute('value')")
    cancelled = page.evaluate("""() => window.releaseTrace.findLast(e => e.type === 'progress')""")
    record("cancellable_shap_started", cancelled is not None and "request_id" in cancelled and "generation" in cancelled)
    page.locator("#cancel-task").click()
    page.wait_for_function("""document.querySelector('#notice').textContent.includes('취소했습니다') &&
        window.releaseTrace.some(e => e.label === 'cancel')""")
    # Replay an actual old request identity via the same browser event entry point.
    # A sentinel makes an incorrectly accepted terminal event observable.
    page.evaluate("""old => window.xaiDesktop.onEvent({...old, type: 'result',
      fragments: {xai: '<b id="stale-release-event">stale</b>'}})""", cancelled)
    record("cancel_and_discard_replayed_event", page.locator("#stale-release-event").count() == 0)
    page.evaluate(*request_script(case, "ig", {"ig-steps": 32, "shap-n": 64, "shap-bg": 4}))
    wait_idle(1)
    record("cancelled_shap_sent_no_result", page.evaluate("""old => window.releaseTrace.filter(
       e => e.type === 'result' && e.request_id === old.request_id && e.generation === old.generation
    ).length === 1""", cancelled))  # The one entry is the explicit replay above.

    previous = worker(process)
    page.evaluate(*request_script(case, "shap", {"shap-n": 1008, "shap-bg": 32}))
    page.wait_for_function("document.querySelector('#task-progress').hasAttribute('value')")
    previous.kill()
    previous.wait(timeout=10)
    page.wait_for_function("!document.querySelector('#retry-prepare').hidden")
    record("worker_failure_visible", "retry" in page.locator("#notice").inner_text().lower())
    page.evaluate("""() => {
      document.querySelector('[name=xai-method][value=ig]').checked = true;
      document.querySelector('#shap-n').value = 64;
      document.querySelector('#shap-bg').value = 4;
    }""")
    page.locator("#retry-prepare").click()
    wait_idle(1)
    record("retry_recreated_child", worker(process).pid != previous.pid)


def drive(page, process, desktop, output, mode, contract, report):
    timeout = contract["operation_timeout_seconds"] * 1000
    v3 = mode != "legacy"
    markers = []
    family = "non_nt" if v3 else "nt"

    def mark(label, **detail):
        markers.append({"label": label, "time": time.time(), "family": family, **detail})

    def record(label, passed, **detail):
        report["checks"].append({"check": label, "passed": bool(passed), **detail})
        print(f"{'PASS' if passed else 'FAIL'} {label}", flush=True)
        if not passed:
            raise AssertionError(label)

    def wait_idle(images=None):
        predicate = f"({IDLE}) || document.querySelector('#retry-prepare')?.hidden === false"
        page.wait_for_function(predicate, timeout=timeout)
        if page.locator("#notice .error").count():
            raise RuntimeError(page.locator("#notice").inner_text())
        if images:
            page.wait_for_function("""([selector, count]) => {
                const images = Array.from(document.querySelectorAll(selector));
                return images.length === count && images.every(img => img.complete && img.naturalWidth > 0);
            }""", arg=[IMAGES, images], timeout=20000)

    def capture():
        rows = page.locator("#detail-results tbody tr").evaluate_all(
            "rows => rows.map(r => ({site: r.dataset.siteKey, head: r.dataset.head, "
            "raw: Number(r.dataset.raw), percent: Number(r.dataset.eff), cells: [...r.cells].map(c => c.innerText)}))")
        record(f"{family}:prediction_rows", bool(rows) and all(row["head"] for row in rows), count=len(rows))
        result = {"rows": rows, "xai": {}}
        page.locator("#tab-xai").click()
        for case in contract["cases"]:
            for method in (("ig", "shap") if family == "non_nt" else ("ig", "shap", "rollout")):
                page.evaluate(*request_script(case, method))
                wait_idle(1 if method == "ig" else 2)
                time.sleep(contract["settle_seconds"])
                key = f"{case['site']}:{case['head']}:{method}"
                result["xai"][key] = page.locator("[data-slot=xai] :is(figcaption,.caption,summary,details p)").all_text_contents()
                record(f"{family}:{key}:decoded", page.locator(IMAGES).first.is_visible()
                       and page.locator("#xai-head").input_value() == case["head"]
                       and bool(result["xai"][key]))
                if v3 and method != "rollout":
                    record(f"{family}:{key}:units", "출력 단위 참고" in result["xai"][key])
        report["families"][family] = result

    def switch(selected):
        nonlocal family
        old = worker(process)
        descendants = old.children(recursive=True)
        page.locator("#model-family").select_option(selected)
        family = selected
        wait_idle()
        selected_method = page.locator("[name=xai-method]:checked").input_value()
        wait_idle(1 if selected_method == "ig" else 2)
        record(f"{family}:worker_replaced", not old.is_running() and not any(p.is_running() for p in descendants),
               before=old.pid, after=worker(process).pid)
        record(f"{family}:dna_retained", page.locator("#dna").input_value() == contract["dna"])

    try:
        page.wait_for_function("""typeof window.pywebview?.api?.predict === 'function' &&
            typeof window.pywebview?.api?.explain === 'function' && window.xaiDesktop""", timeout=60000)
        page.evaluate(OBSERVE)
        wait_idle()
        if v3:
            record("default_non_nt", page.locator("#model-family").input_value() == "non_nt")
            record("non_nt_rollout_disabled", page.locator("[name=xai-method][value=rollout]").is_disabled())
        if mode == "v3-relaunch":
            return
        mark("startup_ready")
        time.sleep(contract["settle_seconds"])
        settings = page.evaluate("""() => Object.fromEntries(['ig-steps','shap-n','shap-bg','graph-size']
            .map(id => [id, Number(document.getElementById(id).value)]))""")
        record("default_settings", list(settings.values()) == [32, 64, 4, 80], settings=settings)
        page.locator("#dna").fill(contract["dna"])
        page.locator("#analyze").click()
        wait_idle(1 if v3 else 2)
        time.sleep(contract["settle_seconds"])
        capture()
        if not v3:
            record("prediction_and_explanation_boundaries_captured", page.evaluate("""() =>
              window.releaseTrace.some(e => e.label === 'predict') &&
              window.releaseTrace.filter(e => e.label === 'explain').length >= 7
            """))
            mark("window_close")
            return
        switch("nt")
        time.sleep(contract["settle_seconds"])
        capture()
        if mode == "source":
            mark("window_close")
            return
        switch("non_nt")
        mark("return_stable")
        time.sleep(contract["stable_seconds"])
        mark("acceptance_extras")
        stress(page, process, contract["cases"][0], wait_idle, record)
        before = page.evaluate("window.releaseTrace.filter(e => e.label === 'explain').length")
        page.evaluate("""() => {
          const slider = document.querySelector('#graph-size');
          slider.value = 40;
          slider.dispatchEvent(new Event('input'));
        }""")
        record("resize_existing_graph_without_recompute", page.evaluate("""before =>
          [...document.querySelectorAll('[data-slot=xai] figure')].length > 0 &&
          [...document.querySelectorAll('[data-slot=xai] figure')].every(f => f.style.width === '40%') &&
          window.releaseTrace.filter(e => e.label === 'explain').length === before
        """, before))
        for selected in ("non_nt", "nt"):
            if selected != family:
                switch(selected)
            for tab in ("best", "variants", "detail", "xai"):
                page.locator("#tab-" + tab).click()
                record(f"{family}:{tab}:visible", page.locator("#view-" + tab).is_visible()
                       and bool(page.locator(f"[data-slot={tab}]").inner_text().strip()))
            exported = csv_export(page, desktop, output / f"{family}.csv")
            models = {tuple(r[-2:]) for r in exported[1:]}
            expected_names = ({"NT SpCas9", "NT Cas12a"} if family == "nt" else
                              {"9-Hydra v2", "Cas12a CNN+BiLSTM manifest"})
            record(f"{family}:csv_model_version", exported[0][-2:] == ["모델", "모델 버전"]
                   and {name for name, _ in models} == expected_names
                   and all(version and version != "—" for _, version in models), models=sorted(models))
        mark("window_close")
    finally:
        trace = page.evaluate("window.releaseTrace || []")
        save(output / "events.json", trace)
        phases = sorted(markers + [event for event in trace if event["label"] != "event"], key=lambda event: event["time"])
        current = "non_nt" if v3 else "nt"
        with (output / "phases.jsonl").open("w", encoding="utf-8") as stream:
            for index, phase in enumerate(phases, 1):
                current = phase.get("selected", current)
                stream.write(json.dumps({"family": current, **phase, "id": index}) + "\n")


def run(output, mode, automation_site, source_python=None):
    contract_file = output / "contract.json"
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    for name, expected in contract["observer_sha256"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Observer changed after registration: {name}")
    if mode == "source":
        command = [str(source_python), "-X", "utf8", str(ROOT / "desktop_entry.py")]
    else:
        release_mode = "v3" if mode == "v3-relaunch" else mode
        folder = Path(contract["releases"][release_mode]["root"])
        if identify(folder, release_mode) != contract["releases"][release_mode]:
            raise ValueError(f"{mode} release changed after registration")
        command = [str(folder / EXE[release_mode])]
    target = output / mode
    target.mkdir(exist_ok=False)
    cwd = target / "unrelated-cwd"
    cwd.mkdir()
    site.addsitedir(str(automation_site))
    application = import_module("pywinauto").Application

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    browser_args = (f"--remote-debugging-address=127.0.0.1 --remote-debugging-port={port} "
                    '--disable-background-networking --proxy-server=http://127.0.0.1:9 '
                    '--host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE localhost"')
    env = {**os.environ, **contract["environment"], "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS": browser_args,
           "WEBVIEW2_USER_DATA_FOLDER": str(target / "webview-profile"), "MPLCONFIGDIR": str(target / "matplotlib")}
    if mode != "source":
        env = {**env, **{name: str(target / "empty-hf") for name in
                         ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")}}
        env.pop("PYTHONPATH", None)
    launch = {"command": command, "cwd": str(cwd), "contract_sha256": digest(contract_file),
              "browser_args": browser_args}
    if mode == "source":
        launch["source"] = source_identity(ROOT)
        probe = """
import importlib.metadata as metadata
import json
import sys
from huggingface_hub import try_to_load_from_cache
from predict import NT_TOKENIZER_REPO
files = {name: try_to_load_from_cache(NT_TOKENIZER_REPO, name)
         for name in ('vocab.txt', 'tokenizer_config.json', 'config.json')}
print(json.dumps({
    'python': sys.version,
    'distributions': {d.metadata['Name']: d.version for d in metadata.distributions()},
    'tokenizer_cache': {name: path for name, path in files.items() if isinstance(path, str)},
}))
"""
        runtime = json.loads(subprocess.check_output(
            [str(source_python), "-c", probe], cwd=ROOT, env=env, encoding="utf-8"
        ))
        runtime["tokenizer_cache"] = {
            name: {"path": path, "sha256": digest(Path(path))}
            for name, path in runtime["tokenizer_cache"].items()
        }
        launch["runtime"] = runtime
    save(target / "launch.json", launch)
    report = {"status": "fail", "checks": [], "families": {}}
    with (target / "window.log").open("w", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=1) as executor:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        sampler = executor.submit(sample_tree, process, target, contract["sample_interval_seconds"],
                                  contract["run_timeout_seconds"])
        desktop = None
        try:
            desktop = application(backend="win32").connect(process=process.pid, timeout=20)
            with sync_playwright() as driver:
                endpoint = webview_endpoint(port, process, 60)
                browser = driver.chromium.connect_over_cdp(endpoint)
                context = browser.contexts[0]
                page = context.pages[0] if context.pages else context.wait_for_event("page")
                cdp = context.new_cdp_session(page)
                cdp.send("Network.emulateNetworkConditions", {"offline": True, "latency": 0,
                         "downloadThroughput": 0, "uploadThroughput": 0})
                report["webview"] = cdp.send("Browser.getVersion")
                drive(page, process, desktop, target, mode, contract, report)
            report["status"] = "pass"
        except Exception:  # noqa: BLE001 -- preserve failures from the frozen-window boundary
            report["error"] = traceback.format_exc()
        finally:
            try:
                if process.poll() is None:
                    if desktop is None:
                        process.terminate()
                    else:
                        desktop.top_window().close()
            except Exception:  # noqa: BLE001 -- retain evidence if the native window is unresponsive
                report.update(status="fail", window_close_error=traceback.format_exc())
                process.terminate()
            finally:
                save(target / "window.json", report)
        result = sampler.result()
    print(json.dumps({"window": report["status"], "error": report.get("error"),
                      "sampler": {k: v for k, v in result.items() if k != "process_inventory"}}))
    return int(report["status"] != "pass" or result["exit_code"] != 0 or
               bool(result["survivors"] or result["access_errors"]) or result["timed_out"])


def compare(output):
    summarize(output)
    summary_file = output / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary["scope"] = "frozen Windows releases"
    save(summary_file, summary)
    source = json.loads((output / "source/window.json").read_text(encoding="utf-8"))
    frozen = summary["runs"]["v3"]["window"]
    parity = {"rtol": 1e-5, "atol": 1e-8, "families": {}}
    for family in ("non_nt", "nt"):
        old, new = (report["families"][family] for report in (source, frozen))
        matches = []
        for left, right in zip(old["rows"], new["rows"]):
            exact = all(left[key] == right[key] for key in ("site", "head", "cells"))
            numeric = all(math.isclose(left[key], right[key], rel_tol=1e-5, abs_tol=1e-8)
                          for key in ("raw", "percent"))
            matches.append(exact and numeric)
        parity["families"][family] = {"source_rows": len(old["rows"]), "frozen_rows": len(new["rows"]),
                                      "prediction_parity": len(old["rows"]) == len(new["rows"]) and all(matches),
                                      "xai_cases": len(new["xai"]), "xai_captions_parity": old["xai"] == new["xai"]}
    with (output / "v3/nt.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_v3 = [row[:-2] for row in csv.reader(stream)]
    with (BASELINE / "crispr_prediction_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        parity["nt_csv_base_columns_match_preserved"] = csv_v3 == list(csv.reader(stream))
    save(output / "parity.json", parity)
    return int(not parity["nt_csv_base_columns_match_preserved"]
               or any(not item["prediction_parity"] or not item["xai_captions_parity"]
                      for item in parity["families"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("output", type=Path)
    registration.add_argument("--legacy", type=Path, required=True)
    release_selection = registration.add_mutually_exclusive_group(required=True)
    release_selection.add_argument("--v3", type=Path)
    release_selection.add_argument("--legacy-only", action="store_true")
    execution = commands.add_parser("run")
    execution.add_argument("output", type=Path)
    execution.add_argument("mode", choices=("legacy", "v3", "v3-relaunch", "source"))
    execution.add_argument("--automation-site", type=Path, required=True)
    execution.add_argument("--source-python", type=Path, default=Path(sys.executable))
    comparison = commands.add_parser("compare")
    comparison.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "register":
        register(args.output.resolve(), args.legacy.resolve(), args.v3.resolve() if args.v3 else None)
        return 0
    if args.command == "compare":
        return compare(args.output.resolve())
    return run(args.output.resolve(), args.mode, args.automation_site.resolve(), args.source_python.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
