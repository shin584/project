"""Drive the real app/bridge/DOM. Launched by measure_ram.py, never by pytest."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback


def run(source, output, mode, contract):
    sys.path.insert(0, str(source))
    # Spawn inherits these paths; observer factories stay outside product code.
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    os.environ["DASHBOARD_RAM_OUTPUT"] = str(output)
    os.environ["DASHBOARD_RAM_MODE"] = mode
    import psutil
    import webview
    from bundle_runtime import configure_offline
    from desktop import api as api_module

    configure_offline()
    report = {"status": "fail", "checks": [], "observations": []}
    phase_lock = threading.Lock()
    phase_number = 0
    family = "nt" if mode == "legacy" else "non_nt"
    owner = []

    def phase(label, **details):
        nonlocal phase_number
        with phase_lock:
            phase_number += 1
            item = {"id": phase_number, "label": label, "family": family,
                    "time": time.time(), **details}
            with (output / "phases.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item) + "\n")
        print(label, details, flush=True)

    def check(name, passed, **details):
        report["checks"].append({"check": name, "passed": bool(passed), **details})
        if not passed:
            raise AssertionError(name)

    # Observe existing exposed calls, including automatic explanations after predict.
    for name in ("prepare", "scan_sites", "predict", "explain", "select_family"):
        original = getattr(api_module.DesktopApi, name, None)
        if original is None:
            continue

        def wrap(original, name):
            @functools.wraps(original)
            def invoke(self, *args, **kwargs):
                nonlocal family
                if name == "select_family":
                    family = args[0]
                details = {}
                if name == "explain":
                    details = dict(zip(("dna", "site", "head", "method", "params", "size"), args))
                    details.pop("dna", None)
                phase(name, **details)
                return original(self, *args, **kwargs)
            return invoke
        setattr(api_module.DesktopApi, name, wrap(original, name))

    if mode == "v3":
        from desktop import computation
        computation.FAMILY_FACTORIES.update(non_nt="ram_worker:ObservedNonNT", nt="ram_worker:ObservedNT")
        original_dispatcher = computation.ProcessDispatcher

        def observe_dispatcher(*args, **kwargs):
            dispatcher = original_dispatcher(*args, **kwargs)
            owner.append(dispatcher)
            return dispatcher
        computation.ProcessDispatcher = observe_dispatcher
    else:
        from desktop import renderer
        from ram_worker import ObservedLegacy
        renderer.PredictionRenderer = ObservedLegacy

    original_create = webview.create_window

    def create(*args, **kwargs):
        window = original_create(*args, **kwargs)
        if window is None:
            raise RuntimeError("Could not create WebView2 window")

        def wait(expression):
            deadline = time.monotonic() + contract["operation_timeout_seconds"]
            while time.monotonic() < deadline:
                if window.evaluate_js(expression):
                    return
                error = window.evaluate_js("document.querySelector('.error')?.textContent")
                if error:
                    raise RuntimeError(error)
                time.sleep(.1)
            raise TimeoutError(expression)

        def idle():
            wait("!document.getElementById('analyze').disabled && document.getElementById('task-state').hidden")
            # Include async image decode/compositing and the idle retained caches.
            wait("Array.from(document.querySelectorAll('[data-slot=xai] img')).every(i => i.complete && i.naturalWidth > 0)")
            time.sleep(contract["settle_seconds"])

        def observe(label):
            report["observations"].append({"label": label, "time": time.time(), "family": family,
                "ui_model_modules": sorted({"torch", "keras", "tensorflow", "transformers", "tokenizers",
                                             "numpy", "matplotlib"}.intersection(sys.modules)),
                "worker_pid": owner[0].pid if owner else None,
                "images": window.evaluate_js("document.querySelectorAll('[data-slot=xai] img').length")})

        def explanations():
            for case in contract["cases"]:
                for method in (("ig", "shap", "rollout") if family == "nt" else ("ig", "shap")):
                    # Set all controls in one task and submit one existing form event.
                    script = """(() => {
                      const site = document.getElementById('xai-site');
                      site.value = SITE;
                      const heads = JSON.parse(site.selectedOptions[0].dataset.heads);
                      const head = document.getElementById('xai-head');
                      head.replaceChildren(...heads.map(h => new Option(h,h)));
                      head.value = HEAD;
                      document.querySelector('[name=xai-method][value=METHOD]').checked = true;
                      document.getElementById('xai-controls').requestSubmit();
                    })()""".replace("SITE", json.dumps(case["site"])).replace("HEAD", json.dumps(case["head"])).replace("METHOD", method)
                    window.evaluate_js(script)
                    idle()
                    expected = 2 if method == "shap" else 1
                    check(f"{family}:{case['site']}:{method}:rendered", window.evaluate_js(
                        "document.querySelectorAll('[data-slot=xai] img').length") == expected)
                    observe(f"{case['site']}:{method}")

        def drive():
            try:
                wait("!document.getElementById('analyze').disabled")
                idle()
                observe("startup")
                window.evaluate_js("document.getElementById('dna').value=" + json.dumps(contract["dna"]) +
                                   ";document.getElementById('analyze').click()")
                idle()
                observe("prediction_and_automatic_xai")
                explanations()
                if mode == "v3":
                    check("default_is_non_nt", owner[0].family == "non_nt")
                    for new_family in ("nt", "non_nt"):
                        old = psutil.Process(owner[0].pid)
                        descendants = old.children(recursive=True)
                        old_identity = {"pid": old.pid, "create_time": old.create_time()}
                        window.evaluate_js("""(() => {
                          const select = document.getElementById('model-family'); select.value = FAMILY;
                          select.dispatchEvent(new Event('change'));
                        })()""".replace("FAMILY", json.dumps(new_family)))
                        idle()
                        check(f"{new_family}:previous_worker_and_children_exited",
                              not old.is_running() and not any(p.is_running() for p in descendants),
                              old=old_identity, descendants=[p.pid for p in descendants])
                        check(f"{new_family}:dna_retained", window.evaluate_js("document.getElementById('dna').value") == contract["dna"])
                        observe("after_switch_automatic_analysis")
                        if new_family == "nt":
                            explanations()
                        else:
                            phase("return_stable")
                            time.sleep(contract["stable_seconds"])
                            observe("return_stable")
                check("v3_ui_has_no_model_runtime", mode == "legacy" or all(
                    not item["ui_model_modules"] for item in report["observations"]))
                report["status"] = "pass"
            except Exception:
                report["error"] = traceback.format_exc()
                print(report["error"], flush=True)
            finally:
                phase("closing")
                window.destroy()

        window.events.loaded += lambda: threading.Thread(target=drive, daemon=True).start()
        return window

    webview.create_window = create
    from desktop.app import main
    try:
        main()
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        (output / "window.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    source, output, mode, contract_path = sys.argv[1:]
    raise SystemExit(run(Path(source), Path(output), mode,
                         json.loads(Path(contract_path).read_text(encoding="utf-8"))))
