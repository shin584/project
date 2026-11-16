"""Disposable #179 executable harness, independent of the desktop API work."""

import argparse
import base64
import io
import json
import os
import random
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from bundle_runtime import (
    configure_offline,
    is_packaged,
    resource_root,
    validate_bundle,
)

configure_offline()
# A new empty cache on every invocation prevents the developer's cache hiding omissions.
_cache = tempfile.TemporaryDirectory(prefix="xai-gate1-hf-")
os.environ["HF_HOME"] = _cache.name
os.environ["HF_HUB_CACHE"] = str(Path(_cache.name) / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(Path(_cache.name) / "transformers")


class Monitor:
    """Sample the application tree, including WebView2 subprocesses."""

    def __init__(self):
        self.stop = threading.Event()
        self.peak_webview_rss = 0
        self.peak_tree_rss = 0
        self.listeners = set()
        self.errors = set()
        self.network_attempts = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def audit(self, event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
            self.network_attempts.append(event)
            raise RuntimeError(f"Gate 1 forbids Python network access: {event}")

    def run(self):
        import psutil

        root = psutil.Process()
        while not self.stop.is_set():
            total = webview = 0
            try:
                processes = [root, *root.children(recursive=True)]
                for proc in processes:
                    try:
                        rss = proc.memory_info().rss
                        total += rss
                        if proc.name().lower() == "msedgewebview2.exe":
                            webview += rss
                        for conn in proc.net_connections(kind="tcp"):
                            if conn.status == psutil.CONN_LISTEN:
                                self.listeners.add(f"{proc.pid}:{conn.laddr.port}")
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied as exc:
                        self.errors.add(str(exc))
                self.peak_tree_rss = max(self.peak_tree_rss, total)
                self.peak_webview_rss = max(self.peak_webview_rss, webview)
            except psutil.Error as exc:
                self.errors.add(str(exc))
            self.stop.wait(0.1)

    def snapshot(self):
        return {
            "webview2_peak_rss_bytes": self.peak_webview_rss,
            "process_tree_peak_rss_bytes": self.peak_tree_rss,
            "tcp_listeners": sorted(self.listeners),
            "monitor_errors": sorted(self.errors),
            "blocked_python_network_attempts": self.network_attempts[:],
            "sampling_interval_seconds": 0.1,
        }


def compute():
    """Exercise actual scanner, lazy predictors, uncached SHAP, backward, attention."""
    import numpy as np
    import torch
    from pam_scanner import ScanResult, scan_all
    from predict import load_predictors, run_prediction
    from xai import explain, shap_cache

    torch.set_num_threads(4)
    rng = random.Random(179)
    dna = list("".join(rng.choice("ACGT") for _ in range(81)))
    dna[16:20] = "TTTA"
    dna[40:43] = "AGG"
    dna = "".join(dna)
    scanned = scan_all(dna)
    cas12a = next(
        s
        for s in scanned.sites
        if s.cas_type == "Cas12a" and s.strand == "+" and s.pam_start == 16
    )
    spcas9 = next(
        s
        for s in scanned.sites
        if s.cas_type == "SpCas9" and s.strand == "+" and s.pam_start == 40
    )
    predictors = load_predictors(warmup=False)
    # Only Cas12a is requested first; SpCas9 must load on the subsequent NGG request.
    first = run_prediction(ScanResult(dna, sites=[cas12a]), predictors)
    second = run_prediction(ScanResult(dna, sites=[spcas9]), predictors)
    if not first.variant_results or not any(
        row.cas_name == "SpCas9(WT)" for row in second.variant_results
    ):
        raise RuntimeError("Both model predictions, including SpCas9(WT), are required")
    handle = predictors[("A", "Cas12a")].explain_handle("Cas12a")
    seq = cas12a.model_input_seq
    key = shap_cache.shap_cache_key("A/Cas12a", "Cas12a", seq, 8, 1)
    if shap_cache.load(key) is not None:
        raise RuntimeError("SHAP gate sequence unexpectedly hit the precomputed cache")
    progress = []
    shap = explain.kernel_shap(
        handle,
        seq,
        nsamples=8,
        n_background=1,
        on_batch=lambda done, total: progress.append([done, total]),
    )
    if not progress:
        raise RuntimeError("SHAP did not execute model inference")
    ig = explain.integrated_gradients(handle, seq, steps=2, chunk_size=1)
    rollout = explain.attention_rollout(handle, seq)
    for values in (
        shap.values,
        ig.attribution,
        rollout.position_profile,
        [shap.pred, ig.pred],
        [r.raw_score for r in second.variant_results],
    ):
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite model/XAI output")
    if any(
        values.shape != (len(seq),)
        for values in (shap.values, ig.attribution, rollout.position_profile)
    ):
        raise RuntimeError("XAI output length differs from the model window")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 4), layout="constrained")
    axes = figure.subplots(3, 1)
    for ax, values, label in zip(
        axes,
        (shap.values, ig.attribution, rollout.position_profile),
        ("SHAP", "IG", "Rollout"),
    ):
        ax.bar(range(len(seq)), values)
        ax.set_ylabel(label)
    image = io.BytesIO()
    figure.savefig(image, format="png")
    uri = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")
    return {
        "input_dna": dna,
        "prediction_counts": {
            "Cas12a": len(first.variant_results),
            "SpCas9": len(second.variant_results),
        },
        "shap": {
            "cache_hit": False,
            "nsamples": 8,
            "n_background": 1,
            "inference_batches": len(progress),
            "pred": shap.pred,
        },
        "ig": {"steps": 2, "pred": ig.pred},
        "rollout": {"n_layers": rollout.n_layers},
    }, uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.cwd() / "gate1-result.json")
    parser.add_argument(
        "--headless", action="store_true", help="Diagnostic only; cannot pass GUI gate"
    )
    parser.add_argument("--auto-close", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "smoke_status": "fail",
        "gate_status": "pending_clean_environment",
        "frozen": is_packaged(),
        "empty_hf_cache": True,
        "webview2_policy": "bundled_fixed_version",
        "chart_rendered": False,
        "executable": sys.executable,
    }
    monitor = Monitor()
    sys.addaudithook(monitor.audit)

    def save():
        report.update(monitor.snapshot())
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def run(window=None):
        started = time.monotonic()
        try:
            if window is not None and not window.events.loaded.wait(60):
                raise RuntimeError("WebView did not load within 60 seconds")
            results, uri = compute()
            report.update(results)
            if window is not None:
                window.evaluate_js(
                    "document.getElementById('chart').src = " + json.dumps(uri)
                )
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if window.evaluate_js(
                        "document.getElementById('chart').naturalWidth > 0"
                    ):
                        report["chart_rendered"] = True
                        break
                    time.sleep(0.1)
                if not report["chart_rendered"]:
                    raise RuntimeError("WebView did not display the PNG data URI")
                window.evaluate_js(
                    "document.getElementById('status').textContent = 'Checks complete'"
                )
            if monitor.listeners or monitor.errors or monitor.network_attempts:
                raise RuntimeError(
                    "Network/port monitoring failed; inspect result JSON"
                )
            report["smoke_status"] = "pass" if window is not None else "headless_only"
        except Exception:  # noqa: BLE001 -- executable boundary must persist failure evidence
            report["error"] = traceback.format_exc()
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            save()
            if window is not None and args.auto_close:
                window.destroy()

    try:
        if not is_packaged():
            raise RuntimeError("Run the PyInstaller executable for this gate")
        validate_bundle()
        monitor.thread.start()
        if args.headless:
            run()
        else:
            import webview

            webview.settings["WEBVIEW2_RUNTIME_PATH"] = str(
                resource_root() / "webview2"
            )
            webview.settings["REMOTE_DEBUGGING_PORT"] = None
            window = webview.create_window(
                "#179 packaging gate",
                html="""<!doctype html>
                <meta charset="utf-8"><h1>Offline packaging gate</h1>
                <p id="status">Running both models and XAI…</p><img id="chart" alt="XAI chart">""",
            )
            webview.start(
                run, args=(window,), gui="edgechromium", http_server=False, debug=False
            )
    except Exception:  # noqa: BLE001 -- include GUI/import failures in the gate report
        report["error"] = traceback.format_exc()
    finally:
        monitor.stop.set()
        if monitor.thread.is_alive():
            monitor.thread.join(timeout=5)
        if monitor.listeners or monitor.errors or monitor.network_attempts:
            report["smoke_status"] = "fail"
        save()
        _cache.cleanup()
    return 0 if report["smoke_status"] in {"pass", "headless_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
