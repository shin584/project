"""Exploratory performance only; does not validate numerical quality or gates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIGS = [("ig", n, 0) for n in (32, 64, 128, 256)] + [
    ("shap", n, b) for n, b in ((64, 4), (128, 4), (256, 4), (512, 4),
                              (64, 8), (64, 16), (64, 32), (512, 32))]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def child(model):
    sys.path.insert(0, str(ROOT))
    import psutil
    process = psutil.Process()
    versions = {}
    for pkg in ("keras", "torch", "numpy", "shap", "h5py", "psutil"):
        versions[pkg] = importlib.metadata.version(pkg)
    contract = json.loads((ROOT / "docs/runtime/contract.json").read_text(encoding="utf-8"))
    spec = contract["models"][model + "_non_nt"]
    report = {
        "label": "exploratory performance; numerical quality not revalidated",
        "run_id": HERE.name, "model": model, "pid": os.getpid(),
        "protocol": "perf-v1: single fresh process per model; sequential; seed0; disk cache off; chunk8; 3 repeats; 10ms RSS sampling",
        "script_sha256": digest(__file__),
        "contract_sha256": digest(ROOT / "docs/runtime/contract.json"),
        "model_sha256": digest(ROOT / spec["path"]),
        "code_sha256": {p: digest(ROOT / p) for p in
                        ("non_nt_runtime.py", "xai/explain.py", "model_source.py", "predict.py")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "processor": platform.processor(),
                        "packages": versions, "torch_threads": 4,
                        "rss_poll_seconds": 0.01},
        "case": contract["cases"][model][0], "head": spec["verified_head"],
        "rows": [], "status": "running"
    }
    configs = CONFIGS
    if model == "cas12a":
        priority = [("ig", 32, 0), ("ig", 256, 0), ("shap", 64, 4),
                    ("shap", 512, 32), ("shap", 512, 4), ("shap", 64, 32)]
        configs = priority + [c for c in CONFIGS if c not in priority]
        report["protocol"] += "; Cas12a reordered to prioritize main comparisons within 180s; all other settings retained"
    report["configuration_order"] = configs
    output = HERE / (model + ".json")
    save(output, report)
    started = time.perf_counter()
    import non_nt_runtime as rt
    from xai import explain
    rt.import_keras()
    report["runtime_import_wall_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    runtime = rt.NonNTRuntime.load(model)
    report["model_load_wall_seconds"] = time.perf_counter() - started
    report["tensorflow_importable"] = rt.tensorflow_present()
    report["tensorflow_imported"] = "tensorflow" in sys.modules
    window = report["case"]["window"]
    handle = rt.explain_handle(runtime, head=spec["verified_head"]["index"],
                               window_length=len(window),
                               derive=rt.cas12a_derive if model == "cas12a" else None)
    warmup_started = time.perf_counter()
    explain.integrated_gradients(handle, window, steps=8, chunk_size=8)
    # Load SHAP/scipy/sklearn before timing; tiny warmup excludes their import overhead.
    explain.kernel_shap(handle, window, nsamples=2, n_background=1,
                        seed=0, use_disk_cache=False)
    report["warmup_wall_seconds"] = time.perf_counter() - warmup_started
    save(output, report)
    for method, count, background in configs:
        row = {"method": method, "steps" if method == "ig" else "nsamples": count,
               "n_background": background, "repeats": []}
        report["rows"].append(row)
        for repeat in range(3):
            baseline = process.memory_info().rss
            samples = [baseline]
            done = threading.Event()
            def sample():
                while not done.wait(0.01):
                    samples.append(process.memory_info().rss)
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            cpu_before = process.cpu_times()
            began = time.perf_counter()
            try:
                if method == "ig":
                    explain.integrated_gradients(handle, window, steps=count, chunk_size=8)
                else:
                    explain.kernel_shap(handle, window, nsamples=count,
                                        n_background=background, seed=0, use_disk_cache=False)
            finally:
                elapsed = time.perf_counter() - began
                cpu_after = process.cpu_times()
                mem = process.memory_info()
                samples.append(mem.rss)
                done.set()
                sampler.join()
            row["repeats"].append({
                "repeat": repeat + 1, "wall_seconds": elapsed,
                "cpu_seconds": cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
                "baseline_rss_bytes": baseline, "sampled_peak_rss_bytes": max(samples),
                "peak_increase_over_baseline_bytes": max(samples) - baseline,
                "after_rss_bytes": mem.rss, "process_lifetime_peak_wset_bytes": mem.peak_wset,
                "rss_samples": len(samples)
            })
            row["median_wall_seconds"] = statistics.median(r["wall_seconds"] for r in row["repeats"])
            row["median_cpu_seconds"] = statistics.median(r["cpu_seconds"] for r in row["repeats"])
            row["max_sampled_peak_rss_bytes"] = max(r["sampled_peak_rss_bytes"] for r in row["repeats"])
            row["median_baseline_rss_bytes"] = statistics.median(r["baseline_rss_bytes"] for r in row["repeats"])
            save(output, report)
        print(json.dumps({"model": model, **{k: v for k, v in row.items() if k != "repeats"}}), flush=True)
    report["status"] = "completed"
    save(output, report)


def parent():
    results = []
    for model in ("spcas9", "cas12a"):
        began = time.perf_counter()
        with (HERE / (model + ".stdout.log")).open("w", encoding="utf-8") as stdout, \
             (HERE / (model + ".stderr.log")).open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                                     "--model", model], cwd=ROOT, stdout=stdout, stderr=stderr,
                                    env={**os.environ, "KERAS_BACKEND": "torch"})
            try:
                returncode = proc.wait(timeout=180)
                status = "completed" if returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                returncode, status = proc.returncode, "timeout_180_seconds"
        result = {"model": model, "status": status, "returncode": returncode,
                  "wall_seconds_including_startup": time.perf_counter() - began}
        results.append(result)
        save(HERE / "orchestration.json", results)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("spcas9", "cas12a"))
    args = parser.parse_args()
    try:
        child(args.model) if args.model else parent()
    except Exception:
        traceback.print_exc()
        raise
