"""Derive the #203 comparison and lifecycle checks from retained raw evidence."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from measure_ram import save


def peak(phases):
    return {metric: max(p[metric]["peak"] for p in phases) for metric in ("rss", "private")}


def runtime_rows(folder):
    return [json.loads(line) for path in sorted(folder.glob("runtime-*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()]


def non_nt_only(row):
    return len(row["predictors"]) == 2 and all(
        p["non_nt_runtime_loaded"] and not p["nt_model_loaded"] and not p["nt_tokenizer_loaded"]
        and p["class"].startswith("NonNT") for p in row["predictors"])


def analyze(output, cold):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    legacy = summary["runs"]["legacy"]
    v3 = summary["runs"]["v3"]
    phases = v3["phases"]
    switches = [p for p in phases if p["label"] == "select_family"]
    first_switch, return_switch = switches
    default = [p for p in phases if p["time"] < first_switch["time"]]
    nt = [p for p in phases if first_switch["time"] <= p["time"] < return_switch["time"]]
    stable = next(p for p in phases if p["label"] == "return_stable")
    first_auto = next(p for p in default if p["label"] == "explain")
    runtime = runtime_rows(output / "v3")
    before = [r for r in runtime if r["time"] < first_switch["time"] and r["operation"] == "predict"]
    after = [r for r in runtime if r["time"] > return_switch["time"] and r["operation"] == "predict"]
    nt_loaded = [r for r in runtime if first_switch["time"] < r["time"] < return_switch["time"]
                 and r["operation"] == "predict"]
    worker_pids = {r["pid"] for r in runtime}
    nt_pids = {r["pid"] for r in nt_loaded}
    checks = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    for name, run in (("legacy", legacy), ("v3", v3)):
        check(name + ":window", run["window"]["status"] == "pass" and all(c["passed"] for c in run["window"]["checks"]))
        sampler = run["sampler"]
        check(name + ":sampler_exit", sampler["exit_code"] == 0 and not sampler["survivors"]
              and not sampler["access_errors"] and not sampler["timed_out"])
        commands = [p["cmdline"] for p in sampler["process_inventory"].values()]
        check(name + ":webview_renderer_and_gpu_included", all(
            any(any(token in arg for arg in command) for command in commands)
            for token in ("--type=renderer", "--type=gpu-process")))
    check("v3:ui_runtime_not_loaded", all(not row["ui_model_modules"] for row in v3["window"]["observations"]))
    check("v3:default_non_nt_only", len(before) == 1 and non_nt_only(before[0]))
    check("v3:return_non_nt_only", len(after) == 1 and non_nt_only(after[0]))
    check("v3:both_nt_models_and_tokenizers_loaded", len(nt_loaded) == 1 and len(nt_loaded[0]["predictors"]) == 2
          and all(p["nt_model_loaded"] and p["nt_tokenizer_loaded"] for p in nt_loaded[0]["predictors"]))
    concurrent = 0
    nt_after_return = 0
    samples = 0
    with gzip.open(output / "v3/samples.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            pids = {p["pid"] for p in row["processes"]}
            concurrent = max(concurrent, len(worker_pids & pids))
            if row["time"] >= stable["time"]:
                nt_after_return += len(nt_pids & pids)
            samples += 1
    check("v3:one_worker_at_a_time", concurrent == 1 and len(worker_pids) == 3)
    check("v3:no_nt_worker_after_return", nt_after_return == 0)
    check("v3:raw_sample_count_matches", samples == v3["sampler"]["samples"])
    baseline_peak = peak([p for p in legacy["phases"] if p["label"] != "closing"])
    default_peak = peak(default)
    result = {"scope": "source execution on this PC, development TensorFlow cost included; not distribution validation",
              "legacy_peak_bytes": baseline_peak, "default_non_nt_peak_bytes": default_peak,
              "v3_nt_peak_bytes": peak(nt),
              "observed_reduction_percent": {metric: 100 * (1 - default_peak[metric] / baseline_peak[metric])
                                               for metric in ("rss", "private")},
              "return_stable": {metric: stable[metric] for metric in ("rss", "private")},
              "initial_auto_xai_tail_median_bytes": {metric: first_auto[metric]["tail_median"] for metric in ("rss", "private")},
              "return_minus_initial_auto_bytes": {metric: stable[metric]["tail_median"] - first_auto[metric]["tail_median"]
                                                   for metric in ("rss", "private")},
              "max_concurrent_workers": concurrent, "nt_worker_observations_after_return": nt_after_return,
              "worker_pids": sorted(worker_pids), "checks": checks,
              "cold_supplement": None}
    if cold is not None:
        supplement = json.loads((cold / "summary.json").read_text(encoding="utf-8"))["runs"]["legacy"]
        cold_runtime = runtime_rows(cold / "legacy")
        last = max(cold_runtime, key=lambda r: r["time"])
        counters = last["compute_counts"]
        check("cold:real_shap_without_disk_hit", counters["kernel_shap"] == 2 and counters["disk_shap_lookups"] == 2
              and counters["disk_shap_hits"] == 0 and last["disk_shap_bypass"])
        check("cold:window_and_cleanup", supplement["window"]["status"] == "pass"
              and supplement["sampler"]["exit_code"] == 0 and not supplement["sampler"]["survivors"]
              and not supplement["sampler"]["access_errors"] and not supplement["sampler"]["timed_out"])
        result["cold_supplement"] = {"path": str(cold), "peak_bytes": peak(
            [p for p in supplement["phases"] if p["label"] != "closing"]), "compute_counts": counters}
        cold_peak = result["cold_supplement"]["peak_bytes"]
        result["cold_supplement"]["observed_reduction_percent"] = {
            metric: 100 * (1 - default_peak[metric] / cold_peak[metric]) for metric in ("rss", "private")}
    result["counts"] = {"passed": sum(c["passed"] for c in checks),
                        "failed": sum(not c["passed"] for c in checks), "skipped": 0}
    save(output / "comparison.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["counts"]["failed"] > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cold", type=Path)
    args = parser.parse_args()
    raise SystemExit(analyze(args.output.resolve(), args.cold.resolve() if args.cold else None))
