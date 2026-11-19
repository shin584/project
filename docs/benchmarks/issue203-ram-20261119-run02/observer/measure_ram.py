"""Register, measure and summarize #203 source-window process trees on Windows.

The external sampler can also monitor a packaged executable with the `watch`
command. Packaged workloads must be driven through their own actual UI.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_identity(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    files = git("ls-files", "*.py").splitlines()
    resources = sorted([*root.glob("models/*.safetensors"), *root.glob("models/non_nt/**/*.keras"),
                        *root.glob(".packaging/assets/tokenizer/*"),
                        *root.glob("data/precomputed_shap/*.npz")])
    return {"root": str(root), "commit": git("rev-parse", "HEAD"),
            "tracked_diff": git("diff", "HEAD", "--", "*.py"),
            "python_sha256": {name: digest(root / name) for name in files},
            "resources": {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                          for p in resources if p.is_file()}}


def register(output, legacy):
    output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads((ROOT / "docs/equivalence-baseline/capture.json").read_text(encoding="utf-8"))
    distributions = ("torch", "keras", "transformers", "safetensors", "numpy", "shap", "psutil",
                     "pywebview", "pythonnet", "tensorflow", "h5py", "matplotlib")
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    contract = {
        "issue": 203, "registered_at": datetime.now(timezone.utc).isoformat(),
        "scope": "source execution on this development PC; no distribution certification",
        "sources": {"legacy": source_identity(legacy), "v3": source_identity(ROOT)},
        "observer_sha256": {p.name: digest(p) for p in (HERE / "measure_ram.py", HERE / "ram_window.py", HERE / "ram_worker.py")},
        "pc": {"host": platform.node(), "platform": platform.platform(), "processor": platform.processor(),
               "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
               "ram_bytes": psutil.virtual_memory().total},
        "runtime": {"python": sys.version, "executable": sys.executable, "packages": versions},
        "environment": {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "KERAS_BACKEND": "torch",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
        "dna": baseline["input_seq"],
        "cases": [{"site": row["site_key"], "head": row["head"]} for row in baseline["xai"]],
        "settings": baseline["settings"], "graph_size_percent": 80,
        "sample_interval_seconds": .1, "settle_seconds": 5, "stable_seconds": 10,
        "operation_timeout_seconds": 900, "run_timeout_seconds": 3600,
        "sequence": ["legacy:startup,predict,automatic-XAI,both-Cas-IG/SHAP/Rollout,close",
                     "v3:default-nonNT,startup,predict,automatic-XAI,both-Cas-IG/SHAP",
                     "v3:switch-NT,automatic-analysis,both-Cas-IG/SHAP/Rollout",
                     "v3:return-nonNT,automatic-analysis,10-second-stable,close"],
        "metrics": {"rss": "sum of simultaneous process working sets; shared pages can be counted more than once",
                    "private": "sum of process PrivateUsage/commit bytes, including paged-out memory; not resident RAM",
                    "peak": "largest observed simultaneous sum in each phase, never sum of per-process lifetime peaks",
                    "stable": "median/min/max of final 2 seconds in each phase; return has a dedicated 10-second phase"},
        "limitations": ["One sequential run per program; not a statistical performance guarantee",
                        "100 ms target polling can miss shorter transients; actual sample times and gaps retained",
                        "UI/worker/WebView2 and tracked descendants retained after reparenting; sub-interval children can be missed",
                        "Instrumented production app; observers retain metadata only; no substitution of models/IPC/rendering/cache",
                        "Prediction naturally triggers default XAI; bridge phase marks separate calls; render/compositing counted until next call",
                        "Legacy numeric XAI units/disk-cache policy preserved; v3 corrected units and cache policy preserved",
                        "Development environment includes TensorFlow; actual imports and their memory cost are included",
                        "No absolute RAM cap or target reduction rate; model byte sizes identify assets only"]}
    save(output / "contract.json", contract)
    print(output / "contract.json", flush=True)


def sample_tree(process, output, interval, timeout):
    known = {}
    inventory = {}
    started = time.monotonic()
    max_gap = 0
    previous = started
    sampling_errors = []
    timed_out = False
    exited_at = None
    count = 0
    with gzip.open(output / "samples.jsonl.gz", "wt", encoding="utf-8") as stream:
        while True:
            tick = time.monotonic()
            max_gap = max(max_gap, tick - previous)
            previous = tick
            alive = []
            candidates = list(known.values())
            try:
                root = psutil.Process(process.pid)
                candidates.extend([root, *root.children(recursive=True)])
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied as exc:
                sampling_errors.append(str(exc))
            for child in candidates:
                try:
                    identity = (child.pid, child.create_time())
                    if identity not in known:
                        known[identity] = child
                        inventory[str(identity)] = {"pid": child.pid, "created": identity[1], "ppid": child.ppid(),
                                                    "name": child.name(), "exe": child.exe(), "cmdline": child.cmdline()}
                    if child.is_running() and identity not in {item[0] for item in alive}:
                        alive.append((identity, child))
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied as exc:
                    sampling_errors.append(str(exc))
            rows = []
            for identity, child in alive:
                try:
                    memory = child.memory_info()
                    rows.append({"pid": child.pid, "created": identity[1], "ppid": child.ppid(),
                                 "rss": memory.rss, "private": memory.private,
                                 "page_faults": memory.num_page_faults})
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied as exc:
                    sampling_errors.append(str(exc))
            stream.write(json.dumps({"time": time.time(), "elapsed": tick - started,
                                     "query_seconds": time.monotonic() - tick, "processes": rows,
                                     "rss": sum(p["rss"] for p in rows),
                                     "private": sum(p["private"] for p in rows),
                                     "host_available": psutil.virtual_memory().available}) + "\n")
            count += 1
            if process.poll() is not None:
                exited_at = exited_at or tick
                if tick - exited_at >= 3:
                    break
            if tick - started > timeout:
                timed_out = True
                break
            time.sleep(max(0, interval - (time.monotonic() - tick)))
    survivors = [p.pid for p in known.values() if p.is_running()]
    result = {"root_pid": process.pid, "exit_code": process.poll(), "samples": count,
              "seconds": time.monotonic() - started, "max_sample_gap_seconds": max_gap,
              "access_errors": sampling_errors, "survivors": survivors, "timed_out": timed_out,
              "process_inventory": inventory}
    save(output / "sampler.json", result)
    # Cleanup applies only to identities discovered under this launched root.
    for child in known.values():
        if child.is_running():
            child.kill()
    process.wait(timeout=30)
    return result


def run(output, mode):
    contract_path = output / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for name, expected in contract["observer_sha256"].items():
        if digest(HERE / name) != expected:
            raise ValueError("Observer changed since registration: " + name)
    source = Path(contract["sources"][mode]["root"])
    # Verify source identity without loading any model into the measured UI.
    if source_identity(source) != contract["sources"][mode]:
        raise ValueError("Source/resources changed since registration")
    target = output / mode
    target.mkdir(exist_ok=False)
    command = [sys.executable, "-X", "utf8", str(HERE / "ram_window.py"), str(source), str(target), mode, str(contract_path)]
    save(target / "launch.json", {"command": command, "cwd": str(source), "contract_sha256": digest(contract_path),
                                  "started_at": datetime.now(timezone.utc).isoformat()})
    with (target / "window.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=source, env={**os.environ, **contract["environment"]},
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        result = sample_tree(process, target, contract["sample_interval_seconds"], contract["run_timeout_seconds"])
    print(json.dumps({k: v for k, v in result.items() if k != "process_inventory"}), flush=True)
    return int(result["exit_code"] != 0 or bool(result["survivors"] or result["access_errors"]) or result["timed_out"])


def summarize(output):
    summary = {"scope": "source only", "runs": {}}
    for mode in ("legacy", "v3"):
        target = output / mode
        phases = [{"time": 0, "id": 0, "family": "nt" if mode == "legacy" else "non_nt", "label": "startup_imports"}]
        phases.extend(json.loads(line) for line in (target / "phases.jsonl").read_text(encoding="utf-8").splitlines())
        groups = defaultdict(list)
        phase_index = 0
        with gzip.open(target / "samples.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                sample = json.loads(line)
                while phase_index + 1 < len(phases) and phases[phase_index + 1]["time"] <= sample["time"]:
                    phase_index += 1
                groups[phase_index].append(sample)
        records = []
        for index, samples in groups.items():
            tail = [s for s in samples if s["time"] >= samples[-1]["time"] - 2]
            record = {**phases[index], "samples": len(samples), "observed_seconds": samples[-1]["time"] - samples[0]["time"]}
            for metric in ("rss", "private"):
                record[metric] = {"peak": max(s[metric] for s in samples),
                                  "tail_median": statistics.median(s[metric] for s in tail),
                                  "tail_min": min(s[metric] for s in tail), "tail_max": max(s[metric] for s in tail)}
            records.append(record)
        summary["runs"][mode] = {"phases": records,
                                  "window": json.loads((target / "window.json").read_text(encoding="utf-8")),
                                  "sampler": json.loads((target / "sampler.json").read_text(encoding="utf-8"))}
    save(output / "summary.json", summary)
    print(output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("output", type=Path)
    registration.add_argument("--legacy", type=Path, required=True)
    execution = commands.add_parser("run")
    execution.add_argument("output", type=Path)
    execution.add_argument("mode", choices=("legacy", "v3"))
    summary = commands.add_parser("summarize")
    summary.add_argument("output", type=Path)
    watch = commands.add_parser("watch", help="External sampler for a manually driven source/frozen app")
    watch.add_argument("output", type=Path)
    watch.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        register(output, args.legacy.resolve())
    elif args.command == "run":
        return run(output, args.mode)
    elif args.command == "summarize":
        summarize(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        if not command:
            parser.error("watch requires executable argv after --")
        save(output / "launch.json", {"command": command, "cwd": os.getcwd(), "scope": "manual workload; see external preregistration"})
        with (output / "window.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            result = sample_tree(process, output, .1, 3600)
        return int(result["exit_code"] != 0 or bool(result["survivors"] or result["access_errors"]) or result["timed_out"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
