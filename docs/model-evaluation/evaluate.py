"""#202: freeze inputs, evaluate both complete test sets, retain replay evidence.

Run from any directory; --work must be new. Inference uses the copied runtime
source and pinned checkpoint bytes, in separate TensorFlow/torch processes.
No training or partition creation code is called.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def performance(target, prediction) -> dict:
    """Score a head, keeping NaN-label exclusions explicit and failing on bad output."""
    target, prediction = (np.asarray(a, dtype=np.float64) for a in (target, prediction))
    require(target.ndim == 1 and target.shape == prediction.shape, "metric shape mismatch")
    require(np.isfinite(prediction).all(), "nonfinite prediction")
    require(not np.isinf(target).any(), "infinite target")
    measured = ~np.isnan(target)
    y, p = target[measured], prediction[measured]
    values: dict[str, float | None] = dict.fromkeys(("pearson", "spearman", "mae", "mse"))
    reasons = {}
    if len(y):
        values.update(mae=float(np.mean(np.abs(y - p))), mse=float(np.mean((y - p) ** 2)))
    else:
        reasons.update(mae="no measured targets", mse="no measured targets")
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(p) == 0:
        reason = "fewer than two observations or constant target/prediction"
        reasons.update(pearson=reason, spearman=reason)
    else:
        values.update(pearson=float(np.asarray(pearsonr(y, p))[0]),
                      spearman=float(np.asarray(spearmanr(y, p))[0]))
    return {"n": len(y), "missing": int((~measured).sum()), **values,
            "unavailable_reasons": reasons}


def compare(target, expected, actual, policy: dict) -> dict:
    """Judge all outputs and both raw/clipped metrics under the frozen bounds."""
    expected, actual = (np.asarray(a, dtype=np.float64) for a in (expected, actual))
    require(expected.shape == actual.shape == np.shape(target), "parity shape mismatch")
    require(np.isfinite(expected).all() and np.isfinite(actual).all(), "nonfinite parity input")
    errors = np.abs(expected - actual)
    output_ok = bool(np.all(errors <= policy["output"]["atol"]
                            + policy["output"]["rtol"] * np.abs(expected)))
    metrics = {}
    for label, a, b in (("raw", expected, actual),
                        ("clipped", np.clip(expected, 0, 1), np.clip(actual, 0, 1))):
        reference, observed = performance(target, a), performance(target, b)
        deltas = {}
        for name, tolerance in policy["metric_absolute_tolerance"].items():
            x, y = reference[name], observed[name]
            delta = abs(x - y) if x is not None and y is not None else None
            passed = (x is None and y is None) if delta is None else delta <= tolerance
            deltas[name] = {"absolute_difference": delta, "passed": passed}
        metrics[label] = {"reference": reference, "observed": observed,
                          "differences": deltas}
    passed = output_ok and all(d["passed"] for m in metrics.values()
                               for d in m["differences"].values())
    return {"passed": passed, "output_passed": output_ok,
            "max_absolute_difference": float(errors.max()),
            "outputs": expected.size, "metrics": metrics}


def verify_inventory(work: Path) -> dict:
    registration = read(work / "registration.json")
    for name, digest in registration["files"].items():
        require(sha(work / name) == digest, f"frozen input changed: {name}")
    return registration


def initialize(work: Path, source: Path, partition: Path) -> None:
    work.mkdir(parents=True, exist_ok=False)
    runtime = work / "runtime"
    runtime.mkdir()
    # Copy the module graph, not installed libraries or any NT weights.
    sources = list(ROOT.glob("*.py")) + [HERE / "evaluate.py", HERE / "contract.json"]
    for directory in ("cas12a", "core", "xai"):
        sources.extend((ROOT / directory).rglob("*.py"))
    for path in sources:
        destination = runtime / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    manifest = read(ROOT / "models/non_nt/manifest.json")
    for name, identity in manifest["files"].items():
        require(sha(ROOT / name) == identity["sha256"], f"preserved asset changed: {name}")
        destination = runtime / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    shutil.copyfile(ROOT / "models/non_nt/manifest.json", runtime / "models/non_nt/manifest.json")
    inputs = work / "inputs"
    inputs.mkdir()
    shutil.copyfile(source, inputs / "cas12a.xlsx")
    shutil.copyfile(partition, inputs / "partition.json")
    contract = read(runtime / "docs/model-evaluation/contract.json")
    require(sha(inputs / "cas12a.xlsx") == contract["cas12a"]["source_sha256"], "wrong Cas12a source")
    versions = {name: version(name) for name in
                ("numpy", "pandas", "scipy", "keras", "torch", "tensorflow", "h5py", "openpyxl")}
    files = {p.relative_to(work).as_posix(): sha(p) for p in work.rglob("*") if p.is_file()}
    write(work / "registration.json", {
        "schema": "dashboard-v3-test-registration-v1", "issue": 202,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": sha(runtime / "docs/model-evaluation/contract.json"),
        "base_commit": contract["base_commit"], "python": sys.version,
        "platform": platform.platform(), "libraries": versions, "files": files,
        "acquisition": {"cas12a_source": str(source), "cas12a_partition": str(partition)},
        "historical_limitations": [
            "Cas12a training source attestation is retrospective; see preserved provenance.json.",
            "SpCas9 historical evaluation records a dirty source snapshot digest; its full contemporaneous source snapshot is not in the adopted vault. Saved split, labels, predictions and both checkpoint files are retained and replayed here.",
            "No new training, model selection, split change, or NT comparison is performed."
        ]})


def prepare(work: Path, contract: dict) -> None:
    """Use the preserved training source to verify and encode the Cas12a test."""
    asset = ROOT / contract["cas12a"]["directory"]
    original = work / "original"
    with zipfile.ZipFile(asset / "source_snapshot.zip") as archive:
        archive.extractall(original)
    sys.path.insert(0, str(original))
    # These modules exist only in the checksum-pinned historical source archive.
    encoder = importlib.import_module("cas12a.cnn_manifest")
    digest = importlib.import_module("cas12a.nt_data").digest

    manifest = read(work / "inputs/partition.json")
    require(digest(manifest) == contract["cas12a"]["partition_manifest_sha256"], "wrong partition manifest")
    config = read(asset / "configuration.json")
    require(manifest["partition_sha256"] == config["partition"]["partition_sha256"]
            == contract["cas12a"]["partition_sha256"], "training/test partition mismatch")
    for name in ("train", "validation"):
        require(manifest["partitions"][name] == config["membership"][name + "_source_rows"], f"{name} membership changed")
    require(manifest["partitions"]["test"] == config["membership"]["held_out"]["source_rows"], "test membership changed")
    require(config["membership"]["held_out_used_for_fitting_or_selection"] is False, "test used for selection")
    library = encoder.load_library(work / "inputs/cas12a.xlsx", manifest)
    by_identity = encoder.verify_against_library(manifest, library)
    members = manifest["partitions"]["test"]
    sequence, features, targets = encoder.partition_arrays(by_identity, members)
    require(len(members) == contract["cas12a"]["rows"] and np.isfinite(targets).all(), "invalid Cas12a test")
    np.savez_compressed(work / "cas12a-inputs.npz", sequence=sequence, features=features,
                        targets=targets[:, None], source_rows=np.array(members),
                        windows=np.array([by_identity[i]["sequence"] for i in members]))
    write(work / "preprocessing.json", {
        "cas12a": {"rows": len(members), "source_rows": library["source_row_count"],
                    "usable_rows": len(library["rows"]), "excluded": library["excluded"],
                    "partition_counts": manifest["counts"],
                    "test_disjoint_from_training_and_validation": True,
                    "source_and_table_and_row_hashes_verified": True}})
    import pandas as pd

    directory = ROOT / contract["spcas9"]["directory"]
    frame = pd.read_csv(directory / "published_split/test.csv")
    saved = pd.read_csv(directory / "test_eval/predictions.csv")
    require(len(frame) == len(saved) == contract["spcas9"]["rows"], "wrong SpCas9 test size")
    require(frame.row_index.is_unique and frame.row_index.equals(saved.row_index), "SpCas9 row order mismatch")
    require(frame.sequence.equals(saved.sequence) and (frame.split == "test").all(), "SpCas9 sequence/split mismatch")
    labels = frame[contract["spcas9"]["heads"]].to_numpy(dtype=np.float64) / 100
    archived_targets = saved[[h + "_true" for h in contract["spcas9"]["heads"]]].to_numpy()
    np.testing.assert_allclose(labels, archived_targets, rtol=0, atol=1e-15, equal_nan=True)
    # Original one-hot convention, independent of deployed one_hot().
    sequence = np.zeros((len(frame), 30, 4), dtype=np.float32)
    for i, window in enumerate(frame.sequence):
        require(len(window) == 30, "SpCas9 window length changed")
        for j, base in enumerate(window):
            if base in "ACGT":
                sequence[i, j, "ACGT".index(base)] = 1
    np.savez_compressed(work / "spcas9-inputs.npz", sequence=sequence, targets=labels,
                        source_rows=frame.row_index.to_numpy(), windows=frame.sequence.to_numpy(dtype=str),
                        archived=saved[[h + "_pred" for h in contract["spcas9"]["heads"]]].to_numpy())
    write(work / "prepared.json", {name: sha(work / name) for name in
                                  ("spcas9-inputs.npz", "cas12a-inputs.npz", "preprocessing.json")})


def verify_prepared(work: Path) -> None:
    for name, digest in read(work / "prepared.json").items():
        require(sha(work / name) == digest, f"prepared input changed: {name}")


def reference(work: Path, contract: dict) -> None:
    verify_prepared(work)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import keras
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(contract["inference"]["tensorflow_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    require(keras.backend.backend() == "tensorflow", "wrong reference backend")
    outputs = {}
    for key in ("spcas9", "cas12a"):
        policy = contract[key]
        with np.load(work / f"{key}-inputs.npz") as inputs:
            arguments = [inputs["sequence"]]
            if key == "cas12a":
                arguments.append(inputs["features"])
            for kind in (("original", "export") if key == "spcas9" else ("original",)):
                model_path = ROOT / policy["original" if kind == "original" else "deployed"]
                model: Any = keras.saving.load_model(model_path, compile=False)
                # Select by layer name: original SpCas9 has a dict output container.
                ordered = keras.Model(model.inputs, [model.get_layer(n).output for n in policy["layers"]])
                batches = []
                for start in range(0, policy["rows"], contract["inference"]["batch_size"]):
                    arrays = [a[start:start + contract["inference"]["batch_size"]] for a in arguments]
                    result = ordered(arrays[0] if len(arrays) == 1 else arrays, training=False)
                    result = result if isinstance(result, (tuple, list)) else [result]
                    batches.append(np.column_stack([np.asarray(a).reshape(-1) for a in result]))
                outputs[f"{key}_{kind}"] = np.concatenate(batches)
                keras.backend.clear_session()
    np.savez_compressed(work / "reference.npz", **outputs)
    write(work / "reference-runtime.json", {"backend": keras.backend.backend(),
          "keras": keras.__version__, "tensorflow": tf.__version__, "dtype": "float32"})


def deployed(work: Path, contract: dict) -> None:
    verify_prepared(work)
    sys.path.insert(0, str(ROOT))
    from non_nt_runtime import NonNTRuntime, cas12a_derive, one_hot, runtime_versions

    outputs, preprocessing = {}, {}
    for key in ("spcas9", "cas12a"):
        with np.load(work / f"{key}-inputs.npz") as inputs:
            arrays = [np.stack([one_hot(w) for w in inputs["windows"]])]
            if key == "cas12a":
                arrays.append(np.stack([cas12a_derive(w) for w in inputs["windows"]]))
            for name, array in zip(("sequence", "features"), arrays, strict=False):
                np.testing.assert_array_equal(array, inputs[name])
            preprocessing[key] = {"rows": len(arrays[0]), "all_inputs_exact": True}
            runtime = NonNTRuntime.load(key)
            require(runtime.checkpoint_sha256 == contract[key]["deployed_sha256"], "deployed SHA mismatch")
            require(runtime.head_names == contract[key]["layers"], "deployed head order mismatch")
            outputs[key] = runtime.forward_all(arrays, batch_size=contract["inference"]["batch_size"])
            del runtime
    np.savez_compressed(work / "deployed.npz", **outputs)
    write(work / "deployed-runtime.json", {**runtime_versions(), "preprocessing": preprocessing})


def assemble(work: Path, contract: dict) -> dict:
    registration = verify_inventory(work)
    verify_prepared(work)
    comparisons = {}
    historical = read(ROOT / contract["spcas9"]["directory"] / "test_eval/results.json")
    export = read(ROOT / contract["spcas9"]["directory"] / "export-manifest.json")
    require(historical["predictions"]["checkpoint_sha256"] == export["source"]["sha256"]
            == contract["spcas9"]["original_sha256"], "historical/export source SHA differs")
    sp_directory = ROOT / contract["spcas9"]["directory"]
    training = read(sp_directory / "training/training_log.json")
    require(training["checkpoint"]["sha256"] == export["source"]["sha256"], "training source SHA differs")
    require(sha(sp_directory / "training/training_log.json") == historical["execution"]["training_log_sha256"], "training log differs")
    require(training["split"] == historical["predictions"]["split"], "historical training/test split differs")
    for name, digest in training["split"]["files"].items():
        require(sha(sp_directory / "published_split" / name) == digest, f"SpCas9 split changed: {name}")
    for key in ("spcas9", "cas12a"):
        for field in ("original", "deployed"):
            require(sha(ROOT / contract[key][field]) == contract[key][field + "_sha256"], "checkpoint SHA mismatch")
    inherited = {}
    with np.load(work / "reference.npz") as ref, np.load(work / "deployed.npz") as actual:
        for key in ("spcas9", "cas12a"):
            with np.load(work / f"{key}-inputs.npz") as inputs:
                require(actual[key].shape == inputs["targets"].shape, "missing test outputs")
                pairs = {"original_vs_deployed": (ref[f"{key}_original"], actual[key])}
                if key == "spcas9":
                    pairs.update(original_vs_export=(ref["spcas9_original"], ref["spcas9_export"]),
                                 archived_vs_original=(inputs["archived"], ref["spcas9_original"]))
                comparisons[key] = {}
                for label, (expected, observed) in pairs.items():
                    comparisons[key][label] = {
                        head: compare(inputs["targets"][:, i], expected[:, i], observed[:, i], contract["parity"])
                        for i, head in enumerate(contract[key]["heads"])}
                if key == "spcas9":
                    for i, head in enumerate(contract[key]["heads"]):
                        recomputed = performance(inputs["targets"][:, i], inputs["archived"][:, i])
                        recorded = historical["heads"][head]
                        require(recomputed["n"] == recorded["n"], "historical head coverage differs")
                        for metric in contract["metrics"]["names"]:
                            require(abs(recomputed[metric] - recorded[metric]) < 1e-12, "historical metric replay differs")
                        inherited[head] = {"recorded": {n: recorded[n] for n in ("n", "pearson", "spearman", "mae", "mse")},
                                           "recomputed_from_preserved_csv": recomputed}
    checks = [v["passed"] for model in comparisons.values() for pair in model.values() for v in pair.values()]
    report = {"issue": 202, "status": "pass" if all(checks) else "fail",
              "counts": {"passed": sum(checks), "failed": len(checks) - sum(checks), "skipped": 0},
              "contract_sha256": registration["contract_sha256"],
              "registration_sha256": sha(work / "registration.json"),
              "handoff": {
                  "spcas9": {"source_sha256": export["source"]["sha256"],
                              "deployed_sha256": export["artifact"]["sha256"],
                              "split": training["split"], "selection": training["stopping"],
                              "historical_execution": historical["execution"],
                              "head_mapping": export["heads"]},
                  "cas12a": {"checkpoint": read(ROOT / contract["cas12a"]["directory"] / "checkpoint.json"),
                              "validation_evidence_kind": "checkpoint selection only, not independent test",
                              "new_evaluation_kind": "held-out test membership fixed before checkpoint selection"}},
              "historical_spcas9": inherited, "new_full_test_comparisons": comparisons,
              "preprocessing": read(work / "preprocessing.json"),
              "reference_runtime": read(work / "reference-runtime.json"),
              "deployed_runtime": read(work / "deployed-runtime.json"),
              "limitations": registration["historical_limitations"],
              "files": {name: sha(work / name) for name in
                        ("spcas9-inputs.npz", "cas12a-inputs.npz", "reference.npz", "deployed.npz",
                         "prepared.json", "preprocessing.json", "reference-runtime.json", "deployed-runtime.json")}}
    return report


def audit(work: Path) -> dict:
    """Independently recompute every reported metric/verdict from retained arrays."""
    # assemble resolves checkpoint and inherited evidence relative to this copied source.
    report = read(work / "report.json")
    for name, digest in report["files"].items():
        require(sha(work / name) == digest, f"raw evidence changed: {name}")
    rebuilt = assemble(work, read(HERE / "contract.json"))
    require(report == rebuilt, "report does not match retained evidence")
    return rebuilt


def run(work: Path, source: Path, partition: Path) -> int:
    initialize(work, source, partition)
    runner = work / "runtime/docs/model-evaluation/evaluate.py"
    stages = []
    for stage in ("prepare", "reference", "deployed", "report", "audit"):
        command = [sys.executable, "-X", "utf8", str(runner), stage, "--work", str(work)]
        started = time.monotonic()
        with (work / f"{stage}.log").open("wb") as log:
            completed = subprocess.run(command, cwd=work / "runtime", stdout=log, stderr=subprocess.STDOUT, check=False)
        stages.append({"command": command, "exit_code": completed.returncode,
                       "seconds": time.monotonic() - started})
        write(work / "execution.json", stages)
        print(f"{stage}: exit {completed.returncode}", flush=True)
        if completed.returncode:
            print((work / f"{stage}.log").read_text(encoding="utf-8"), flush=True)
            return completed.returncode
    return 0 if read(work / "report.json")["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "prepare", "reference", "deployed", "report", "audit"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cas12a-source", type=Path)
    parser.add_argument("--cas12a-partition", type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    if args.stage == "run":
        if args.cas12a_source is None or args.cas12a_partition is None:
            parser.error("run requires --cas12a-source and --cas12a-partition")
        return run(work, args.cas12a_source.resolve(), args.cas12a_partition.resolve())
    require(ROOT == work / "runtime", "Use the bundle's runtime/docs/model-evaluation/evaluate.py for retained stages")
    verify_inventory(work)
    contract = read(HERE / "contract.json")
    if args.stage == "report":
        write(work / "report.json", assemble(work, contract))
    elif args.stage == "audit":
        print(json.dumps(audit(work)["counts"]))
    else:
        {"prepare": prepare, "reference": reference, "deployed": deployed}[args.stage](work, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
