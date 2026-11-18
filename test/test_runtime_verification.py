"""#206: fixture evidence through the existing verification CLIs/report/exit code.

These are harness regressions, not official two-model measurements (#212/#213).
"""

import json
import hashlib
import importlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def cli(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "docs/runtime" / script), *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )


def test_assembler_reports_missing_evidence_as_failure(tmp_path):
    output = tmp_path / "report.json"
    result = cli("assemble_verification.py", "--work", tmp_path / "missing",
                 "--output", output)
    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"]["overall"] == "fail"
    assert report["open_failures"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_receipt(work, stage):
    """Build raw-result fixtures, never adopt historical/real model evidence."""
    path = work / f"{stage}.receipt.json"
    receipt = load(path)
    receipt["artifacts"] = {name: sha(work / name) for name in receipt["artifacts"]}
    write(path, receipt)
    if stage == "reference":
        for dependent in ("verify", "diagnose"):
            path = work / f"{dependent}.receipt.json"
            receipt = load(path)
            receipt["dependencies"][stage] = sha(work / "reference.receipt.json")
            write(path, receipt)


@pytest.fixture(scope="module")
def complete_fixture(tmp_path_factory):
    """Synthetic complete evidence with independently assigned passing values.

    init exercises the public premeasurement freeze against pinned assets, but
    every measurement below is a fixture, not official model validation.
    """
    work = tmp_path_factory.mktemp("runtime-fixture") / "bundle"
    result = cli("verify_contract.py", "init", "--work", work, "--run-id", "fixture-206")
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = load(work / "manifest.json")
    contract = load(work / "snapshot/contract.json")
    protocol = load(work / "snapshot/protocol.json")
    docs = {
        "reference": {"backend": "tensorflow", "keras": "3.14.1", "tensorflow": "2.21.0",
                      "models": {key: {"layer_names": layers} for key, layers in manifest["topology"].items()}},
        "ram": {key: {**{stage: 1000000 for stage in (
            "interpreter", "runtime_import", "model_loaded", "after_forward", "after_ig", "peak_working_set")},
            "all_finite": True, "exit_code": 0, "held_models": [key], "command": ["fixture-ram", key]}
            for key in ("spcas9", "cas12a")},
        "offline": {"run": {"status": "pass", "network_blocked": True,
                              "versions": {**contract["runtime"]["versions"], "backend": "torch", "torch_threads": "4"},
                              "tensorflow_importable": False, "tensorflow_in_sys_modules_after_load": False,
                              "models": {"spcas9": {"n_outputs": 9, "all_finite": True, "gradient_abs_sum": [1.0]},
                                         "cas12a": {"n_outputs": 1, "all_finite": True, "gradient_abs_sum": [1.0, 1.0]}}}},
        "investigation": {"investigations": [
            {"id": name, "state": "pass", "planned": definition["planned"],
            "completed": definition["planned"], "raw_arrays": definition.get("required_arrays", [name])}
            for name, definition in protocol["investigations"].items()]},
    }
    for stage in ("verify", "diagnose"):
        names = manifest["expected_checks"][stage]
        docs[stage] = {
            "checks": [{"check": name, "passed": True, "measured": 0} for name in names],
            "n_checks": len(names), "n_failed": 0, "status": "pass",
            "runtime": {**contract["runtime"]["versions"], "backend": "torch", "torch_threads": "4"},
        }
    docs["verify"]["contract_sha256"] = sha(work / "snapshot/contract.json")
    docs["verify"]["reference_runtime"] = {"keras": "3.14.1", "tensorflow": "2.21.0", "backend": "tensorflow"}
    docs["verify"]["measurements"] = {}
    for key, model_key in {"spcas9": "spcas9_non_nt", "cas12a": "cas12a_non_nt"}.items():
        spec = contract["models"][model_key]
        heads = [f"head{h}" for h in range(spec["n_outputs"])]
        heads[spec["verified_head"]["index"]] = spec["verified_head"]["name"]
        docs["verify"]["measurements"][key] = {
            "verified_head": spec["verified_head"], "checkpoint_sha256": spec["sha256"],
            "input_names": [i["name"] for i in spec["inputs"]], "head_names": heads,
            "cases": {c["id"]: {"window": c["window"]} for c in contract["cases"][key]},
        }
        docs["reference"]["models"][key].update(
            verified_head=spec["verified_head"], checkpoint_sha256=spec["sha256"])
    for stage, files in manifest["stages"].items():
        identity = {"run_id": manifest["run_id"], "manifest_sha256": sha(work / "manifest.json"),
                    "stage": stage, "backend": "tensorflow" if stage == "reference" else "torch"}
        docs[stage]["identity"] = identity
        if stage == "offline":
            # Synthetic recovery provenance for assembler regressions only.
            restored = tmp_path_factory.getbasetemp() / "synthetic-restored"
            from_files = {}
            for name, item in manifest["sources"].items():
                relative = ("payload/" + name.split("/", 1)[1] if name.startswith(("code/", "assets/"))
                            else "payload/recovery/" + name)
                from_files[relative] = item["sha256"]
            from_files["payload/recovery/inputs.npz"] = manifest["inputs_sha256"]
            wheels = {name: {"version": version, "file": f"wheels/{name}.whl"}
                      for name, version in manifest["offline_dependencies"].items()}
            from_files.update({wheel["file"]: "0" * 64 for wheel in wheels.values()})
            docs[stage].update(command=["fixture-runner"], exit_code=0, recovery={
                "bundle_manifest_sha256": sha(work / "manifest.json"),
                "restored_root": str(restored), "files": from_files, "verified_files": from_files,
                "wheels": wheels, "installed_files": {"dependencies/fixture.py": "0" * 64},
                "imported_dependency_files": {"dependencies/fixture.py": "0" * 64},
                "dependency_checksum_scope": "synthetic fixture",
                "install": {"command": ["fixture-install"], "exit_code": 0}})
            docs[stage]["run"].update(
                tensorflow_import_attempt="ModuleNotFoundError", external_prerequisites={"python": "fixture"},
                network_probes=[{"kind": kind, "blocked": True} for kind in ("tcp", "dns", "reverse_dns", "udp")],
                network_scope="synthetic fixture",
                held_models=["spcas9", "cas12a"], unexpected_module_origins={},
                rss={stage: 1000000 for stage in ("interpreter", "runtime_import", "spcas9_model_loaded",
                     "spcas9_after_gradients", "cas12a_model_loaded", "cas12a_after_gradients", "peak_working_set")},
                module_origins={name: str(restored / (name + ".py")) for name in (
                    "non_nt_runtime", "model_source", "predict", "numpy", "keras", "torch", "h5py", "psutil")},
                path_guard={"original_roots": [str(ROOT)], "violations": [], "scope": "synthetic fixture",
                            "probes": [{"path": item["source"], "blocked": True}
                                       for item in manifest["sources"].values()]})
            for key in ("spcas9", "cas12a"):
                spec = contract["models"][key + "_non_nt"]
                docs[stage]["run"]["models"][key].update(checkpoint=str(restored / "payload" / spec["path"]),
                    sha256=spec["sha256"], head=spec["verified_head"], window=contract["cases"][key][0]["window"])
        write(work / files[0], docs[stage])
        if len(files) > 1:
            arrays = {name: np.ones(shape or [1], dtype=np.float64)
                      for name, shape in manifest["expected_arrays"].get(stage, {}).items()}
            if stage == "investigation":
                arrays = {name: np.full(protocol["investigations"][row["id"]].get("raw_shapes", {}).get(
                    name.rsplit("/", 1)[-1], [1]), 0.125)
                    for row in docs[stage]["investigations"] for name in row["raw_arrays"]}
            np.savez_compressed(work / files[1], **arrays)
        write(work / f"{stage}.receipt.json", {
            "identity": identity, "command": ["fixture-producer", stage],
            "environment": {"executable": "fixture-only"}, "exit_code": 0,
            "started_at": "2026-11-18T00:00:00Z", "ended_at": "2026-11-18T00:00:01Z",
            "dependencies": {}, "artifacts": {name: sha(work / name) for name in files},
        })
    for stage in ("verify", "diagnose"):
        path = work / f"{stage}.receipt.json"
        receipt = load(path)
        receipt["dependencies"] = {"reference": sha(work / "reference.receipt.json")}
        write(path, receipt)
    write(work / "journal.json", {"state": "ready", "completed": list(manifest["stages"])})
    return work


@pytest.fixture
def bundle(tmp_path, complete_fixture):
    work = tmp_path / "bundle"
    shutil.copytree(complete_fixture, work)
    return work


def assemble(work, tmp_path):
    output = tmp_path / "report.json"
    result = cli("assemble_verification.py", "--work", work, "--output", output)
    assert output.exists(), result.stdout + result.stderr
    return result, load(output)


def test_complete_bundle_can_be_reassembled_without_measurement(bundle, tmp_path):
    before = {p.relative_to(bundle).as_posix(): sha(p) for p in bundle.rglob("*") if p.is_file()}
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 0, report
    assert report["status"]["overall"] == "pass"
    again, duplicate = assemble(bundle, tmp_path)
    assert again.returncode == 0
    assert duplicate == report
    assert {p.relative_to(bundle).as_posix(): sha(p) for p in bundle.rglob("*") if p.is_file()} == before


def test_offline_rejects_original_worktree_staging(bundle, tmp_path):
    """A finite run with TF absent is insufficient independent recovery evidence."""
    path = bundle / "staged.json"
    document = load(path)
    document.pop("recovery", None)
    write(path, document)
    refresh_receipt(bundle, "offline")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert report["status"]["overall"] == "fail"
    assert report["claims"]["offline_tensorflow_free_run"]["status"] == "fail"


@pytest.mark.parametrize("fault", ["mixed_preparation", "checksum", "module_origin", "model_origin",
    "held_model", "original_access", "missing_probe", "unblocked_probe", "dependencies",
    "install_exit", "runner_exit", "network_probe", "tf_import", "rss", "model_checksum", "wheel_missing", "head", "reverse_dns"])
def test_offline_provenance_failures_reach_aggregate_exit(bundle, tmp_path, fault):
    path = bundle / "staged.json"
    document = load(path)
    recovery, run = document["recovery"], document["run"]
    if fault == "mixed_preparation":
        recovery["bundle_manifest_sha256"] = "0" * 64
    elif fault == "checksum":
        recovery["files"]["payload/non_nt_runtime.py"] = "0" * 64
    elif fault == "module_origin":
        run["module_origins"]["non_nt_runtime"] = str(ROOT / "non_nt_runtime.py")
    elif fault == "model_origin":
        run["models"]["cas12a"]["checkpoint"] = str(ROOT / "models/non_nt/model.keras")
    elif fault == "held_model":
        run["held_models"] = ["cas12a"]
    elif fault == "original_access":
        run["path_guard"]["violations"] = [{"event": "open", "path": str(ROOT / "predict.py")}]
    elif fault == "missing_probe":
        run["path_guard"]["probes"].pop()
    elif fault == "unblocked_probe":
        run["path_guard"]["probes"][0]["blocked"] = False
    elif fault == "dependencies":
        recovery["installed_files"] = {}
    elif fault == "install_exit":
        recovery["install"]["exit_code"] = 1
    elif fault == "runner_exit":
        document["exit_code"] = 1
    elif fault == "network_probe":
        run["network_probes"] = []
    elif fault == "tf_import":
        run["tensorflow_import_attempt"] = "succeeded"
    elif fault == "rss":
        run.pop("rss", None)
    elif fault == "model_checksum":
        run["models"]["cas12a"]["sha256"] = "0" * 64
    elif fault == "wheel_missing":
        recovery["wheels"].pop("h5py")
    elif fault == "head":
        run["models"]["spcas9"]["head"]["index"] = 0
    elif fault == "reverse_dns":
        run["network_probes"] = [{"kind": kind, "blocked": True} for kind in ("tcp", "dns", "udp")]
    write(path, document)
    refresh_receipt(bundle, "offline")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1, report
    assert report["status"]["overall"] == "fail"
    assert report["claims"]["offline_tensorflow_free_run"]["status"] == "fail"


@pytest.mark.parametrize("state,expected,code", [
    ("pass", "pass", 0), ("unverified", "incomplete", 1),
    ("unavailable", "incomplete", 1), ("time_limited", "incomplete", 1),
    ("unresolved", "completed_with_findings", 0),
    ("no_eligible_candidate", "completed_with_findings", 0),
    ("not_applicable", "pass", 0),
])
def test_q17_states_at_report_boundary(bundle, tmp_path, state, expected, code):
    path = bundle / "investigations.json"
    document = load(path)
    row = next(r for r in document["investigations"] if r["id"] == "float64_spcas9")
    row.update(state=state, reason="prespecified fixture condition")
    if state in ("unverified", "unavailable", "time_limited", "not_applicable"):
        row.update(completed=0, raw_arrays=[])
    if state == "time_limited":
        row["termination"] = "planned_budget"
    write(path, document)
    refresh_receipt(bundle, "investigation")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == code, report
    assert report["status"]["overall"] == expected, report
    assert report["investigation_counts"]["float64_spcas9"]["completed"] == row["completed"]


@pytest.mark.parametrize("failure", ["offline", "missing_stage", "missing_check", "failed_check",
                                      "tamper", "mixed", "partial", "interrupted"])
def test_required_failure_always_dominates_findings(bundle, tmp_path, failure):
    path = bundle / "investigations.json"
    doc = load(path)
    doc["investigations"][0]["state"] = "no_eligible_candidate"
    write(path, doc)
    refresh_receipt(bundle, "investigation")
    if failure == "offline":
        path = bundle / "staged.json"
        doc = load(path)
        doc["run"]["status"] = "fail"
        write(path, doc)
        refresh_receipt(bundle, "offline")
    elif failure == "missing_stage":
        (bundle / "ram.receipt.json").unlink()
    elif failure in ("failed_check", "missing_check"):
        path = bundle / "verification-core.json"
        doc = load(path)
        if failure == "missing_check":
            doc["checks"].pop()
            doc["n_checks"] -= 1
        else:
            doc["checks"][0]["passed"] = False
            doc.update(n_failed=1, status="fail")
            receipt = load(bundle / "verify.receipt.json")
            receipt["exit_code"] = 1
            write(bundle / "verify.receipt.json", receipt)
        write(path, doc)
        refresh_receipt(bundle, "verify")
    elif failure == "tamper":
        with (bundle / "core-raw.npz").open("ab") as stream:
            stream.write(b"changed after measurement")
    elif failure == "mixed":
        path = bundle / "diagnose.json"
        doc = load(path)
        doc["identity"]["run_id"] = "another-run"
        write(path, doc)
        refresh_receipt(bundle, "diagnose")
    elif failure == "partial":
        path = bundle / "offline.receipt.json"
        receipt = load(path)
        receipt["artifacts"] = {}
        write(path, receipt)
    else:
        write(bundle / "journal.json", {"state": "running", "completed": []})
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1, report
    assert report["status"]["overall"] == "fail"


@pytest.mark.parametrize("key", ["python", "keras", "torch", "numpy", "shap", "h5py"])
def test_missing_registered_version_fails_even_with_passing_check_list(bundle, tmp_path, key):
    path = bundle / "verification-core.json"
    doc = load(path)
    del doc["runtime"][key]
    write(path, doc)
    refresh_receipt(bundle, "verify")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert any(key in error for error in report["evidence_errors"])


@pytest.mark.parametrize("kind", ["relative", "scaled", "missing"])
def test_nonrepresentative_layer_violation_cannot_hide(bundle, tmp_path, kind):
    manifest = load(bundle / "manifest.json")
    first, second = manifest["topology"]["spcas9"][:2]
    reference_path = bundle / "reference.npz"
    target_path = bundle / "core-raw.npz"
    with np.load(reference_path) as source:
        reference = dict(source)
    with np.load(target_path) as source:
        target = dict(source)
    # Largest absolute error passes due to its scale. The smaller-error layer fails.
    a, b = (f"spcas9/S1/layer/{layer}" for layer in (first, second))
    reference[a], target[a] = np.array([1000.0]), np.array([1000.01])
    if kind == "relative":
        reference[b], target[b] = np.array([1e-5]), np.array([1.1e-5])
    elif kind == "scaled":
        reference[b], target[b] = np.array([1.0]), np.array([1.0002])
    else:
        del target[b]
    np.savez_compressed(reference_path, **reference)
    np.savez_compressed(target_path, **target)
    refresh_receipt(bundle, "reference")
    refresh_receipt(bundle, "verify")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert any(b in error for error in report["evidence_errors"])


@pytest.mark.parametrize("stage,name", [
    ("offline", "spcas9/output"),
    ("offline", "cas12a/gradient/1"),
    ("offline", "cas12a/gradient_output"),
    ("verify", "spcas9/S1/shap/seed1/values"),
    ("verify", "spcas9/S1/shap/seed2/terms"),
    ("verify", "cas12a/C2/shap/repeat/values"),
    ("verify", "cas12a/C2/shap/finer/terms"),
    ("verify", "cas12a/C2/ig/finer/input_totals"),
    ("verify", "cas12a/C2/grad/alpha0.5/1"),
    ("diagnose", "cas12a/C3/directional/terms"),
    ("investigation", "float64_cas12a"),
])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_raw_values_fail_in_every_run(bundle, tmp_path, stage, name, value):
    manifest = load(bundle / "manifest.json")
    path = bundle / manifest["stages"][stage][1]
    with np.load(path) as source:
        arrays = dict(source)
    arrays[name].flat[-1] = value
    np.savez_compressed(path, **arrays)
    refresh_receipt(bundle, stage)
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert any(name in error for error in report["evidence_errors"])


def test_measurement_cli_refuses_resume_without_touching_evidence(bundle):
    before = sha(bundle / "verification-core.json")
    result = cli("verify_contract.py", "verify", "--work", bundle)
    assert result.returncode == 1
    assert "cannot resume" in result.stdout
    assert sha(bundle / "verification-core.json") == before


def test_missing_candidate_measurements_are_incomplete(bundle, tmp_path):
    path = bundle / "investigations.json"
    document = load(path)
    document["investigations"] = [r for r in document["investigations"] if r["id"] != "ig_spcas9"]
    write(path, document)
    refresh_receipt(bundle, "investigation")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert report["status"]["overall"] == "incomplete", report


def test_unexpected_budget_interruption_requires_new_bundle(bundle, tmp_path):
    path = bundle / "investigations.json"
    document = load(path)
    row = document["investigations"][0]
    row.update(state="time_limited", completed=1, reason="killed", termination="unexpected")
    write(path, document)
    refresh_receipt(bundle, "investigation")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert report["status"]["overall"] == "fail"


def test_init_cannot_adopt_a_previous_or_partial_bundle(bundle):
    result = cli("verify_contract.py", "init", "--work", bundle, "--run-id", "replacement")
    assert result.returncode == 1
    assert "empty bundle" in result.stdout


def test_changed_frozen_input_blocks_all_further_measurements(tmp_path):
    work = tmp_path / "new-run"
    extra = tmp_path / "registered-producer.py"
    extra.write_text("# registered before measurement\n", encoding="utf-8")
    result = cli("verify_contract.py", "init", "--work", work, "--run-id", "changed-input",
                 "--extra-code", extra)
    assert result.returncode == 0, result.stdout
    extra.write_text("# changed after init\n", encoding="utf-8")
    result = cli("verify_contract.py", "investigation", "--work", work)
    assert result.returncode == 1
    assert "full remeasurement" in result.stdout
    assert not (work / "investigations.json").exists()
    extra.write_text("# registered before measurement\n", encoding="utf-8")
    again = cli("verify_contract.py", "investigation", "--work", work)
    assert again.returncode == 1
    assert "cannot resume" in again.stdout


def test_unexpected_running_stage_refuses_another_stage(tmp_path):
    work = tmp_path / "interrupted"
    result = cli("verify_contract.py", "init", "--work", work, "--run-id", "interrupted")
    assert result.returncode == 0, result.stdout
    write(work / "journal.json", {"state": "running", "active": "verify", "completed": []})
    result = cli("verify_contract.py", "investigation", "--work", work)
    assert result.returncode == 1
    assert "cannot resume" in result.stdout
    assert not (work / "investigations.json").exists()


@pytest.mark.parametrize("fault", ["layer_relative", "seed2_nonfinite", "version_missing"])
def test_measurement_cli_records_failures_and_keeps_collecting(tmp_path, monkeypatch, fault):
    """Exercise the real CLI collection with synthetic runtime/XAI adapters."""
    monkeypatch.syspath_prepend(str(ROOT / "docs/runtime"))
    verify = importlib.import_module("verify_contract")
    import non_nt_runtime as rt
    from xai import explain

    work = tmp_path / "measurement-fixture"
    initialized = cli("verify_contract.py", "init", "--work", work, "--run-id", "measurement-fixture")
    assert initialized.returncode == 0, initialized.stdout
    manifest = load(work / "manifest.json")
    contract = load(work / "snapshot/contract.json")
    models = {"spcas9": "spcas9_non_nt", "cas12a": "cas12a_non_nt"}

    class SyntheticRuntime:
        def __init__(self, key):
            self.key = key
            self.spec = contract["models"][models[key]]
            self.input_shapes = [tuple(i["shape"]) for i in self.spec["inputs"]]
            self.n_outputs = self.spec["n_outputs"]
            self.head_names = [f"head{h}" for h in range(self.n_outputs)]
            self.head_names[self.spec["verified_head"]["index"]] = self.spec["verified_head"]["name"]

        def input_names(self):
            return [i["name"] for i in self.spec["inputs"]]

        def layer_names(self):
            return manifest["topology"][self.key]

        def ops_present(self):
            return self.spec["ops_to_confirm"]

        def weight_digests(self):
            return {}

        def forward(self, arrays, head=0, **kwargs):
            total = np.zeros(len(arrays[0])) + 1 + head * 0.01
            for array in arrays:
                weights = np.linspace(0.001, 0.01, np.asarray(array[0]).size)
                total += np.asarray(array).reshape(len(array), -1) @ weights
            return total

        def gradients(self, arrays, head=0):
            return self.forward(arrays, head), [np.broadcast_to(
                np.linspace(0.001, 0.01, np.asarray(a[0]).size).reshape(a[0].shape), a.shape).copy()
                for a in arrays]

        def layer_activations(self, arrays):
            values = {layer: np.array([1.0]) for layer in self.layer_names()}
            if fault == "layer_relative" and self.key == "spcas9":
                a, b = self.layer_names()[:2]
                values[a] = np.array([1000.01])
                values[b] = np.array([1.1e-5])
            return values

    versions = {**contract["runtime"]["versions"], "backend": "torch", "torch_threads": "4"}
    if fault == "version_missing":
        versions.pop("h5py")
    monkeypatch.setattr(rt, "runtime_versions", lambda: versions)
    monkeypatch.setattr(rt.NonNTRuntime, "load", SyntheticRuntime)

    def synthetic_shap(handle, window, *, seed=0, **kwargs):
        values = np.linspace(0.001, 0.01, len(window))
        if fault == "seed2_nonfinite" and seed == 2:
            values[-1] = float("nan")
        return explain.SHAPResult(values, 0.0, 1.0, 0.0)

    monkeypatch.setattr(explain, "kernel_shap", synthetic_shap)
    # Reference is an independent literal runtime adapter's known linear answer.
    reference = {}
    metadata = {"backend": "tensorflow", "keras": "3.14.1", "tensorflow": "2.21.0", "models": {}}
    frozen = dict(np.load(work / "snapshot/inputs.npz"))
    for key, model in models.items():
        runtime = SyntheticRuntime(key)
        metadata["models"][key] = {"weights": {}, "layer_names": runtime.layer_names()}
        for case in contract["cases"][key]:
            prefix = f"{key}/{case['id']}"
            inputs = [frozen[f"{prefix}/input/{i}"] for i in range(len(runtime.input_shapes))]
            reference[prefix + "/outputs"] = np.array([
                runtime.forward(inputs, head=h)[0] for h in range(runtime.n_outputs)])
            for layer in runtime.layer_names():
                reference[f"{prefix}/layer/{layer}"] = np.array([1.0])
            if fault == "layer_relative" and key == "spcas9":
                a, b = runtime.layer_names()[:2]
                reference[f"{prefix}/layer/{a}"] = np.array([1000.0])
                reference[f"{prefix}/layer/{b}"] = np.array([1e-5])
    identity = {"run_id": "measurement-fixture", "manifest_sha256": sha(work / "manifest.json"),
                "stage": "reference", "backend": "tensorflow"}
    metadata["identity"] = identity
    write(work / "reference.json", metadata)
    np.savez_compressed(work / "reference.npz", **reference)
    write(work / "reference.receipt.json", {
        "identity": identity, "command": ["synthetic-reference"], "environment": {"kind": "fixture"},
        "exit_code": 0, "dependencies": {}, "artifacts": {
            name: sha(work / name) for name in ("reference.json", "reference.npz")}})
    write(work / "journal.json", {"state": "ready", "completed": ["reference"]})
    monkeypatch.setattr(sys, "argv", ["verify_contract.py", "verify", "--work", str(work)])
    assert verify.main() == 1
    core = load(work / "verification-core.json")
    assert set(core["measurements"]) == {"spcas9", "cas12a"}
    assert len(core["measurements"]["cas12a"]["cases"]) == 4
    failed_names = {r["check"] for r in core["checks"] if not r["passed"]}
    if fault == "layer_relative":
        layer = manifest["topology"]["spcas9"][1]
        assert f"spcas9/S1/layerwise/{layer}/rel" in failed_names
        assert "spcas9/layerwise_parity/rel" in failed_names
    elif fault == "seed2_nonfinite":
        assert "spcas9/S1/finiteness/shap_seed2" in failed_names
        assert "finiteness/raw" in failed_names
    else:
        assert "version/h5py" in failed_names
    assert load(work / "journal.json")["state"] == "ready"
    assert load(work / "verify.receipt.json")["exit_code"] == 1
    # Valid numeric failures are not interruption: the next stage can collect.
    monkeypatch.setattr(sys, "argv", ["verify_contract.py", "investigation", "--work", str(work)])
    assert verify.main() == 0
    assert (work / "investigation.receipt.json").exists()


def test_planned_budget_stop_preserves_results_and_counts(bundle, tmp_path):
    path = bundle / "investigations.json"
    doc = load(path)
    row = next(r for r in doc["investigations"] if r["id"] == "ig_spcas9")
    row.update(state="time_limited", completed=5, reason="prespecified budget exhausted",
               termination="planned_budget", raw_arrays=row["raw_arrays"][:10])
    write(path, doc)
    refresh_receipt(bundle, "investigation")
    checksum = sha(bundle / "investigation-raw.npz")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert report["status"]["overall"] == "incomplete"
    assert report["investigation_counts"]["ig_spcas9"] == {"completed": 5, "incomplete": 11}
    replay, duplicate = assemble(bundle, tmp_path)
    assert replay.returncode == 1
    assert duplicate == report
    assert sha(bundle / "investigation-raw.npz") == checksum
    attempt = cli("verify_contract.py", "investigation", "--work", bundle)
    assert attempt.returncode == 1
    assert "cannot resume" in attempt.stdout


@pytest.mark.parametrize("field", ["head", "window", "checkpoint", "raw_case", "raw_seed", "candidate_seed"])
def test_identity_and_registered_investigation_inventory_mismatch_fails(bundle, tmp_path, field):
    if field in ("raw_case", "raw_seed"):
        path = bundle / "core-raw.npz"
        with np.load(path) as source:
            arrays = dict(source)
        name = "spcas9/S5/outputs" if field == "raw_case" else "spcas9/S1/shap/seed99/values"
        arrays[name] = np.array([0.5])
        np.savez_compressed(path, **arrays)
        refresh_receipt(bundle, "verify")
    elif field == "candidate_seed":
        path = bundle / "investigations.json"
        doc = load(path)
        row = next(r for r in doc["investigations"] if r["id"] == "shap_confirmation_spcas9")
        row["raw_arrays"] = [n for n in row["raw_arrays"] if "background2/subset2" not in n]
        write(path, doc)
        refresh_receipt(bundle, "investigation")
    else:
        path = bundle / "verification-core.json"
        doc = load(path)
        model = doc["measurements"]["spcas9"]
        if field == "head":
            model["verified_head"]["index"] = 0
        elif field == "checkpoint":
            model["checkpoint_sha256"] = "different-model"
        else:
            model["cases"]["S1"]["window"] = "A" * 30
        write(path, doc)
        refresh_receipt(bundle, "verify")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert report["status"]["overall"] == "fail"


@pytest.mark.parametrize("stage", ["ram", "offline"])
def test_missing_required_stage_measurements_cannot_pass(bundle, tmp_path, stage):
    filename = "ram.json" if stage == "ram" else "staged.json"
    path = bundle / filename
    doc = load(path)
    if stage == "ram":
        # A peak alone does not account for the import/load/forward/IG stages.
        doc["spcas9"] = {"peak_working_set": 1000000}
    else:
        doc["run"] = {"status": "pass"}
    write(path, doc)
    refresh_receipt(bundle, stage)
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert report["status"]["overall"] == "fail"


@pytest.mark.parametrize("fault", ["identity", "termination"])
def test_registered_producer_cannot_restamp_another_runs_result(tmp_path, fault):
    work = tmp_path / "producer-run"
    producer = tmp_path / "producer.py"
    payload = ({"identity": {"run_id": "old-run"}} if fault == "identity" else {
        "investigations": [{"state": "unverified", "termination": "unexpected"}]})
    producer.write_text(
        "import json, sys\nfrom pathlib import Path\nimport numpy as np\n"
        "work = Path(sys.argv[1])\n"
        f"(work / 'investigations.json').write_text(json.dumps({payload!r}))\n"
        "np.savez_compressed(work / 'investigation-raw.npz')\n", encoding="utf-8")
    protocol = load(ROOT / "docs/runtime/protocol-206.json")
    protocol["investigation_command"] = ["{python}", str(producer), "{work}"]
    protocol_path = tmp_path / "registered-protocol.json"
    write(protocol_path, protocol)
    initialized = cli("verify_contract.py", "init", "--work", work, "--run-id", "new-run",
                      "--protocol", protocol_path, "--extra-code", producer)
    assert initialized.returncode == 0, initialized.stdout
    result = cli("verify_contract.py", "investigation", "--work", work)
    assert result.returncode == 1
    if fault == "identity":
        assert "identity mismatch" in result.stdout.lower()
        assert load(work / "investigations.json")["identity"]["run_id"] == "old-run"
    else:
        assert "unexpected interruption" in result.stdout.lower()
    assert load(work / "journal.json")["state"] == "aborted"


@pytest.mark.parametrize("key", ["python", "keras", "torch", "numpy", "shap", "h5py"])
@pytest.mark.parametrize("fault", ["missing", "mismatch"])
def test_offline_target_must_report_all_six_frozen_versions(bundle, tmp_path, key, fault):
    path = bundle / "staged.json"
    doc = load(path)
    versions = {**load(bundle / "snapshot/contract.json")["runtime"]["versions"],
                "backend": "torch", "torch_threads": "4"}
    if fault == "missing":
        versions.pop(key)
    else:
        versions[key] = "0"
    doc["run"]["versions"] = versions
    write(path, doc)
    refresh_receipt(bundle, "offline")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert report["status"]["overall"] == "fail"
    assert any(key in error for error in report["evidence_errors"])


@pytest.mark.parametrize("state", ["pass", "unverified", "unresolved", "not_applicable"])
def test_unexpected_termination_cannot_be_hidden_by_investigation_state(bundle, tmp_path, state):
    path = bundle / "investigations.json"
    doc = load(path)
    row = next(r for r in doc["investigations"] if r["id"] == "float64_spcas9")
    row.update(state=state, termination="unexpected", reason="unexpected producer interruption")
    if state in ("unverified", "not_applicable"):
        row.update(completed=0, raw_arrays=[])
    write(path, doc)
    refresh_receipt(bundle, "investigation")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert report["status"]["overall"] == "fail"


@pytest.mark.parametrize("name", ["ig/spcas9/S1/steps32/attribution",
                                 "shap/spcas9/S1/ns64/bg4/background0/subset1/values",
                                 "confirmation/cas12a/C1/background2/subset2/terms"])
def test_completed_candidate_raw_output_shape_must_match_preregistered_case(bundle, tmp_path, name):
    path = bundle / "investigation-raw.npz"
    with np.load(path) as source:
        arrays = dict(source)
    arrays[name] = np.ones(99)
    np.savez_compressed(path, **arrays)
    refresh_receipt(bundle, "investigation")
    result, report = assemble(bundle, tmp_path)
    assert result.returncode == 1
    assert report["status"]["overall"] == "fail"
