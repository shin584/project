"""Immutable inputs and single-attempt stage receipts for runtime verification.

Checksums detect changed/mixed evidence; they are not digital signatures. Never
adopt an existing result by stamping it with today's code/contract identity.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile

from verification_policy import MODELS, expected_checks, expected_arrays, require_planned_termination

ROOT = Path(__file__).resolve().parents[2]
STAGES = {
    "reference": ("reference.json", "reference.npz"),
    "verify": ("verification-core.json", "core-raw.npz"),
    "diagnose": ("diagnose.json", "diagnose-raw.npz"),
    "ram": ("ram.json",),
    "offline": ("staged.json", "offline-raw.npz"),
    "investigation": ("investigations.json", "investigation-raw.npz"),
}
DEPENDENCIES = {"verify": ("reference",), "diagnose": ("reference",)}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    # Write once for evidence; atomic replacement is used only by the journal.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def environment() -> dict:
    return {"executable": sys.executable, "python": platform.python_version(),
            "platform": platform.platform(), "cwd": str(Path.cwd()),
            "variables": {key: os.environ.get(key) for key in (
                "KERAS_BACKEND", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")}}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact(work: Path, name: str) -> Path:
    path = (work / name).resolve()
    if not path.is_relative_to(work.resolve()) or path.is_symlink():
        raise ValueError(f"Evidence path escapes bundle: {name}")
    return path


def local_sources(entries: list[Path]) -> list[Path]:
    """Retain local imports too, including modules introduced by integration."""
    pending, found = list(entries), set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parts = path.relative_to(ROOT).parent.parts
                    package = parts[:len(parts) - node.level + 1]
                    module = ".".join((*package, module)).rstrip(".")
                modules += [module, *(f"{module}.{alias.name}" for alias in node.names)]
        for module in modules:
            for base in (ROOT, path.parent):
                location = base.joinpath(*module.split("."))
                candidate = location.with_suffix(".py")
                if candidate.is_file():
                    pending.append(candidate)
                for directory in (location, *location.parents):
                    if directory == ROOT or not directory.is_relative_to(ROOT):
                        break
                    initializer = directory / "__init__.py"
                    if initializer.is_file():
                        pending.append(initializer)
    return sorted(found)


def initialize(work: Path, run_id: str, contract_path: Path,
               protocol_path: Path, extra_code: list[Path] = ()) -> dict:
    """Freeze inputs before ANY stage, refusing old or partially populated dirs."""
    import numpy as np
    if work.exists() and any(work.iterdir()):
        raise ValueError("A new empty bundle is required; partial resume is forbidden")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    contract, protocol = read_json(contract_path), read_json(protocol_path)
    if protocol["frozen_contract_sha256"] != digest(contract_path):
        raise ValueError("Additive protocol does not match the frozen contract")
    work.mkdir(parents=True, exist_ok=True)
    snapshot = work / "snapshot"
    snapshot.mkdir()
    sources = {}

    def retain(source: Path, relative: str):
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[relative] = {"source": str(source.resolve()), "sha256": digest(target)}

    retain(contract_path, "contract.json")
    retain(protocol_path, "protocol.json")
    retain(ROOT / "docs/runtime/offline-requirements.txt", "offline-requirements.txt")
    # Include the local import closure, not the unrelated UI or NT model bytes.
    code = [ROOT / name for name in (
        "non_nt_runtime.py", "model_source.py", "predict.py", "pam_scanner.py",
        "cas12a/__init__.py", "cas12a/features.py", "xai/__init__.py",
        "xai/explain.py")]
    code += sorted((ROOT / "docs/runtime").glob("*.py"))
    for path in local_sources(code):
        retain(path, "code/" + path.relative_to(ROOT).as_posix())
    for index, path in enumerate(extra_code):
        retain(path, f"extra/{index}/{path.name}")
    manifest = ROOT / "models/non_nt/manifest.json"
    retain(manifest, "assets/models/non_nt/manifest.json")
    for spec in read_json(manifest)["models"].values():
        for field in ("export_manifest", "checkpoint_manifest"):
            if field in spec:
                retain(ROOT / spec[field], "assets/" + spec[field])
    topology = {}
    for key, model_key in MODELS.items():
        spec = contract["models"][model_key]
        path = ROOT / spec["path"]
        if digest(path) != spec["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {key}")
        retain(path, "assets/" + spec["path"])
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("config.json"))["config"]
        topology[key] = [layer["config"]["name"] for layer in config["layers"]
                         if layer["class_name"] != "InputLayer"]

    # Canonical case bytes, head/input/settings/protocol are all fixed here.
    sys.path.insert(0, str(ROOT))
    import non_nt_runtime as rt
    arrays = {}
    for key in topology:
        for case in contract["cases"][key]:
            values = [rt.one_hot(case["window"])[None]]
            if key == "cas12a":
                values.append(rt.cas12a_derive(case["window"])[None])
            for index, value in enumerate(values):
                arrays[f"{key}/{case['id']}/input/{index}"] = value
    np.savez_compressed(snapshot / "inputs.npz", **arrays)
    frozen = {
        "schema_version": 1, "issue": 206, "run_id": run_id,
        "created_at": now(), "environment": environment(),
        "sources": sources, "inputs_sha256": digest(snapshot / "inputs.npz"),
        "offline_dependencies": dict(line.split("==") for line in
            (snapshot / "offline-requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")),
        "topology": topology,
        "expected_checks": expected_checks(contract, topology),
        "expected_arrays": expected_arrays(contract, topology),
        "stages": {k: list(v) for k, v in STAGES.items()},
    }
    write_json(work / "manifest.json", frozen)
    (work / "manifest.sha256").write_text(digest(work / "manifest.json"), encoding="ascii")
    write_json(work / "journal.json", {"state": "ready", "completed": []})
    return frozen


def load_manifest(work: Path, *, live: bool = False) -> dict:
    manifest_path = work / "manifest.json"
    if digest(manifest_path) != (work / "manifest.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Manifest checksum mismatch")
    manifest = read_json(manifest_path)
    for name, item in manifest["sources"].items():
        if digest(artifact(work, "snapshot/" + name)) != item["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
        if live and digest(Path(item["source"])) != item["sha256"]:
            raise ValueError(f"Input changed; new bundle/full remeasurement required: {name}")
    if digest(work / "snapshot/inputs.npz") != manifest["inputs_sha256"]:
        raise ValueError("Frozen input checksum mismatch")
    return manifest


def identity(work: Path, stage: str) -> dict:
    manifest = read_json(work / "manifest.json")
    return {"run_id": manifest["run_id"], "manifest_sha256": digest(work / "manifest.json"),
            "stage": stage, "backend": "tensorflow" if stage == "reference" else "torch"}


def verify_receipt(work: Path, stage: str) -> dict:
    receipt = read_json(work / f"{stage}.receipt.json")
    if receipt["identity"] != identity(work, stage):
        raise ValueError(f"Stage identity mismatch: {stage}")
    if set(receipt["artifacts"]) != set(STAGES[stage]):
        raise ValueError(f"Partial artifact set: {stage}")
    for name, checksum in receipt["artifacts"].items():
        if digest(artifact(work, name)) != checksum:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    for name, checksum in receipt.get("auxiliary_artifacts", {}).items():
        if digest(artifact(work, name)) != checksum:
            raise ValueError(f"Execution log checksum mismatch: {name}")
    for dependency in DEPENDENCIES.get(stage, ()):
        if receipt["dependencies"].get(dependency) != digest(work / f"{dependency}.receipt.json"):
            raise ValueError(f"Mixed dependency: {stage}/{dependency}")
    if not receipt.get("command") or not receipt.get("environment") or "exit_code" not in receipt:
        raise ValueError(f"Missing execution metadata: {stage}")
    if stage == "investigation":
        require_planned_termination(read_json(work / STAGES[stage][0]))
    return receipt


def update_journal(work: Path, journal: dict) -> None:
    temporary = work / "journal.tmp"
    temporary.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    temporary.replace(work / "journal.json")


@contextmanager
def stage_execution(work: Path, stage: str, *, snapshot_only: bool = False):
    """A recorded numerical failure may continue; interrupted stages may not."""
    if snapshot_only and stage != "offline":
        raise ValueError("Only independent recovery may execute from transported snapshots")
    journal = read_json(work / "journal.json")
    if journal["state"] != "ready" or stage in journal["completed"]:
        raise ValueError("Interrupted/finished stage cannot resume; use a new bundle")
    try:
        load_manifest(work, live=not snapshot_only)
    except (OSError, ValueError, KeyError):
        journal.update(state="invalidated", active=stage)
        update_journal(work, journal)
        raise
    for name in STAGES[stage]:
        if (work / name).exists():
            raise ValueError(f"Refusing pre-existing result: {name}")
    dependencies = {}
    for dependency in DEPENDENCIES.get(stage, ()):
        verify_receipt(work, dependency)
        dependencies[dependency] = digest(work / f"{dependency}.receipt.json")
    try:
        for completed in journal["completed"]:
            verify_receipt(work, completed)
    except (OSError, ValueError, KeyError):
        journal.update(state="invalidated", active=stage)
        update_journal(work, journal)
        raise
    # Exclusive marker also prevents concurrent writers from mixing stages.
    with (work / "active-stage.json").open("x", encoding="utf-8") as stream:
        json.dump({"stage": stage, "started_at": now()}, stream)
    journal.update(state="running", active=stage)
    update_journal(work, journal)
    receipt = {"identity": identity(work, stage), "started_at": now(),
               "command": [sys.executable, *sys.argv], "environment": environment(),
               "dependencies": dependencies, "input_mode": "snapshot" if snapshot_only else "live"}
    result = {"exit_code": None}
    try:
        yield result
        load_manifest(work, live=not snapshot_only)
        if result["exit_code"] not in (0, 1):
            raise ValueError("Stage did not finish with a valid numerical result")
        document_path = work / STAGES[stage][0]
        document = read_json(document_path)
        if stage == "investigation":
            require_planned_termination(document)
        if "identity" in document and document["identity"] != identity(work, stage):
            raise ValueError(f"Result identity mismatch; cannot restamp existing evidence: {stage}")
        document["identity"] = identity(work, stage)
        document_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        receipt.update(ended_at=now(), exit_code=result["exit_code"],
                       final_environment=environment(),
                       auxiliary_artifacts={name: digest(work / name) for name in (
                           ("investigation-stdout.txt", "investigation-stderr.txt") if stage == "investigation" else
                           ("offline-install-stdout.txt", "offline-install-stderr.txt",
                            "offline-run-stdout.txt", "offline-run-stderr.txt") if stage == "offline" else ())
                           if (work / name).exists()},
                       artifacts={name: digest(work / name) for name in STAGES[stage]})
        write_json(work / f"{stage}.receipt.json", receipt)
        journal["completed"].append(stage)
        journal.update(state="ready", active=None)
        update_journal(work, journal)
        (work / "active-stage.json").unlink()
    except BaseException:
        journal.update(state="aborted", active=stage)
        update_journal(work, journal)
        raise


class RawEvidence:
    """Retain required arrays and every observed intermediate forward/gradient."""

    def __init__(self):
        self.arrays = {}

    def add(self, name: str, value):
        import numpy as np
        if name in self.arrays:
            raise ValueError(f"Duplicate raw measurement: {name}")
        self.arrays[name] = np.array(value, copy=True)

    def observe(self, runtime, prefix):
        """Wrap this stage's runtime instance, including calls inside IG/SHAP."""
        for method in ("forward", "gradients"):
            original = getattr(runtime, method)
            count = [0]

            def recorded(arrays, *args, _fn=original, _count=count, _method=method, **kwargs):
                name = f"{prefix}/observed/{_method}/{_count[0]}"
                _count[0] += 1
                for index, value in enumerate(arrays):
                    self.add(f"{name}/input/{index}", value)
                result = _fn(arrays, *args, **kwargs)
                if _method == "gradients":
                    self.add(name + "/output", result[0])
                    for index, value in enumerate(result[1]):
                        self.add(f"{name}/gradient/{index}", value)
                else:
                    self.add(name + "/output", result)
                return result

            setattr(runtime, method, recorded)

    def save(self, path):
        import numpy as np
        if path.exists():
            raise ValueError(f"Refusing to overwrite raw evidence: {path}")
        np.savez_compressed(path, **self.arrays)
