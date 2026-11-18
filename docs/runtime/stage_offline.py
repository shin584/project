"""Prepare frozen inputs, then independently restore and run both real models.

prepare --work <bundle> --wheelhouse <wheels> --output <new-preparation>
<new-restored-root> --prepared <preparation> --work <bundle> --output <bundle>/staged.json
Python/stdlib/OS and pip are external prerequisites; see recovery-209.md.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import zipfile
import csv
import base64
from email.parser import BytesParser
from concurrent.futures import ThreadPoolExecutor

from offline_policy import recovery_failures, restored_name
from verification_bundle import ROOT, digest, load_manifest, read_json, stage_execution, write_json
from verification_policy import runtime_version_failures


def safe_file(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"Prepared path escapes root: {relative}")
    return path


def dependency_checksums(root):
    # Wheel hashes cover the full distributions. Check installation manifests
    # and native executable files here; imported Python files are checked below
    # against their wheel RECORD hashes. Avoid reading unused development headers.
    paths = [p for p in (root / "dependencies").rglob("*")
             if p.is_file() and (p.name == "RECORD" or p.suffix.lower() in (".dll", ".pyd"))]
    with ThreadPoolExecutor(max_workers=16) as workers:
        checksums = workers.map(digest, paths)
        return {p.relative_to(root).as_posix(): checksum for p, checksum in zip(paths, checksums)}


def verify_imported_dependencies(root, origins):
    expected = {}
    for record in (root / "dependencies").glob("*.dist-info/RECORD"):
        with record.open(encoding="utf-8", newline="") as stream:
            for name, checksum, _ in csv.reader(stream):
                if checksum.startswith("sha256="):
                    expected[(record.parent.parent / name).resolve()] = base64.urlsafe_b64decode(
                        checksum[7:] + "=" * (-len(checksum[7:]) % 4)).hex()
    checked = {}
    for value in set(origins.values()):
        path = Path(value)
        if path.is_relative_to(root / "dependencies"):
            checksum = digest(path)
            if expected.get(path) != checksum:
                raise ValueError(f"Imported dependency disagrees with wheel RECORD: {path}")
            checked[path.relative_to(root).as_posix()] = checksum
    return checked


def prepare(work, wheelhouse, output):
    manifest = load_manifest(work, live=True)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, item in manifest["sources"].items():
        target = safe_file(output, restored_name(name))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / "snapshot" / name, target)
        files[target.relative_to(output).as_posix()] = item["sha256"]
    shutil.copyfile(work / "snapshot/inputs.npz", output / "payload/recovery/inputs.npz")
    files["payload/recovery/inputs.npz"] = manifest["inputs_sha256"]
    wheels = {}
    wheel_dir = output / "wheels"
    wheel_dir.mkdir()
    for source in sorted(wheelhouse.glob("*.whl")):
        if source.is_symlink():
            raise ValueError("Wheel symlinks are not preparation inputs")
        with zipfile.ZipFile(source) as archive:
            metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA") and n.count("/") == 1]
            if len(metadata) != 1:
                raise ValueError(f"Invalid wheel metadata: {source.name}")
            info = BytesParser().parsebytes(archive.read(metadata[0]))
            name = info["Name"].lower().replace("_", "-")
            if name.startswith("tensorflow") or name in wheels:
                raise ValueError(f"Unexpected/duplicate target wheel: {name}")
            wheels[name] = {"version": info["Version"], "file": "wheels/" + source.name}
        target = wheel_dir / source.name
        shutil.copyfile(source, target)
        files["wheels/" + source.name] = digest(target)
    requirements = output / "payload/recovery/offline-requirements.txt"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, version = line.split("==")
        if wheels.get(name, {}).get("version", "").split("+")[0] != version:
            raise ValueError(f"Missing/mismatched declared wheel: {line}")
    data = {"schema_version": 1, "bundle_manifest_sha256": digest(work / "manifest.json"),
            "files": files, "wheels": wheels, "command": [sys.executable, *sys.argv],
            "original_roots": [str(ROOT), str(work), str(output), sysconfig.get_path("purelib")],
            "original_files": sorted({item["source"] for item in manifest["sources"].values()})}
    for name, checksum in files.items():
        if digest(safe_file(output, name)) != checksum:
            raise ValueError(f"Preparation checksum mismatch: {name}")
    write_json(output / "preparation.json", data)
    print(json.dumps({"status": "prepared", "files": len(files), "wheels": len(wheels),
                      "manifest_sha256": digest(output / "preparation.json"), "output": str(output)}))


def run(work, prepared, restored):
    data = read_json(prepared / "preparation.json")
    manifest = load_manifest(work)
    if data["bundle_manifest_sha256"] != digest(work / "manifest.json"):
        raise ValueError("Prepared inputs belong to a different bundle")
    for name, item in manifest["sources"].items():
        if data["files"].get(restored_name(name)) != item["sha256"]:
            raise ValueError(f"Prepared source disagrees with frozen bundle: {name}")
    if data["files"].get("payload/recovery/inputs.npz") != manifest["inputs_sha256"]:
        raise ValueError("Prepared canonical inputs disagree with frozen bundle")
    originals = [Path(p).resolve() for p in data["original_roots"]] + [prepared, work, ROOT]
    if any(restored.is_relative_to(p) or p.is_relative_to(restored) for p in originals):
        raise ValueError("Restored root must be independent of original/prepared/bundle paths")
    for name, checksum in data["files"].items():
        if digest(safe_file(prepared, name)) != checksum:
            raise ValueError(f"Prepared input checksum mismatch: {name}")
    restored.mkdir(parents=True, exist_ok=False)
    for name in data["files"]:
        target = safe_file(restored, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(prepared, name), target)
    verified = {name: digest(safe_file(restored, name)) for name in data["files"]}
    if verified != data["files"]:
        raise ValueError("Restored input checksum mismatch")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("PYTHON", "PIP_"))
           and k not in ("KERAS_BACKEND", "TORCH_HOME")}
    command = [sys.executable, "-I", "-m", "pip", "--isolated", "install", "--no-index",
               "--no-cache-dir", "--disable-pip-version-check", "--only-binary=:all:", "--no-compile",
               "--find-links", str(restored / "wheels"), "--target", str(restored / "dependencies"),
               "-r", str(restored / "payload/recovery/offline-requirements.txt")]
    install = subprocess.run(command, capture_output=True, text=True, cwd=restored, env=env)
    (work / "offline-install-stdout.txt").write_text(install.stdout, encoding="utf-8")
    (work / "offline-install-stderr.txt").write_text(install.stderr, encoding="utf-8")
    recovery = {**data, "restored_root": str(restored), "verified_files": verified,
                "preparation_sha256": digest(prepared / "preparation.json"),
                "install": {"command": command, "exit_code": install.returncode},
                "installed_files": dependency_checksums(restored)}
    recovery["dependency_checksum_scope"] = "full wheel archives, installed RECORD/native files before and after, imported modules against RECORD"
    result = {"recovery": recovery, "run": {"status": "fail"}, "exit_code": install.returncode}
    if install.returncode == 0:
        config = {"original_roots": sorted({*data["original_roots"], str(prepared), str(work), str(ROOT),
                                            sysconfig.get_path("purelib"),
                                            *(item["source"] for item in manifest["sources"].values())}),
                  "original_files": sorted({item["source"] for item in manifest["sources"].values()})}
        write_json(restored / "runner-config.json", config)
        command = [sys.executable, "-I", "-S", str(restored / "payload/docs/runtime/offline_runner.py"), str(restored)]
        finished = subprocess.run(command, capture_output=True, text=True, cwd=restored, env=env)
        result.update(command=command, executable=sys.executable, exit_code=finished.returncode)
        (work / "offline-run-stdout.txt").write_text(finished.stdout, encoding="utf-8")
        (work / "offline-run-stderr.txt").write_text(finished.stderr, encoding="utf-8")
        if (restored / "runner-report.json").exists():
            result["run"] = read_json(restored / "runner-report.json")
        if (restored / "offline-raw.npz").exists():
            shutil.copyfile(restored / "offline-raw.npz", work / "offline-raw.npz")
    contract = read_json(work / "snapshot/contract.json")
    errors = runtime_version_failures(result["run"].get("versions", {}), contract)
    if dependency_checksums(restored) != recovery["installed_files"]:
        errors.append("Restored dependencies changed during execution")
    try:
        recovery["imported_dependency_files"] = verify_imported_dependencies(
            restored, result["run"].get("module_origins", {}))
    except (OSError, ValueError) as error:
        errors.append(str(error))
    errors += recovery_failures(result, load_manifest(work), digest(work / "manifest.json"), contract)
    for name, checksum in verified.items():
        if digest(safe_file(restored, name)) != checksum:
            errors.append(f"Restored input changed during execution: {name}")
    result["run"]["validation_errors"] = errors
    if errors:
        result["run"]["status"] = "fail"
    write_json(work / "staged.json", result)
    if not (work / "offline-raw.npz").exists():
        with zipfile.ZipFile(work / "offline-raw.npz", "w"):
            pass
    print(json.dumps({"status": result["run"]["status"], "errors": errors, "output": str(work / "staged.json")}))
    return 0 if result["run"]["status"] == "pass" and result["exit_code"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", help="new independent restored directory, or 'prepare'")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--snapshot-only", action="store_true",
                        help="run transported frozen recovery without consulting original sources")
    args = parser.parse_args()
    work = args.work.resolve()
    try:
        if args.staging == "prepare":
            if not args.wheelhouse:
                parser.error("prepare requires --wheelhouse")
            prepare(work, args.wheelhouse.resolve(), args.output.resolve())
            return 0
        if not args.prepared:
            parser.error("recovery requires --prepared; developer site-packages staging is forbidden")
        if args.output.resolve() != work / "staged.json":
            parser.error("--output must be <work>/staged.json")
        if args.snapshot_only:
            manifest = load_manifest(work)
            if digest(Path(__file__)) != manifest["sources"]["code/docs/runtime/stage_offline.py"]["sha256"]:
                raise ValueError("Recovery controller differs from frozen code")
        with stage_execution(work, "offline", snapshot_only=args.snapshot_only) as result:
            try:
                result["exit_code"] = run(work, args.prepared.resolve(), Path(args.staging).resolve())
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                write_json(work / "staged.json", {"run": {"status": "fail", "error": str(error)}, "exit_code": 1})
                with zipfile.ZipFile(work / "offline-raw.npz", "w"):
                    pass
                result["exit_code"] = 1
        return result["exit_code"]
    except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "fail", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
