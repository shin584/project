"""Verify both #213 archives and replay their frozen assembler without measurement.

Use a NEW --output directory. --finalize also makes new, byte-identical terminal
copies next to the existing archives, refusing to overwrite either generation.
The versioned #213 index is the checksum anchor, not the temporary measurement
folder. Python, NumPy and the OS remain external replay prerequisites.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def verify_copies(spec):
    results = []
    for location in spec["paths"]:
        path = Path(location)
        checksum = digest(path)
        if checksum != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError(f"Archive checksum/size mismatch: {path}")
        results.append({"path": str(path), "sha256": checksum, "bytes": path.stat().st_size})
    if len(results) != 2:
        raise ValueError("Requires both agreed local-volume copies")
    return results


def extract_verified(archive_path, restored, index):
    inventory = index["artifact_inventory"]
    expected = set(inventory) | {"archive-index-213.json"}
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError("Archive entry inventory mismatch")
        embedded = json.loads(archive.read("archive-index-213.json"))
        if embedded["run_id"] != index["run_id"] or embedded["files"] != inventory:
            raise ValueError("Embedded/versioned archive index mismatch")
        # Check every member against the versioned index before extraction.
        for name in expected:
            target = (restored / name).resolve()
            if not target.is_relative_to(restored) or "\\" in name:
                raise ValueError(f"Archive path escapes restored root: {name}")
            if name in inventory:
                with archive.open(name) as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                if (checksum != inventory[name]["sha256"]
                        or archive.getinfo(name).file_size != inventory[name]["bytes"]):
                    raise ValueError(f"Archive member checksum/size mismatch: {name}")
        archive.extractall(restored)
    return {p.relative_to(restored).as_posix(): digest(p)
            for p in restored.rglob("*") if p.is_file()}


def verify_vault(restored, index):
    work = restored / "bundle"
    registration = read(restored / "preregistration.json")
    sources = read(work / "manifest.json")["sources"]
    names = [name for name, item in sources.items() if name.endswith("/model-vault.json")
             and item["sha256"] == registration["model_vault"]["vault_record_sha256"]]
    if len(names) != 1:
        raise ValueError("Missing registered pinned vault")
    vault = read(work / "snapshot" / names[0])
    contract = read(work / "snapshot/contract.json")
    results = []
    for location in vault["locations"]:
        path = Path(location)
        if digest(path) != vault["archive_sha256"] or path.stat().st_size != vault["archive_bytes"]:
            raise ValueError(f"Pinned model vault checksum/size mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            hashes = {key: hashlib.sha256(archive.read(model["path"])).hexdigest()
                      for key, model in contract["models"].items()}
        if any(hashes[key] != model["sha256"] for key, model in contract["models"].items()):
            raise ValueError(f"Pinned model bytes differ from measured bytes: {path}")
        results.append({"path": str(path), "sha256": vault["archive_sha256"], "models": hashes})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=int, choices=(0, 1), default=1,
                        help="Restore from C: (0) or D: (1); always verify both")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    index = read(args.index)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Requires a new output directory: {output}")
    originals = [Path(index["base"]).resolve(),
                 Path(index["frozen_sources"]["code/non_nt_runtime.py"]["source"]).parent.resolve()]
    if any(output.is_relative_to(root) for root in originals):
        raise ValueError("Replay output must be outside original measurement/source roots")
    copies = verify_copies(index["archive"])
    previous = verify_copies(index["previous_archive"])
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    restored = output / "restored"
    restored.mkdir()
    before = extract_verified(Path(copies[args.source]["path"]), restored, index)
    work = restored / "bundle"
    manifest = read(work / "manifest.json")
    if (manifest["run_id"] != index["run_id"]
            or digest(work / "manifest.json") != index["manifest_sha256"]):
        raise ValueError("Restored manifest/run identity mismatch")
    vault = verify_vault(restored, index)
    runner = output / "guarded_replay_214.py"
    shutil.copyfile(Path(__file__).with_name(runner.name), runner)
    config = {"restored": str(restored), "original_roots": list(map(str, originals)),
              "probe_files": [str(Path(index["base"]) / "bundle/manifest.json"),
                              *[item["source"] for item in manifest["sources"].values()]],
              "output": str(output)}
    write(output / "replay-config.json", config)
    command = [sys.executable, "-I", "-B", "-X", "utf8", str(runner),
               str(output / "replay-config.json")]
    with (output / "replay.stdout.txt").open("x", encoding="utf-8") as stdout, \
            (output / "replay.stderr.txt").open("x", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=output, stdout=stdout, stderr=stderr)
    report_path = output / "summary-regenerated.json"
    expected_report = restored / "summary-final.json"
    report = read(report_path)
    guard = read(output / "guard-report.json")
    expected_exit = 0 if report["status"]["overall"] in ("pass", "completed_with_findings") else 1
    if (result.returncode != expected_exit or digest(report_path) != digest(expected_report)
            or report["status"] != index["status"] or guard["violations"]
            or not all(row["blocked"] for row in guard["probes"])):
        raise ValueError("Replay report/exit/raw links/guard differs from archived evidence")
    after = {p.relative_to(restored).as_posix(): digest(p)
             for p in restored.rglob("*") if p.is_file()}
    if before != after:
        raise ValueError("Replay modified restored evidence")
    final = None
    if args.finalize:
        paths = [Path(row["path"]).with_name(index["run_id"] + "-final-evidence.zip") for row in copies]
        if any(path.exists() for path in paths):
            raise FileExistsError("Refusing to overwrite a final archive")
        for path in paths:
            with Path(copies[0]["path"]).open("rb") as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
        final = {**index["archive"], "paths": list(map(str, paths))}
        verify_copies(final)
    record = {"issue": 214, "run_id": index["run_id"], "started_at": started,
              "completed_at": datetime.now(timezone.utc).isoformat(),
              "index_sha256": digest(args.index), "integration_commit": index["integration_commit"],
              "recorder_sha256": digest(Path(__file__)), "guard_sha256": digest(runner),
              "archive_copies": copies, "previous_archive_copies": previous, "final_archive": final,
              "archive_source": copies[args.source]["path"], "restored_files": before,
              "model_vault": vault, "command": command, "assembler_exit_code": result.returncode,
              "status": report["status"], "report_sha256": digest(report_path),
              "byte_identical_report_and_raw_links": True, "unchanged_restored_files": len(before),
              "guard": guard, "measurements_started": 0,
              "scope": "report replay only; two local volumes, no off-host/physical independence or OS isolation"}
    write(output / "replay-214.json", record)
    print(json.dumps({"report": str(report_path), "status": report["status"],
                      "assembler_exit_code": result.returncode, "verified_files": len(before)}))
    # Archive/replay verification succeeded. The nested numerical exit remains
    # independently recorded as 1; this exit is not a numerical-quality claim.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
