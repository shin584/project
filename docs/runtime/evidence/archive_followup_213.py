"""Archive the terminal #213 boundary without overwriting #212 evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification_bundle import digest, load_manifest, read_json, verify_receipt, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    base, record = args.base.resolve(), args.record.resolve()
    work = base / "bundle"
    manifest = load_manifest(work, live=True)
    journal = read_json(work / "journal.json")
    required = {"reference", "verify", "diagnose", "float32", "ram", "offline", "investigation"}
    if journal["state"] != "ready" or journal.get("active") is not None or set(journal["completed"]) != required:
        raise ValueError("Requires all registered stages to have a terminal receipt")
    for stage in required:
        verify_receipt(work, stage)
    followup = read_json(base / "followup-213.json")
    start = read_json(base / "followup-213-start.json")
    summary = base / "summary-final.json"
    if (followup["manifest_sha256"] != digest(work / "manifest.json")
            or followup["summary_sha256"] != digest(summary)
            or digest(summary) != digest(base / "summary-final-replayed.json")):
        raise ValueError("Final manifest/report/replay mismatch")
    for name, checksum in start["boundary_files"].items():
        if name != "journal.json" and digest(work / name) != checksum:
            raise ValueError(f"Changed required boundary: {name}")
    previous = read_json(base / "archive-copy.json")
    for location in previous["paths"]:
        if digest(Path(location)) != previous["sha256"]:
            raise ValueError(f"Changed #212 archive: {location}")
    run_id = manifest["run_id"]
    archive_path = base / (run_id + "-investigation-evidence.zip")
    copy_path = Path("D:/MC_project-backups/dashboard-v3/issue205") / run_id / archive_path.name
    index_path = base / "archive-index-213.json"
    copy_record = base / "archive-copy-213.json"
    scripts = [Path(__file__).with_name(name) for name in (
        "run_followup_213.py", "observe_followup_213.py", "archive_followup_213.py")]
    for target in (archive_path, copy_path, index_path, copy_record, record,
                   *(base / script.name for script in scripts)):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite evidence: {target}")
    for script in scripts:
        shutil.copyfile(script, base / script.name)
    if digest(base / "run_followup_213.py") != start["controller_sha256"]:
        raise ValueError("Controller changed during measurement")
    files = sorted([p for name in ("bundle", "prepared") for p in (base / name).rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts]
                   + [p for p in base.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl", ".txt", ".py")])
    index = {p.relative_to(base).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size} for p in files}
    write_json(index_path, {"run_id": run_id, "files": index, "previous_archive": previous})
    files.append(index_path)
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(base).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(files):
            raise ValueError("Archive entry inventory mismatch")
        for path in files:
            with archive.open(path.relative_to(base).as_posix()) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest(path):
                    raise ValueError(f"Archive checksum mismatch: {path}")
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive_path, copy_path)
    checksum = digest(archive_path)
    if digest(copy_path) != checksum:
        raise ValueError("C:/D: archive copies differ")
    backup = {"paths": [str(archive_path), str(copy_path)], "sha256": checksum,
              "bytes": archive_path.stat().st_size, "verified_entries": len(files),
              "scope": "complete bundle, prepared dependencies, required and final reports, commands/logs, external RSS observations and recording tools",
              "independence": "two local volumes; no off-host or physical independence established"}
    write_json(copy_record, backup)
    investigation = read_json(work / "investigations.json")
    report = read_json(summary)
    commands = sorted([read_json(p) for p in base.glob("*.command.json")], key=lambda row: row["started_at"])
    write_json(record, {"issue": 213, "run_id": run_id, "base": str(base),
        "integration_commit": read_json(base / "preregistration.json")["integration_commit"],
        "manifest_sha256": digest(work / "manifest.json"), "frozen_sources": manifest["sources"],
        "canonical_inputs_sha256": manifest["inputs_sha256"], "journal": journal,
        "commands": commands, "followup": followup, "status": report["status"],
        "candidate_budget": investigation["candidate_budget"],
        "candidate_summary": investigation["candidate_summary"],
        "investigation_counts": report["investigation_counts"],
        "artifact_inventory": index, "archive": backup, "previous_archive": previous})
    print(json.dumps(backup, indent=2))


if __name__ == "__main__":
    main()
