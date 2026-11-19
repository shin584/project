"""Execute the already registered #213 stage once; retain the #212 boundary."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_required_212 import execute
from verification_bundle import ROOT, digest, load_manifest, now, read_json, verify_receipt, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    work = base / "bundle"
    manifest = load_manifest(work, live=True)
    journal = read_json(work / "journal.json")
    required = {"reference", "verify", "diagnose", "float32", "ram", "offline"}
    if journal != {"state": "ready", "completed": journal.get("completed"), "active": None} or set(journal["completed"]) != required:
        raise ValueError("Requires the normal, unstarted #212 handoff boundary")
    handoff = read_json(base / "handoff.json")
    if handoff["manifest_sha256"] != digest(work / "manifest.json") or handoff["summary_sha256"] != digest(base / "summary.json"):
        raise ValueError("Changed #212 handoff")
    for stage in required:
        verify_receipt(work, stage)
    for name in ("followup-213-start.json", "investigation-213.command.json", "summary-final.json", "summary-final-replayed.json", "followup-213.json"):
        if (base / name).exists():
            raise FileExistsError(f"Refusing to repeat or overwrite: {name}")
    before = {p.relative_to(work).as_posix(): digest(p) for p in work.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    command = [sys.executable, "-X", "utf8", str(ROOT / "docs/runtime/verify_contract.py"), "investigation", "--work", str(work)]
    write_json(base / "followup-213-start.json", {"run_id": manifest["run_id"], "started_at": now(),
        "command": command, "cwd": str(ROOT), "controller_sha256": digest(Path(__file__)),
        "manifest_sha256": digest(work / "manifest.json"), "boundary_files": before,
        "rule": "Single attempt; no resume, budget extension or candidate fallback."})
    code = execute(command, base, "investigation-213")
    journal = read_json(work / "journal.json")
    if code != 0 or journal["state"] != "ready" or "investigation" not in journal["completed"]:
        raise RuntimeError("Unexpected interruption; preserve evidence and use a new bundle with full required remeasurement")
    verify_receipt(work, "investigation")
    changed = [name for name, checksum in before.items() if name != "journal.json" and digest(work / name) != checksum]
    if changed:
        raise ValueError(f"Required-stage evidence changed: {changed}")
    summary_codes = []
    after = {p.relative_to(work).as_posix(): digest(p) for p in work.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    for label, output in (("assemble-213", "summary-final.json"), ("replay-213", "summary-final-replayed.json")):
        summary_codes.append(execute([sys.executable, "-X", "utf8", str(ROOT / "docs/runtime/assemble_verification.py"),
            "--work", str(work), "--output", str(base / output)], base, label))
    if digest(base / "summary-final.json") != digest(base / "summary-final-replayed.json"):
        raise ValueError("Summary replay differs")
    if after != {p.relative_to(work).as_posix(): digest(p) for p in work.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts}:
        raise ValueError("Summary assembly changed bundle evidence")
    write_json(base / "followup-213.json", {"run_id": manifest["run_id"], "ended_at": now(), "journal": journal,
        "required_boundary_preserved": True, "byte_identical_replay": True,
        "manifest_sha256": digest(work / "manifest.json"), "summary_sha256": digest(base / "summary-final.json"),
        "summary_exit_codes": summary_codes, "status": read_json(base / "summary-final.json")["status"]})
    return max(summary_codes)


if __name__ == "__main__":
    raise SystemExit(main())
