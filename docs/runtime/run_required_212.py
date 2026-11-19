"""Run the preregistered required stages once, stopping at the #213 boundary.

Numerical failures preserve receipts and do not stop collection. Unexpected
interruptions abort this run; this command never resumes an existing directory.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from verification_bundle import ROOT, digest, now, read_json, write_json


def execute(argv, directory, record, cwd=ROOT):
    started = now()
    finished = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    (directory / (record + ".stdout.txt")).write_text(finished.stdout, encoding="utf-8")
    (directory / (record + ".stderr.txt")).write_text(finished.stderr, encoding="utf-8")
    row = {"command": argv, "cwd": str(cwd), "started_at": started, "ended_at": now(),
           "exit_code": finished.returncode}
    write_json(directory / (record + ".command.json"), row)
    print(json.dumps({"stage": record, **row}), flush=True)
    return finished.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=False)
    work, prepared, restored = [base / name for name in ("bundle", "prepared", "restored")]
    docs = ROOT / "docs/runtime"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    registration = {"run_id": args.run_id, "registered_at": now(), "integration_commit": revision,
        "review_baseline_193": "c7df95b22683fbcefc0e560266bff01ef6190dc2",
        "implementation_base": "b358c9a", "contract_sha256": digest(docs / "contract.json"),
        "protocol_sha256": digest(docs / "protocol-212.json"),
        "wheelhouse": str(args.wheelhouse.resolve()), "executable": sys.executable,
        "model_vault": {"vault_record_sha256": digest(ROOT / "docs/preservation/model-vault.json"),
                        **read_json(ROOT / "docs/preservation/model-vault.json")},
        "unstarted_followup": ["candidate_comparison_90_minutes", "conditional_float64_30_minutes"],
        "handoff": "Completed stages only. Changed inputs or unexpected interruption require a new bundle and all required measurements."}
    python = sys.executable
    commands = [
        ("init", [python, "-X", "utf8", str(docs / "verify_contract.py"), "init", "--work", str(work),
            "--run-id", args.run_id, "--protocol", str(docs / "protocol-212.json"),
            "--extra-code", str(base / "preregistration.json"),
            "--extra-code", str(ROOT / "docs/preservation/model-vault.json")], ROOT),
        *[(stage, [python, "-X", "utf8", str(docs / "verify_contract.py"), stage, "--work", str(work)], ROOT)
          for stage in ("reference", "verify", "diagnose", "float32", "ram")],
        ("prepare", [python, "-X", "utf8", str(docs / "stage_offline.py"), "prepare", "--work", str(work),
            "--wheelhouse", str(args.wheelhouse.resolve()), "--output", str(prepared)], ROOT),
        ("offline", [python, "-X", "utf8", "-S", str(prepared / "payload/docs/runtime/stage_offline.py"),
            str(restored), "--prepared", str(prepared), "--work", str(work), "--output", str(work / "staged.json"),
            "--snapshot-only"], prepared),
        ("assemble", [python, "-X", "utf8", str(docs / "assemble_verification.py"), "--work", str(work),
            "--output", str(base / "summary.json")], ROOT),
    ]
    registration["stage_order"] = [stage for stage, _, _ in commands]
    registration["commands"] = [{"stage": stage, "argv": argv, "cwd": str(cwd)} for stage, argv, cwd in commands]
    write_json(base / "preregistration.json", registration)
    for stage, argv, cwd in commands:
        code = execute(argv, base, stage, cwd)
        journal = read_json(work / "journal.json") if (work / "journal.json").exists() else {}
        measurement_missing = stage not in ("init", "prepare", "assemble") and (
            stage not in journal.get("completed", []) or not (work / f"{stage}.receipt.json").is_file())
        if (code not in (0, 1) or journal.get("state") != "ready" or measurement_missing
                or (stage in ("init", "prepare") and code)):
            raise RuntimeError(f"Unexpected stage failure: {stage}; preserve this directory and start a new run")
    write_json(base / "handoff.json", {"run_id": args.run_id, "integration_commit": revision,
        "manifest_sha256": digest(work / "manifest.json"), "journal": read_json(work / "journal.json"),
        "summary_sha256": digest(base / "summary.json"), "state": "completed_required_stages",
        "next_stage": "investigation", "next_ticket": 213, "numerical_status": read_json(base / "summary.json")["status"]})
    return 1 if read_json(base / "summary.json")["status"]["overall"] in ("fail", "incomplete") else 0


if __name__ == "__main__":
    raise SystemExit(main())
