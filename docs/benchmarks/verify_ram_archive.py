"""Verify registered observer/contract bytes from Git, including line endings."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def verify(staged=False):
    checks = []

    def check(path, expected):
        relative = path.relative_to(ROOT).as_posix()
        ref = ":" if staged else "HEAD:"
        blob = subprocess.check_output(["git", "show", ref + relative], cwd=ROOT)
        passed = hashlib.sha256(blob).hexdigest() == expected
        checks.append(passed)
        if not passed:
            print("FAIL", relative)

    for path in sorted(BASE.glob("issue203-ram-*/contract.json")):
        contract = json.loads(path.read_text(encoding="utf-8"))
        observer = path.parent / "observer"
        if not observer.is_dir():
            observer = BASE
        for name, expected in contract["observer_sha256"].items():
            check(observer / name, expected)
        for launch in sorted(path.parent.glob("*/launch.json")):
            record = json.loads(launch.read_text(encoding="utf-8"))
            check(path, record["contract_sha256"])
    print(f"{sum(checks)} passed / {len(checks) - sum(checks)} failed / 0 skipped")
    return int(not checks or not all(checks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    raise SystemExit(verify(parser.parse_args().staged))
