"""Verify or restore the adopted non-NT assets without model runtimes or network."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
MANIFEST = "models/non_nt/manifest.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root: Path) -> dict:
    """Verify all checkpoint and provenance bytes against the pinned manifest."""
    lock = json.loads((ROOT / "docs/preservation/model-vault.json").read_text(encoding="utf-8"))
    manifest_path = root / MANIFEST
    if sha256(manifest_path) != lock["manifest_sha256"]:
        raise ValueError("Asset manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = root / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {name}")
    return {"status": "verified", "file_count": len(manifest["files"]),
            "total_bytes": sum(f["bytes"] for f in manifest["files"].values())}


def restore(archive: Path, destination: Path) -> dict:
    """Restore only the trusted archive into a new directory; never overwrite."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Recovery destination must be new: {destination}")
    lock = json.loads((ROOT / "docs/preservation/model-vault.json").read_text(encoding="utf-8"))
    if archive.stat().st_size != lock["archive_bytes"] or sha256(archive) != lock["archive_sha256"]:
        raise ValueError("Recovery archive checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="non-nt-restore-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            # The whole archive is pinned before extraction. Reject unsafe paths
            # as well, so later archive revisions cannot escape staging.
            for member in bundle.infolist():
                path = (staging / member.filename).resolve()
                if not path.is_relative_to(staging.resolve()):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
        report = verify(staging)
        os.rename(staging, destination)
    return {**report, "destination": str(destination.resolve()), "network_used": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recovery = commands.add_parser("restore")
    recovery.add_argument("--archive", required=True, type=Path)
    recovery.add_argument("--destination", required=True, type=Path)
    check = commands.add_parser("verify")
    check.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = (restore(args.archive, args.destination) if args.command == "restore"
              else verify(args.root))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
