"""Shared provenance and fresh-destination rules for 9-Hydra executions."""

import hashlib
import json
import subprocess
from pathlib import Path

from SpCas9.spcas9_split import file_sha256

SOURCE_DIR = Path(__file__).resolve().parent


def reserve_destination(destination: Path, sources: tuple[Path, ...] = ()) -> Path:
    """Reserve once, rejecting source overlap, including through symlinks."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    resolved = destination.resolve()
    protected = list(sources)
    for source in sources:
        if source.is_dir():
            protected.extend(
                p for p in source.rglob("*") if p.is_symlink() or p.is_junction()
            )
    for source in protected:
        source = source.resolve()
        if (
            resolved == source
            or resolved in source.parents
            or source in resolved.parents
        ):
            raise ValueError(f"diagnostic destination overlaps source: {source}")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def source_record() -> dict:
    """Snapshot SpCas9 Python sources, with scoped Git metadata when available.

    A copied SpCas9 folder needs no repository. Unknown Git metadata is null;
    the source hashes still identify exactly which code was executed.
    """

    def git(*args: str) -> str | None:
        try:
            return (
                subprocess.check_output(
                    ["git", "-C", str(SOURCE_DIR), *args], stderr=subprocess.PIPE
                )
                .decode("utf-8")
                .strip()
            )
        except (OSError, subprocess.CalledProcessError):
            return None

    sources = SOURCE_DIR.glob("*.py")
    files = {
        p.relative_to(SOURCE_DIR.parent).as_posix(): file_sha256(p)
        for p in sorted(sources)
    }
    snapshot = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    status = git("status", "--porcelain", "--untracked-files=normal", "--", ".")
    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "snapshot_sha256": snapshot,
        "files": files,
    }
