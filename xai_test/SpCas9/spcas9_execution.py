"""Shared provenance and fresh-destination rules for 9-Hydra executions."""

import hashlib
import json
import subprocess
from pathlib import Path

from SpCas9.spcas9_data import BASE_DIR
from SpCas9.spcas9_split import file_sha256


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
    """Pin revision, dirty status and the executed project's source snapshot."""
    root = BASE_DIR.parent

    def git(*args: str) -> str:
        return (
            subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.PIPE
            )
            .decode("utf-8")
            .strip()
        )

    sources = list((BASE_DIR / "SpCas9").glob("*.py")) + list(BASE_DIR.glob("*.py"))
    sources += [root / "ruff.toml", BASE_DIR / "CONTEXT.md"]
    sources += list((BASE_DIR / "docs" / "adr").glob("*.md"))
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(sources)}
    snapshot = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        "snapshot_sha256": snapshot,
        "files": files,
    }
