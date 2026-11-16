"""Build the production folder and record hashes for transfer to another PC."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    sources = sorted({*ROOT.glob("*.py"), *ROOT.glob("desktop/*.py"),
                      *ROOT.glob("core/*.py"), *ROOT.glob("xai/*.py"),
                      ROOT / "packaging/desktop.spec", ROOT / "packaging/runtime_hook.py"})
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "packaging/desktop.spec"], cwd=ROOT, check=True)
    if source_hashes != {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}:
        raise RuntimeError("Source files changed during the build; rebuild before distributing")
    bundle = ROOT / "dist/XaiDashboard"
    files = sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "release-manifest.json")
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_host": platform.node(), "python": platform.python_version(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes,
        "versions": {name: version(name) for name in (
            "pyinstaller", "pywebview", "torch", "transformers", "safetensors",
            "numpy", "pandas", "matplotlib", "logomaker", "shap", "psutil",
        )},
        "file_count": len(files), "total_bytes": sum(p.stat().st_size for p in files),
        "files": {p.relative_to(bundle).as_posix(): sha256(p) for p in files},
    }
    (bundle / "release-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Built {bundle}: {report['file_count']} files, {report['total_bytes']} bytes")


if __name__ == "__main__":
    main()
