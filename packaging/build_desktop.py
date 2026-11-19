"""Build an independent Windows x64 v3 release from verified local inputs."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_assets import MANIFEST, restore, sha256

APP = "XaiDashboardV3"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inventory(folder: Path) -> dict:
    return {
        path.relative_to(folder).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(folder.rglob("*")) if path.is_file()
    }


def stage(output: Path, assets: Path, models: Path, archive: Path) -> dict:
    resource_map = read_json(ROOT / "docs/preservation/resources.json")["files"]
    restored = restore(archive, output)
    for name, expected in resource_map.items():
        if name.endswith(".safetensors"):
            relative = Path(name)
            source = models / relative.name
        elif name.startswith(".packaging/assets/"):
            relative = Path(name).relative_to(".packaging/assets")
            source = assets / relative
        else:
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if destination.stat().st_size != expected["bytes"] or sha256(destination) != expected["sha256"]:
            raise ValueError(f"Staged resource differs from the preservation inventory: {name}")
    return restored


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, encoding="utf-8").strip()


def build(output: Path, assets: Path, models: Path, archive: Path, wheelhouse: Path) -> None:
    if (sys.platform != "win32" or platform.machine() != "AMD64"
            or sys.version_info[:2] != (3, 13)):
        raise RuntimeError("Use Windows x64 CPython 3.13.")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Commit tracked changes before producing a release.")
    commit = git("rev-parse", "HEAD")
    output.mkdir(parents=True, exist_ok=False)
    staged = output / "assets"
    restored = stage(staged, assets, models, archive)
    staged_files = inventory(staged)
    write_json(output / "staging.json", {"recovery": restored, "files": staged_files})
    wheels = inventory(wheelhouse)
    if not wheels:
        raise ValueError("The offline wheelhouse is empty.")
    command = [
        sys.executable, "-m", "PyInstaller", "--clean",
        "--distpath", str(output / "dist"), "--workpath", str(output / "build"),
        "packaging/desktop.spec",
    ]
    env = {**os.environ, "DASHBOARD_V3_STAGED_ASSETS": str(staged),
           "KERAS_BACKEND": "torch", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONPATH": str(ROOT / "packaging/offline_build"),
           "PYINSTALLER_CONFIG_DIR": str(output / "pyinstaller-cache")}
    with (output / "build.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked source files changed during the build.")

    bundle = output / "dist" / APP
    shutil.copyfile(ROOT / "packaging/release-v3.txt", bundle / "START-HERE.txt")
    shutil.copyfile(ROOT / "packaging/verify-v3.ps1", bundle / "verify-v3.ps1")
    files = inventory(bundle)
    for name, expected in staged_files.items():
        if files.get("_internal/" + name) != expected:
            raise ValueError(f"Built resource differs from the staged input: {name}")
    adopted = read_json(staged / MANIFEST)
    report = {
        "schema_version": 1, "issue": 204, "executable": f"{APP}.exe",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_host": platform.node(), "python": platform.python_version(),
        "platform": platform.platform(), "source_commit": commit,
        "build_command": command[1:],
        "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
        "wheelhouse": wheels,
        "runtime": {"non_nt": "Keras 3 with PyTorch CPU for forward and gradients",
                    "nt": "PyTorch CPU with local Transformers tokenizer",
                    "auxiliary_runtime": None, "tensorflow": "excluded",
                    "webview2": "bundled Fixed Version x64"},
        "non_nt_archive_sha256": sha256(archive),
        "non_nt_manifest_sha256": staged_files[MANIFEST]["sha256"],
        "non_nt_models": adopted["models"],
        "non_nt_provenance": {
            name: entry for name, entry in adopted["files"].items()
            if name.endswith((".keras", "export-manifest.json", "checkpoint.json", "source_snapshot.zip"))
        },
        "files": files, "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files.values()),
    }
    write_json(bundle / "release-manifest.json", report)
    print(f"Built {bundle} with {report['file_count']} files and {report['total_bytes']} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New staging and build directory")
    parser.add_argument("--assets", type=Path, default=ROOT / ".packaging/assets",
                        help="Directory containing the pinned tokenizer and WebView2 folders")
    parser.add_argument("--models", type=Path, default=ROOT / "models",
                        help="Directory containing the two pinned NT checkpoints")
    parser.add_argument("--non-nt-archive", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.assets.resolve(), args.models.resolve(),
          args.non_nt_archive.resolve(), args.wheelhouse.resolve())


if __name__ == "__main__":
    main()
