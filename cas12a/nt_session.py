"""Colab installation/restart boundary for the NT notebook (#113).

Installing the fixed dependency recipe replaces NumPy and torch on disk while
the running kernel still holds their already-loaded compiled extensions. The
pure-Python half then disagrees with the binary half, so the first deep import
fails - `#110`'s first run died in preflight with `cannot import name '_slice'
from 'numpy._core.umath'` after the installation cell ran without a restart.

This module is the precondition check the notebook calls *before* importing the
workflow, so it imports stdlib only: importing NumPy here would trigger the very
failure it reports. It owns no data, training or recovery logic.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MARKER_NAME = "nt-colab-install.json"
MARKER_VERSION = 1

# Pinned distributions whose loaded binaries the installation can invalidate.
WATCHED = (
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "pandas",
    "scipy",
)

RESTART_STEPS = (
    "Restart the Python session (Runtime > Restart session), re-run the "
    "source-setup cell, SKIP the install cell, then re-run this cell."
)


class SessionRestartRequired(RuntimeError):
    """This kernel predates the installation; importing the workflow would fail."""


class InstallationAlreadyCompleted(RuntimeError):
    """The recipe is already installed on this VM; reinstalling is not routine."""


def default_marker(root=Path("/content")):
    """Marker location outside the repository checkout, which setup re-clones."""
    return Path(root) / MARKER_NAME


def kernel_identity():
    """Identify the live interpreter: a restart necessarily changes this."""
    return {
        "pid": os.getpid(),
        "started": _process_start(),
        "boot": _boot_identity(),
    }


def same_kernel(recorded, current):
    """Whether the recorded installation ran in the interpreter running now."""
    return all(
        recorded.get(key) == current.get(key) for key in ("pid", "started", "boot")
    )


def stale_imports():
    """Loaded modules whose version no longer matches the distribution on disk.

    Only the release portion is compared: a CUDA/CPU wheel reports a local build
    tag the distribution metadata drops (`2.12.0+cu126` installed as `2.12.0`),
    which is agreement, not staleness.
    """
    import importlib.metadata

    stale = {}
    for name in WATCHED:
        loaded = getattr(sys.modules.get(name), "__version__", None)
        if loaded is None:
            continue
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if _release(installed) != _release(loaded):
            stale[name] = {"loaded": loaded, "installed": installed}
    return stale


def _release(version):
    return str(version).split("+", 1)[0]


def read_marker(marker_path=None):
    """The recorded installation, or None when none was recorded on this VM."""
    path = Path(marker_path) if marker_path is not None else default_marker()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        marker = json.loads(text)
        kernel = marker["kernel"]
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"Unreadable nt-colab install marker {path}: {error}. Inspect it; "
            "the installation state of this VM cannot be determined."
        ) from error
    marker["kernel"] = dict(kernel)
    return marker


def record_installation(marker_path=None, identity=None, packages=None):
    """Record which kernel installed the recipe, so the restart is checkable."""
    path = Path(marker_path) if marker_path is not None else default_marker()
    marker = {
        "marker_version": MARKER_VERSION,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "kernel": identity or kernel_identity(),
        "packages": packages,
        "restart_required": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return path


def begin_installation(marker_path=None, force=False):
    """Refuse a repeat installation; re-running the cell is never a repair."""
    marker = read_marker(marker_path)
    if marker is None or force:
        return
    if same_kernel(marker["kernel"], kernel_identity()):
        raise InstallationAlreadyCompleted(
            f"The recipe was already installed by this live session at "
            f"{marker['recorded_utc']} and the restart is still pending; "
            f"reinstalling does not replace it. {RESTART_STEPS}"
        )
    raise InstallationAlreadyCompleted(
        f"The recipe was already installed on this VM at "
        f"{marker['recorded_utc']} by an earlier session, so this restarted "
        "kernel is the one to keep. Re-running this cell removes the torch "
        "extensions and replaces torch/NumPy underneath the loaded modules, "
        "which reintroduces the binary mismatch and forces another restart. "
        "Skip this cell. Set force=True only to install deliberately, and "
        "restart again afterwards."
    )


def require_restarted_session(marker_path=None):
    """Stop before the workflow import when this kernel predates installation.

    Returns the evidence for the notebook to print and keep with the session
    record. A session that already restarted, and one where no installation was
    recorded on this VM, both pass.
    """
    marker = read_marker(marker_path)
    current = kernel_identity()
    restarted = None if marker is None else not same_kernel(marker["kernel"], current)
    if restarted is False:
        raise SessionRestartRequired(
            f"The dependency installation ran in this live Python session "
            f"(pid {current['pid']}, recorded {marker['recorded_utc']}). Its "
            "replaced NumPy/torch binaries do not match the modules this "
            f"kernel already loaded, so the workflow import or preflight will "
            f"fail. {RESTART_STEPS}"
        )
    stale = stale_imports()
    if stale:
        detail = ", ".join(
            f"{name} loaded {versions['loaded']} but {versions['installed']} "
            f"is installed on disk"
            for name, versions in sorted(stale.items())
        )
        raise SessionRestartRequired(
            f"This session holds modules replaced underneath it: {detail}. "
            f"{RESTART_STEPS}"
        )
    return {
        "installation_recorded": marker is not None,
        "restarted": restarted,
        "installation_recorded_utc": marker and marker["recorded_utc"],
        "kernel": current,
        "stale_imports": stale,
    }


def _process_start():
    # Linux (Colab): clock ticks since boot, from the field after the comm
    # parentheses. Absent elsewhere, leaving the pid as the identity.
    try:
        stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return stat.rsplit(") ", 1)[1].split()[19]
    except IndexError:
        return None


def _boot_identity():
    try:
        return (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
    except OSError:
        return None
