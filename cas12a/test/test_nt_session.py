"""Offline checks for the Colab installation/restart boundary (#113).

No Colab session, Drive mount or GPU is involved: the guard is stdlib-only and
decides from a recorded marker plus this interpreter's own loaded modules.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cas12a.nt_session import (
    InstallationAlreadyCompleted,
    SessionRestartRequired,
    begin_installation,
    default_marker,
    kernel_identity,
    record_installation,
    require_restarted_session,
    stale_imports,
)


def other_kernel(identity):
    return dict(identity, pid=identity["pid"] + 1)


def test_default_marker_sits_outside_the_repository_checkout():
    assert default_marker() == Path("/content/nt-colab-install.json")
    assert default_marker(Path("/other")) == Path("/other/nt-colab-install.json")


def test_kernel_identity_reports_this_process_and_is_stable():
    identity = kernel_identity()
    assert identity["pid"] == os.getpid()
    assert set(identity) == {"pid", "started", "boot"}
    assert kernel_identity() == identity


def test_recorded_installation_in_this_live_session_demands_a_restart(tmp_path):
    marker = tmp_path / "install.json"
    record_installation(marker)
    with pytest.raises(SessionRestartRequired, match="Restart"):
        require_restarted_session(marker)


def test_restarted_session_passes_and_reports_the_recorded_installation(tmp_path):
    marker = tmp_path / "install.json"
    record_installation(marker, identity=other_kernel(kernel_identity()))
    report = require_restarted_session(marker)
    assert report["installation_recorded"] is True
    assert report["restarted"] is True
    assert report["stale_imports"] == {}
    assert report["kernel"] == kernel_identity()


def test_absent_marker_does_not_block_but_records_the_unknown_installation(tmp_path):
    report = require_restarted_session(tmp_path / "missing.json")
    assert report["installation_recorded"] is False
    assert report["restarted"] is None
    assert report["stale_imports"] == {}


@pytest.mark.parametrize(
    "content", ["{not json", '{"no": "kernel"}', '{"kernel": "not a mapping"}']
)
def test_unreadable_marker_stops_instead_of_guessing(tmp_path, content):
    marker = tmp_path / "install.json"
    marker.write_text(content, encoding="utf-8")
    with pytest.raises(RuntimeError, match="nt-colab install marker"):
        require_restarted_session(marker)


def test_watched_distributions_stay_a_subset_of_the_pinned_recipe(tmp_path):
    requirements = Path(__file__).parents[1] / "requirements-nt.txt"
    pinned = {
        line.split("==")[0].strip().lower()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    from cas12a.nt_session import WATCHED

    assert {name.lower() for name in WATCHED} <= pinned


def test_marker_records_the_kernel_and_a_pending_restart(tmp_path):
    marker = tmp_path / "install.json"
    written = record_installation(marker)
    assert written == marker
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    assert recorded["kernel"] == kernel_identity()
    assert recorded["restart_required"] is True
    assert recorded["recorded_utc"].endswith("+00:00")


def test_an_interrupted_installation_still_demands_a_restart(tmp_path):
    # The marker is written before packages are replaced, so a failure midway
    # cannot leave the session looking untouched.
    marker = tmp_path / "install.json"
    begin_installation(marker)
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    assert recorded["installation_completed"] is False
    with pytest.raises(SessionRestartRequired, match="Restart"):
        require_restarted_session(marker)


def test_loaded_module_older_than_the_installed_distribution_demands_a_restart(
    tmp_path, monkeypatch
):
    pytest.importorskip("numpy")
    monkeypatch.setitem(
        sys.modules, "numpy", SimpleNamespace(__version__="0.0.0-stale")
    )
    stale = stale_imports()
    assert stale["numpy"] == {"loaded": "0.0.0-stale", "installed": _installed("numpy")}
    with pytest.raises(SessionRestartRequired, match="numpy"):
        require_restarted_session(tmp_path / "missing.json")


def test_matching_loaded_and_installed_versions_are_not_stale():
    numpy = pytest.importorskip("numpy")  # Loaded, so the check is not vacuous.
    assert sys.modules["numpy"] is numpy
    assert "numpy" not in stale_imports()


def test_repeat_installation_after_a_restart_is_refused_as_harmful(tmp_path):
    marker = tmp_path / "install.json"
    record_installation(marker, identity=other_kernel(kernel_identity()))
    with pytest.raises(InstallationAlreadyCompleted, match="already installed"):
        begin_installation(marker)


def test_repeat_installation_before_a_restart_is_refused_and_still_wants_one(tmp_path):
    marker = tmp_path / "install.json"
    record_installation(marker)
    with pytest.raises(InstallationAlreadyCompleted, match="Restart"):
        begin_installation(marker)


def test_first_installation_and_an_explicitly_forced_repeat_proceed(tmp_path):
    marker = tmp_path / "install.json"
    assert begin_installation(marker) is None
    record_installation(marker)
    assert begin_installation(marker, force=True) is None


def _installed(name):
    import importlib.metadata

    return importlib.metadata.version(name)


def test_a_local_wheel_build_tag_is_not_treated_as_staleness(monkeypatch):
    # The Colab recipe installs torch==2.12.0+cu126, whose metadata is 2.12.0.
    pytest.importorskip("torch")
    installed = _installed("torch")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(__version__=f"{installed}+cu126")
    )
    assert "torch" not in stale_imports()
