"""#183: public shell behavior with real WebView2 and no model weights."""

import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows desktop shell")


def test_shell_startup_inputs_views_and_no_http_server(tmp_path):
    pytest.importorskip("webview")
    pytest.importorskip("psutil")
    from desktop.instance import SingleInstance

    with SingleInstance() as instance:
        if not instance.acquired:
            pytest.skip("Close the running desktop to test the real application window")
    output = tmp_path / "shell.json"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "tests/desktop_shell_probe.py"), str(output)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "error" not in report, report.get("error")
    assert report["html_string"] and report["url"] is None
    assert report["shown_before_prepare"]
    assert report["initial"] == {"status": "모델 준비 중", "disabled": True}
    assert report["ready_status"] == "모델 준비 완료"
    assert report["sidebar"] == {
        "labels": ["81bp DNA 서열", "XAI 그래프 크기"],
        "button": "분석 시작",
        "graph": ["40", "100", "5", "80"],
    }
    assert report["duplicate_exit"] == 0, report["duplicate_error"]
    assert report["validation_error"]
    views = report["views"]
    assert [view["label"] for view in views] == [
        "Best Candidates", "변종 비교", "전체 상세 보기", "XAI 설명"
    ]
    for view, panel in zip(views, ["best", "variants", "detail", "xai"]):
        assert view["selected"] == "true"
        assert view["visible"] == ["view-" + panel]
        marker = "data:image/" if panel == "xai" else "<table"
        assert marker in view["content"]
    assert report["resize"] == {"sameImage": True, "width": "40%", "label": "40%"}
    assert report["keyboard_view"] == "Best Candidates"
    assert report["listeners"] == []
    assert not report["server_created"]
    assert report.get("network_attempts", []) == []


@pytest.mark.parametrize("abrupt", [False, True], ids=["normal-exit", "killed"])
def test_instance_can_restart_after_owner_exits(tmp_path, abrupt):
    ready = tmp_path / "ready"
    instance_name = "Local\\XaiDashboardDesktopTest-" + uuid4().hex
    owner = subprocess.Popen(
        [sys.executable, "-c", """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from desktop.instance import SingleInstance
with SingleInstance(sys.argv[3]) as guard:
    assert guard.acquired
    Path(sys.argv[2]).write_text('ready')
    sys.stdin.read(1)
""", str(ROOT), str(ready), instance_name],
        cwd=tmp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Owner did not acquire the instance guard"
        from desktop.instance import SingleInstance

        with SingleInstance(instance_name) as duplicate:
            assert not duplicate.acquired
        if abrupt:
            owner.kill()
        stdout, stderr = owner.communicate(input=b"x" if not abrupt else None, timeout=10)
        if not abrupt:
            assert owner.returncode == 0, (stdout, stderr)
        with SingleInstance(instance_name) as restarted:
            assert restarted.acquired
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.communicate(timeout=10)
