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
        check=False,
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
    assert not report["display_changes_sent_requests"]
    controls = report["xai_controls"]
    assert controls["siteLabel"] == "분석 대상 사이트 (효율 높은 순)"
    assert controls["headLabel"] == "설명할 출력(변종)"
    assert controls["methods"] == [
        "Attention Rollout", "Integrated Gradients (IG)",
        "SHAP (Shapley Additive exPlanations)",
    ]
    assert controls["settings"] == [["8", "256", "8", "32"], ["16", "1024", "16", "64"], ["2", "32", "2", "4"]]
    from core import scan_all
    from desktop.fake import heads_of

    dna = (
        "GATCCAGTCAGCATGCTTTACGATCGTACGATCGAATCGTAGG"
        "CTAGCATCGATCGATGCATCGTAGCTAGCTAGCATCGA"
    )
    sites = scan_all(dna).sites
    assert set(controls["sites"]) == {s.site_key for s in sites}
    assert report["sp_heads"] == heads_of(next(s for s in sites if s.site_key == "SpCas9|+|40"))
    assert 0 < report["shap_progress"]["value"] < 100
    assert "SHAP 계산 중..." in report["shap_progress"]["label"]
    assert "IG · SpCas9-NG · 40단계" in report["replacement_chart"]
    assert report["cancelled_ui"] == {
        "hidden": True, "notice": "계산을 취소했습니다.", "canAnalyze": True, "canRetry": True,
    }
    calls = [c for c in report["bridge_calls"] if c["name"] == "explain"]
    assert any(c["args"][3:5] == ["shap", {"nsamples": 80, "n_background": 6}] for c in calls)
    assert any(c["args"][3:5] == ["ig", {"steps": 40}] for c in calls)
    # #185: 전체 상세 보기 · 필터 · 검색 · CSV 저장
    from desktop.api import CSV_FILENAME, csv_bytes
    from desktop.detail_view import DETAIL_COLUMNS
    from desktop.fake import FakeRenderer

    def table(filters=None):
        return FakeRenderer(step_delay=0).csv_table(dna, lambda d, t: None, filters)

    full = table()[1:]
    before = report["detail_before_search"]
    assert before == {
        "hidden": True,  # Streamlit shows nothing until the search button
        "csvDisabled": True,
        "variants": sorted({cells[1] for cells in full}),
        "total": len(full),
    }
    unfiltered = report["detail_unfiltered"]
    assert unfiltered["shown"] == len(full)
    assert unfiltered["names"] == [cells[1] for cells in full]  # same order as Python
    assert unfiltered["count"] == f"검색 결과: {len(full)}개 / 전체 {len(full)}개"
    assert unfiltered["ranks"] == [str(i) for i in range(1, len(full) + 1)]
    assert not unfiltered["csvDisabled"]
    # The yellow rows are exactly the rows of the site the XAI tab has selected.
    selected_key = report["xai_controls"]["sites"][0]
    assert unfiltered["selected"] == [
        key for key in unfiltered["keys"] if key == selected_key
    ]
    assert unfiltered["selected"], "선택한 사이트의 줄이 칠해지지 않았다"
    assert unfiltered["highlightCount"] == str(len(unfiltered["selected"]))

    filtered = report["detail_filtered"]
    chosen = table({"cas_names": [filtered["variant"]], "dist_range": [0, 10]})
    assert set(filtered["names"]) == {filtered["variant"]}
    assert all(d <= 10 for d in filtered["distances"])
    assert filtered["ranks"] == [str(i) for i in range(1, len(chosen))]
    assert filtered["count"] == f"검색 결과: {len(chosen) - 1}개 / 전체 {len(full)}개"
    # The captions echo the live values, as Streamlit's do (app.py:331, 335).
    assert filtered["distCaption"] == "변이로부터 0 ~ 10 bp 이내"
    assert filtered["effCaption"] == "선택 범위: 0.0% ~ 100.0%"
    assert filtered["lowMax"] == "10"  # the pair cannot be crossed over

    # The saved file holds what the screen searched, byte for byte.
    assert report["save_dialog"] == CSV_FILENAME
    saved = bytes.fromhex(report["csv_hex"])
    assert saved == csv_bytes(chosen)
    assert saved.startswith(b"\xef\xbb\xbf")  # utf-8-sig, like the download button
    assert saved.decode("utf-8-sig").split("\r\n")[0].split(",") == DETAIL_COLUMNS

    assert report["detail_empty"] == {
        "csvDisabled": True,  # Streamlit draws no download button for an empty result
        "noMatch": True,
        "highlightHidden": True,  # nor the 노란 줄 caption beside the warning
        "tableHidden": True,
        "count": f"검색 결과: 0개 / 전체 {len(full)}개",
    }
    assert report["keyboard_view"] == "Best Candidates"
    assert report["stale_generation_ignored"]
    assert report["retry_visible_on_error"]
    assert report["retry_ready"] == "모델 준비 완료"
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
