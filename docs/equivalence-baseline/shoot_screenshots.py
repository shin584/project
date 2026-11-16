"""#181 동등성 기준 — Streamlit 화면 4개 보기의 스크린샷을 찍는다.

게이트 2의 체크리스트는 이 그림들을 기준으로 문구·표 열·색 규칙·차트·필터·CSV·
XAI 3기법·계산 설정·그래프 크기 슬라이더를 항목별로 대조한다.

필요한 것
  · 가중치와 토크나이저 (실제 점수가 찍혀야 기준이 된다)
  · playwright + Edge:  python -m pip install playwright
    브라우저는 설치된 Edge를 쓴다 (`channel="msedge"`). 별도 다운로드가 없다.

사용법
    python docs/equivalence-baseline/shoot_screenshots.py

직접 띄워 둔 Streamlit에 붙이려면:
    python docs/equivalence-baseline/shoot_screenshots.py --url http://localhost:8501

결과는 `docs/equivalence-baseline/screenshots/` 아래 PNG 4~5장이다.
무엇이 보여야 하는지는 같은 폴더의 CHECKLIST.md에 있다.

playwright를 쓸 수 없으면 CHECKLIST.md를 보고 손으로 찍어 같은 이름으로 넣는다.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "screenshots"

sys.path.insert(0, str(HERE))
from capture_baseline import FIXED_INPUT  # noqa: E402  같은 고정 입력을 쓴다

VIEWS = ["Best Candidates", "변종 비교", "전체 상세 보기", "XAI 설명"]
FILES = {
    "Best Candidates": "01-best-candidates.png",
    "변종 비교": "02-variant-heatmap.png",
    "전체 상세 보기": "03-detail.png",
    "XAI 설명": "04-xai.png",
}
VIEWPORT_W, VIEWPORT_H = 1600, 1200
MAX_SHOT_H = 9000          # 본문이 길어도 뷰포트를 이보다 더 늘리지 않는다

# 모델 로드 + 예측은 CPU에서 몇 분 걸린다.
PREDICT_TIMEOUT_MS = 20 * 60 * 1000
XAI_TIMEOUT_MS = 30 * 60 * 1000


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout_s: float) -> bool:
    import urllib.error
    import urllib.request
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.URLError:
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return False


def start_streamlit() -> tuple[subprocess.Popen, str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-m", "streamlit", "run", "app.py",
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    url = f"http://localhost:{port}"
    if not _wait_http(url, 120):
        proc.terminate()
        raise RuntimeError("Streamlit이 2분 안에 뜨지 않았다")
    return proc, url


def _snap(page, path: Path) -> None:
    """본문 전체를 찍는다.

    `full_page=True`는 Streamlit에서 뷰포트만 찍힌다 — 스크롤이 <body>가 아니라
    안쪽 컨테이너(stMain)에 걸려 있어 문서 높이가 뷰포트 높이와 같기 때문이다.
    본문만 요소로 찍으면 사이드바가 빠지므로(체크리스트는 매 장에 사이드바를 요구한다),
    본문 높이에 맞춰 뷰포트를 늘린 뒤 찍는다.
    """
    height = page.evaluate(
        "() => { const m = document.querySelector('[data-testid=\"stMain\"]');"
        " return m ? m.scrollHeight : document.body.scrollHeight; }")
    height = int(min(max(height + 80, VIEWPORT_H), MAX_SHOT_H))
    page.set_viewport_size({"width": VIEWPORT_W, "height": height})
    page.wait_for_timeout(1200)          # 폭이 그대로라 표·차트는 다시 그리지 않는다
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size({"width": VIEWPORT_W, "height": VIEWPORT_H})
    page.wait_for_timeout(500)


def _view_radio(page, name: str):
    """보기 전환 라디오의 선택지.

    Streamlit의 라디오 <input>은 숨겨져 있어(`visually hidden`) 역할로는 못 누른다.
    보이는 라벨 텍스트를 누른다. 본문에도 '전체 상세 보기' 같은 문구가 있으므로
    반드시 라디오 위젯 안쪽으로 범위를 좁힌다 (첫 stRadio가 보기 전환이다).
    """
    return page.get_by_test_id("stRadio").nth(0).get_by_text(name, exact=True)


def _method_radio(page, name: str):
    """XAI 기법 라디오. 'Attention Rollout'을 가진 라디오 위젯으로 찾는다."""
    return (page.get_by_test_id("stRadio")
            .filter(has_text="Attention Rollout")
            .get_by_text(name, exact=True))


def shoot(url: str) -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
                                device_scale_factor=1)
        page.goto(url, wait_until="networkidle", timeout=120_000)

        # 사이드바에 고정 입력을 넣고 분석을 시작한다.
        # 라벨로 찾으면 Streamlit이 위젯을 다시 그리는 사이에 놓칠 때가 있어
        # 사이드바 안의 입력칸을 직접 잡는다.
        sidebar = page.get_by_test_id("stSidebar")
        box = sidebar.locator("input[type='text']").first
        box.wait_for(timeout=120_000)
        box.fill(FIXED_INPUT)
        sidebar.get_by_role("button", name="분석 시작").click()

        # 예측이 끝나면 보기 전환 라디오가 나타난다.
        _view_radio(page, "Best Candidates").wait_for(timeout=PREDICT_TIMEOUT_MS)
        page.wait_for_timeout(2000)

        for view in VIEWS:
            _view_radio(page, view).click()
            page.wait_for_timeout(1500)

            if view == "전체 상세 보기":
                # 필터 기본값 그대로 검색해 표·CSV 버튼까지 보이게 한다.
                page.get_by_role("button", name="검색").click()
                page.wait_for_timeout(2500)

            if view == "XAI 설명":
                # '계산 설정'을 펼쳐 IG 단계·SHAP 샘플/배경 입력칸을 보이게 한다.
                try:
                    page.get_by_text("계산 설정", exact=True).click()
                    page.wait_for_timeout(500)
                except Exception:
                    pass
                # 기본 선택은 Attention Rollout (추론 1회)
                page.wait_for_timeout(3000)

            _snap(page, OUT / FILES[view])
            print(f"  {FILES[view]}")

        # XAI 3기법이 모두 기준에 있어야 한다. IG·SHAP도 각각 찍는다.
        for label, fname, done_text in (
                ("Integrated Gradients (IG)", "05-xai-ig.png",
                 "글자별 기여도 (Attribution Logo, IG)"),
                ("SHAP (Shapley Additive exPlanations)", "06-xai-shap.png",
                 "위치별 기여분 (SHAP)")):
            _method_radio(page, label).click()
            # 계산이 끝나야 그 기법의 제목이 나온다 (SHAP은 사전계산 캐시에 적중한다).
            page.get_by_text(done_text, exact=True).wait_for(timeout=XAI_TIMEOUT_MS)
            page.wait_for_timeout(5000)
            _snap(page, OUT / fname)
            print(f"  {fname}")

        browser.close()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="이미 떠 있는 Streamlit 주소. 없으면 직접 띄운다")
    args = ap.parse_args(argv[1:])

    try:
        import playwright  # noqa: F401
    except ImportError:
        print("playwright가 없다. `python -m pip install playwright` 후 다시 실행하거나,\n"
              "screenshots/CHECKLIST.md를 보고 손으로 찍어 같은 이름으로 넣는다.")
        return 2

    proc = None
    try:
        if args.url:
            url = args.url
        else:
            print("Streamlit 실행 중 ...")
            proc, url = start_streamlit()
        print(f"스크린샷 ({url}) — 모델 로드와 XAI 계산 때문에 수십 분 걸릴 수 있다")
        shoot(url)
    finally:
        if proc is not None:
            proc.terminate()
    print(f"완료 — {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
