"""데스크톱 화면 ↔ 파이썬 호출 경계 (#178).

전체 명세(함수 목록, 조각 슬롯, 이벤트 모양, 요청 ID·단일 작업자 규칙, 변경 절차)는
`docs/desktop-boundary.md`에 있다. 이 파일은 그 명세의 실행 가능한 형태다.

    JS ──(pywebview js_api)──▶ DesktopApi ──▶ Dispatcher(작업자 1개) ──▶ Renderer
    JS ◀──(evaluate_js)────── window.xaiDesktop.onEvent(event)

경계를 바꾸려면 #177 작성자의 승인이 필요하다. 팀원들이 이 경계 위에서 각자
화면을 만들고 있으므로, 말없이 바꾸면 서로의 작업이 깨진다.
"""

from __future__ import annotations

import csv
import html
import itertools
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

#: 조각이 들어갈 수 있는 화면의 자리. 이 밖의 이름은 화면이 받지 않는다.
SLOTS = ("sites", "best", "variants", "detail", "xai", "notice")

#: XAI 기법별 기본 계산 설정. Streamlit 화면의 기본값과 같다 (`app.py:456-461`).
XAI_DEFAULTS: dict[str, dict[str, int]] = {
    "ig": {"steps": 32},
    "shap": {"nsamples": 64, "n_background": 4},
    "rollout": {},
}

OnBatch = Callable[[int, int], None]
Fragments = dict[str, str]

#: 요청 하나가 어떻게 끝났는지. `Dispatcher.outcome()`이 돌려준다.
Outcome = Literal["delivered", "cancelled", "superseded", "stale", "failed"]


class Cancelled(Exception):
    """진행 중인 계산을 접으라는 신호. 진행률 콜백이 던진다."""


class Renderer(Protocol):
    """계산하고 HTML 조각을 만드는 쪽. 가짜와 진짜가 이 모양을 함께 따른다.

    진짜 구현은 #154 코어 facade 위에 올라가, 코어가 낸 숫자를 조각으로 바꾼다.
    (`xai_dashboard/CONTEXT.md`의 '코어'가 _Avoid_로 적은 "백엔드"를 피해 부른다.)

    모든 메서드는 `on_batch(done, total)`을 받아 진행률을 보고한다. 그 호출이
    `Cancelled`를 던지면 즉시 손을 떼야 한다 — 다만 모델 로딩·Rollout처럼 중간에
    끊을 수 없는 구간은 끝까지 진행해도 된다. 지난 요청의 결과는 Dispatcher가
    요청 ID로 버린다.
    """

    def sites_fragment(self, dna: str, on_batch: OnBatch) -> str: ...

    def prediction_fragments(self, dna: str, on_batch: OnBatch) -> Fragments: ...

    def explanation_fragment(
        self,
        dna: str,
        site_key: str,
        head: str,
        method: str,
        params: dict[str, Any],
        on_batch: OnBatch,
    ) -> str: ...

    def csv_table(self, dna: str, view: str, on_batch: OnBatch) -> list[list[str]]: ...


# ════════════════════════════════════════════════
# 요청 ID와 단일 작업자
# ════════════════════════════════════════════════


class Dispatcher:
    """계산을 작업자 스레드 **하나**에서만 돌리고, 지난 요청의 것은 버린다.

    pywebview는 노출된 함수를 호출마다 다른 스레드에서 실행한다. 그래서 이 규칙이
    경계에 없으면 IG 두 개가 겹쳐 돌 수 있고, IG는 토큰 마스크·스팬·길이를
    예측기 인스턴스에 저장하므로(`predict.py`의 `ig_inputs`) 뒤의 계산이 앞의
    마스크를 덮어써 **두 결과가 모두 틀린다.** 성능이 아니라 정확성의 문제다.

    규칙
      - 요청마다 번호가 하나씩 올라간다. 가장 큰 번호가 '지금 요청'이다.
      - 새 요청이 오면 진행 중인 요청은 취소된다 (진행률 콜백이 `Cancelled`).
      - 지금 요청이 아닌 것의 진행률·결과·오류는 화면에 보내지 않는다. 그래서
        화면에 닿는 번호는 절대 뒤로 가지 않는다.
    """

    def __init__(self, emit: Callable[[dict], None]):
        self._emit_event = emit
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._ids = itertools.count(1)
        self._latest = 0
        self._pending: tuple[int, Callable] | None = None
        self._running: int | None = None
        self._outcomes: dict[int, str] = {}
        self._closed = False
        self._worker = threading.Thread(
            target=self._serve, name="xai-desktop-worker", daemon=True
        )
        self._worker.start()

    # ── 화면 쪽에서 부르는 것 ──────────────────────

    def submit(self, job: Callable[[OnBatch], Fragments]) -> int:
        """작업을 맡기고 요청 ID를 돌려준다. 앞의 요청은 이 시점에 취소된다."""
        with self._lock:
            rid = next(self._ids)
            if self._pending is not None:
                self._outcomes.setdefault(self._pending[0], "superseded")
            self._pending = (rid, job)
            self._latest = rid  # 진행 중인 작업의 다음 on_batch가 취소된다
            self._cv.notify_all()
        return rid

    def cancel(self) -> None:
        """진행 중·대기 중인 요청을 모두 접는다. 새 작업은 맡기지 않는다."""
        with self._lock:
            if self._pending is not None:
                self._outcomes.setdefault(self._pending[0], "superseded")
                self._pending = None
            self._latest = next(self._ids)
            self._cv.notify_all()

    # ── 시험·종료용 ────────────────────────────────

    def outcome(self, request_id: int) -> Outcome | None:
        """요청이 어떻게 끝났는지. 아직 진행 중이면 `None`. 값은 `Outcome` 참고."""
        with self._lock:
            return self._outcomes.get(request_id)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        with self._lock:
            return self._cv.wait_for(
                lambda: self._pending is None and self._running is None, timeout
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending = None
            self._latest = next(self._ids)
            self._cv.notify_all()
        self._worker.join(timeout=5)

    # ── 작업자 ─────────────────────────────────────

    def _serve(self) -> None:
        while True:
            with self._lock:
                self._cv.wait_for(lambda: self._pending is not None or self._closed)
                if self._pending is None:
                    if self._closed:
                        return
                    continue
                rid, job = self._pending
                self._pending = None
                if rid != self._latest:
                    self._outcomes.setdefault(rid, "superseded")
                    self._cv.notify_all()
                    continue
                self._running = rid
            try:
                result = self._execute(rid, job)
            except Exception:  # noqa: BLE001 -- 작업자는 무슨 일이 있어도 살아남는다
                # 여기 닿는 것은 이벤트 전송 자체가 터진 경우뿐이다 (창이 닫혀
                # evaluate_js가 실패하는 등). 작업자가 죽으면 이후 모든 요청이
                # 조용히 멈추므로, 이 요청만 실패로 적고 계속 돈다.
                result = "failed"
            with self._lock:
                self._running = None
                self._outcomes[rid] = result
                self._cv.notify_all()

    def _execute(self, rid: int, job: Callable[[OnBatch], Fragments]) -> Outcome:
        def on_batch(done: int, total: int) -> None:
            # 확인과 전송을 같은 잠금 안에서 한다. 그래야 늦게 끝난 지난 작업의
            # 진행률이 새 작업의 이벤트 뒤에 끼어들지 못한다.
            with self._lock:
                if rid != self._latest:
                    raise Cancelled(rid)
                self._emit_event(
                    {
                        "type": "progress",
                        "request_id": rid,
                        "done": int(done),
                        "total": int(total),
                    }
                )

        try:
            # 조각 자리 확인도 try 안에서 한다. 밖에서 하면 Renderer의 실수가
            # 작업자를 죽인다 — 화면에 오류로 보이는 편이 낫다.
            fragments = _checked_slots(job(on_batch))
        except Cancelled:
            return "cancelled"
        except Exception as exc:  # noqa: BLE001 -- 화면 경계: 오류도 조각으로 전한다
            with self._lock:
                if rid != self._latest:
                    return "stale"
                self._emit_event(
                    {"type": "error", "request_id": rid, "html": _error_html(exc)}
                )
            return "failed"

        with self._lock:
            if rid != self._latest:
                return "stale"
            self._emit_event(
                {"type": "result", "request_id": rid, "fragments": fragments}
            )
        return "delivered"


def _checked_slots(fragments: Fragments) -> Fragments:
    unknown = set(fragments) - set(SLOTS)
    if unknown:
        raise KeyError(f"화면에 없는 조각 자리: {sorted(unknown)}")
    return fragments


def _error_html(exc: Exception) -> str:
    return f'<p class="error">{html.escape(str(exc) or exc.__class__.__name__)}</p>'


# ════════════════════════════════════════════════
# JS가 부르는 함수
# ════════════════════════════════════════════════


class DesktopApi:
    """pywebview에 `js_api`로 넘기는 객체. JS가 부르는 것은 이 메서드들뿐이다.

    모든 메서드는 **즉시** `{"request_id": int}`을 돌려주고, 결과는 이벤트로 간다.
    `save_csv`는 저장 대화상자를 닫으면 `{"request_id": None}`을 돌려주고 아무
    작업도 맡기지 않는다. `cancel`도 `{"request_id": None}`이다.
    """

    def __init__(
        self,
        renderer: Renderer,
        dispatcher: Dispatcher,
        ask_save_path: Callable[[str], str | None] | None = None,
    ):
        self._renderer = renderer
        self._dispatcher = dispatcher
        self._ask_save_path = ask_save_path or (lambda name: None)

    def scan_sites(self, dna: str) -> dict:
        return self._submit(
            lambda on_batch: {"sites": self._renderer.sites_fragment(dna, on_batch)}
        )

    def predict(self, dna: str) -> dict:
        return self._submit(
            lambda on_batch: self._renderer.prediction_fragments(dna, on_batch)
        )

    def explain(
        self,
        dna: str,
        site_key: str,
        head: str,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict:
        settings = dict(XAI_DEFAULTS.get(method, {}))
        settings.update(params or {})

        def job(on_batch: OnBatch) -> Fragments:
            if method not in XAI_DEFAULTS:
                raise ValueError(f"모르는 XAI 기법: {method}")
            return {
                "xai": self._renderer.explanation_fragment(
                    dna, site_key, head, method, settings, on_batch
                )
            }

        return self._submit(job)

    def save_csv(self, dna: str, view: str) -> dict:
        # 저장 대화상자는 계산이 아니라 사용자 선택이므로 작업자에 맡기지 않는다.
        # 닫으면 요청 자체가 생기지 않아 진행 중인 계산도 건드리지 않는다.
        chosen = self._ask_save_path(f"{view}.csv")
        if not chosen:
            return {"request_id": None}
        target = Path(chosen)

        def job(on_batch: OnBatch) -> Fragments:
            table = self._renderer.csv_table(dna, view, on_batch)
            with target.open("w", encoding="utf-8-sig", newline="") as handle:
                csv.writer(handle).writerows(table)
            return {
                "notice": '<p class="notice">저장했습니다: '
                f"{html.escape(str(target))}</p>"
            }

        return self._submit(job)

    def cancel(self) -> dict:
        self._dispatcher.cancel()
        return {"request_id": None}

    def _submit(self, job: Callable[[OnBatch], Fragments]) -> dict:
        return {"request_id": self._dispatcher.submit(job)}


def webview_emitter(window) -> Callable[[dict], None]:
    """이벤트를 창의 JS로 보내는 싱크. 화면은 `window.xaiDesktop.onEvent` 하나만 둔다."""

    def emit(event: dict) -> None:
        window.evaluate_js(
            "window.xaiDesktop.onEvent(" + json.dumps(event, ensure_ascii=False) + ");"
        )

    return emit
