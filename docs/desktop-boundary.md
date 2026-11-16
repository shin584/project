# 데스크톱 화면 호출 경계 (#178)

데스크톱 화면(JS)과 파이썬 사이의 **호출 경계**를 한 곳에 적는다. 부모 티켓은
[#177](https://github.com/shin584/project/issues/177)이다.

> **경계 변경에는 #177 작성자의 승인이 필요하다.**
> 함수 목록·인자, 조각 슬롯 이름, 이벤트 모양, 요청 ID·단일 작업자 규칙 중
> 하나라도 바꾸려면 먼저 #177에서 승인을 받고, 이 문서와 `desktop/api.py`,
> `tests/test_desktop_api.py`를 **같은 PR에서 함께** 고친다. 팀원들이 이 경계
> 위에서 각자 화면을 만들고 있어서, 말없이 바뀌면 서로의 작업이 깨진다.

실행 가능한 형태는 `desktop/api.py`, 가중치 없이 도는 가짜 구현은
`desktop/fake.py`에 있다.

## 구조

```
JS ──(pywebview js_api)──▶ DesktopApi ──▶ Dispatcher(작업자 1개) ──▶ Renderer
JS ◀──(evaluate_js)────── window.xaiDesktop.onEvent(event)
```

- **Renderer** — 계산하고 HTML 조각을 만드는 쪽. 가짜(`FakeRenderer`)와 진짜
  (#154 코어 facade 위에 올라갈 구현)가 같은 프로토콜을 따른다.
- **Dispatcher** — 요청 ID를 매기고, 계산을 작업자 스레드 하나에서만 돌리고,
  지난 요청의 진행률·결과를 버린다.
- **DesktopApi** — pywebview에 `js_api`로 넘기는 객체. JS가 부르는 것은 이것뿐이다.

파이썬이 HTML 조각을 만들고 JS는 끼워 넣기만 한다. CDN도, 로컬 HTTP 서버도
쓰지 않는다.

## 1. JS가 부르는 함수

`window.pywebview.api.<이름>`으로 부른다. 모든 함수는 **즉시**
`{"request_id": <정수>}`를 돌려주고, 결과는 반환값이 아니라 이벤트로 온다.

| 함수 | 하는 일 | 결과 슬롯 |
|---|---|---|
| `scan_sites(dna)` | 사이트 탐색 | `sites` |
| `predict(dna)` | 예측 | `best`, `variants`, `detail` |
| `explain(dna, site_key, head, method, params)` | XAI 한 기법 | `xai` |
| `save_csv(dna, view)` | CSV 저장 | `notice` |
| `cancel()` | 진행 중·대기 중 요청 취소 | 없음 |

- `dna` — 사용자가 넣은 81bp 서열 그대로. 검증은 파이썬이 한다 (실패는 `error` 이벤트).
- `site_key` — `"<Cas 종류>|<가닥>|<PAM 시작>"` (예: `"Cas12a|+|16"`).
  조각의 각 행에 `data-site-key` 속성으로 들어 있다.
- `head` — 설명 대상 헤드. Cas12a는 `"Cas12a"`, SpCas9은 변종 이름(예: `"SpCas9(WT)"`).
  조각의 각 행에 `data-head` 속성으로 들어 있다.
- `method` — `"ig"` | `"shap"` | `"rollout"`.
- `params` — `ig`: `{"steps": 정수}` · `shap`: `{"nsamples": 정수, "n_background": 정수}` ·
  `rollout`: `{}`. 빠진 키는 Streamlit 화면의 기본값(`XAI_DEFAULTS`)을 쓴다:
  **IG 32단계, SHAP 샘플 64 × 배경 4** (`app.py:456-461`). 이 숫자가 어긋나면
  #177 게이트 2의 수치 비교가 깨지므로, 바꾸려면 두 화면을 같이 바꾼다.
- `view` — CSV로 내보낼 보기 이름 (`"best"` | `"variants"` | `"detail"`).

`save_csv`는 저장 대화상자를 먼저 띄운다. 사용자가 닫으면 `{"request_id": null}`을
돌려주고 **아무 작업도 맡기지 않는다** — 진행 중인 계산도 건드리지 않는다.
`cancel()`도 `{"request_id": null}`이다.

## 2. 돌려주는 HTML 조각

결과 이벤트의 `fragments`는 `{슬롯 이름: HTML 문자열}`이다. 슬롯은 여섯 개뿐이고
(`SLOTS`), 그 밖의 이름을 쓰면 파이썬 쪽에서 오류가 난다.

| 슬롯 | 내용 |
|---|---|
| `sites` | 후보 사이트 표 |
| `best` | Best Candidates 보기 |
| `variants` | 변종 비교 보기 |
| `detail` | 전체 상세 보기 |
| `xai` | XAI 그림 하나 (`<figure>` + 인라인 `<svg>`) |
| `notice` | 짧은 알림 한 줄 (CSV 저장 결과 등) |

조각은 **자족적이어야 한다**: 바깥 주소(`http://`, `https://`), `<script>`,
`data:`가 아닌 `src`가 들어가면 안 된다. 차트는 인라인 SVG이거나 `data:` URI
이미지다. 인라인 SVG에는 `xmlns`를 넣지 않는다 (HTML 안에서는 필요 없고, 그
값 자체가 바깥 주소다).

JS가 할 일은 슬롯 이름에 맞는 자리에 조각을 끼워 넣는 것뿐이다.

### 표의 열

열 이름과 순서는 Streamlit 화면을 그대로 따라간다. #177 게이트 2가 두 화면을
**고정 입력 하나로 수치 비교**(CSV는 diff)하므로, 여기서 어긋나면 동등성 점검이
깨진다. 가이드 서열은 표시용으로 RNA(T→U)로 바꿔 넣는다.

| 보기 | 열 | 출처 |
|---|---|---|
| Best Candidates | 순위 · 최적 Cas 변종 · PAM · 가이드 서열 (RNA) · 가닥 · 거리 (bp) · 예측 점수 (%) · 원출력 | `app.py:221-229` |
| 변종 비교 | 가이드 × 변종 **피벗** (평평한 목록이 아니다). 열은 그 입력에 나타난 변종만, `SPCAS9_VARIANTS` 순서 | `app.py:260-270` |
| 전체 상세 보기 | 순위 · Cas 변종 · PAM · 가이드 서열 (RNA) · 가닥 · 거리 (bp) · 원출력 · 예측 점수 (%) · 점수 기준 | `app.py:353-364` |

`예측 점수 (%)`의 뜻은 `점수 기준` 열이 정한다 (`절단 효율` 또는
`학습 분포 백분위`). **기준이 다른 행끼리는 점수를 직접 비교할 수 없다.**

## 3. 진행률·취소 이벤트

파이썬이 `window.xaiDesktop.onEvent(event)`를 부른다. 화면은 이 함수 하나만
정의하면 된다. 모든 이벤트에 `type`과 `request_id`가 있다.

```js
{"type": "progress", "request_id": 7, "done": 12, "total": 32}
{"type": "result",   "request_id": 7, "fragments": {"xai": "<figure>…</figure>"}}
{"type": "error",    "request_id": 7, "html": "<p class=\"error\">…</p>"}
```

- `progress` — `done`은 뒤로 가지 않고, 끝나면 `done == total`에 닿는다.
  Rollout처럼 순전파 1회로 끝나는 계산은 진행률을 보내지 않는다.
- `result` — 요청당 최대 한 번. 취소되면 오지 않는다.
- `error` — 입력 검증 실패 등. `html`은 이미 이스케이프돼 있다.

취소는 진행률 콜백이 `Cancelled` 예외를 던지는 방식으로 전해진다 (ADR 0001의
코어 API를 바꾸지 않는다). **모델 로딩과 Rollout은 중간에 취소하지 않는다** —
`_ensure_loaded()`는 끊을 수 없고 Rollout은 짧다. 이 구간은 끝까지 진행하고
결과만 버린다.

## 4. 요청 ID와 단일 작업자

pywebview는 노출된 함수를 **호출마다 다른 스레드에서** 실행한다. 이 규칙이
경계에 없으면 IG 두 개가 겹쳐 돈다. IG는 토큰 마스크·스팬·길이를 **예측기
인스턴스에** 저장하므로(`predict.py`의 `ig_inputs`가 `self._ig_mask`,
`self._ig_spans`, `self._ig_len`에 쓴다), 뒤의 계산이 앞의 마스크를 덮어써
**두 결과가 모두 틀린다.** 성능이 아니라 정확성의 문제다.

규칙은 셋이다.

1. 요청마다 번호가 하나씩 올라간다. 가장 큰 번호가 '지금 요청'이다.
2. 계산은 작업자 스레드 **하나**에서만 돈다. 새 요청이 오면 진행 중인 요청은
   취소된다.
3. 지금 요청이 아닌 것의 진행률·결과·오류는 화면에 보내지 않는다. 그래서
   **화면에 닿는 요청 번호는 절대 뒤로 가지 않는다.**

끝내 취소되지 않고 끝까지 돌아버린 작업(모델 로딩·Rollout)의 결과도 3번에서
버려진다. 각 요청의 끝은 `Dispatcher.outcome(request_id)`로 볼 수 있다.

| 값 | 뜻 |
|---|---|
| `delivered` | 결과를 화면에 보냈다 |
| `cancelled` | 돌던 중 취소됐다 |
| `superseded` | 시작도 하기 전에 다음 요청에 밀렸다 |
| `stale` | 끝까지 돌았지만 지금 요청이 아니라 결과를 버렸다 |
| `failed` | 오류를 화면에 보냈다 |
| `None` | 아직 진행 중 |

## 5. 가짜 구현으로 화면 만들기

`FakeRenderer`는 **모델 가중치 없이** 돈다. 가짜인 것은 점수와 기여도뿐이고,
서열에서 결정적으로 만든 값이라 같은 입력이면 늘 같은 그림이 나온다.
**설명으로 읽으면 안 된다.**

사이트 탐색은 가짜가 아니다. `pam_scanner`를 그대로 import해 쓴다 — 그 모듈은
`math`와 `dataclasses`만 import하므로 가중치도 torch도 끌고 오지 않는다. 탐색
규칙을 베껴 두면 `pam_scanner`가 바뀔 때 조용히 어긋나고, 팀원들은 존재하지 않는
`site_key`에 화면을 묶게 된다. 그래서 형제 체크아웃 `xai_dashboard`(브랜치
`dashboard_core`)가 있어야 `desktop.fake`를 import할 수 있다.

```python
import webview
from desktop.api import DesktopApi, Dispatcher, webview_emitter
from desktop.fake import FakeRenderer

window = webview.create_window("XAI", html=SCREEN_HTML)  # 로컬 서버 없이 문자열로
api = DesktopApi(
    FakeRenderer(step_delay=0.05),
    Dispatcher(webview_emitter(window)),
    ask_save_path=lambda name: window.create_file_dialog(
        webview.SAVE_DIALOG, save_filename=name
    ),
)
window.expose(api.scan_sites, api.predict, api.explain, api.save_csv, api.cancel)
webview.start(gui="edgechromium", http_server=False)
```

`step_delay`는 한 단계에 쉬는 시간이다. 진행률과 취소가 눈에 보이게 하려고 둔
것이니, 화면을 만들면서 적당히 올려 쓰면 된다.

## 6. 아직 경계에 없는 것 (#177 승인이 필요한 결정)

#178은 함수 목록을 네 가지(사이트 탐색·예측·XAI 계산·CSV 저장)로 못박았다.
아래는 하위 티켓이 곧 필요로 하지만 **아직 경계에 넣지 않은 것들**이다. 임의로
넣으면 그것 자체가 승인 없는 경계 변경이 되므로, 여기 적어 두고 #177에서 정한다.

| 빠진 것 | 누가 막히나 | 메모 |
|---|---|---|
| **모델 준비 완료 신호** | 하위 티켓 4 (시작 화면: "모델 준비 중", 버튼 비활성) | 지금 경계에는 준비 상태를 묻거나 알리는 길이 없다. 단일 작업자에 태우면 뒤 요청에 밀려 폐기될 수 있어서, 작업자 밖의 별도 이벤트로 둘지 정해야 한다 |
| **XAI 그래프 크기** | 하위 티켓 7 (XAI 보기) | Streamlit에는 슬라이더가 있다(`app.py:134`). 파이썬이 그림을 그리므로 크기도 `explain`의 인자여야 한다. 지금은 없다. #177은 "그래프 크기를 바꿔도 재계산하지 않아야 한다"고 못박았으므로, 캐시 키에는 들어가면 안 된다 |
| **차트를 `data:` URI로** | 하위 티켓 7 | #177은 차트를 data URI로 끼운다고 적었다. 가짜는 인라인 SVG만 내보내므로 그 경로가 한 번도 돌지 않는다. matplotlib 그림을 쓰는 진짜 구현에서 처음 돈다 |
| **`save_csv`의 보기별 저장** | — | Streamlit의 CSV는 **전체 상세 한 장뿐**이다(`app.py:385`). 경계는 `view`를 받게 해 두었지만, 세 보기를 각각 내보낼지는 정해지지 않았다. 지금은 어느 `view`든 전체 상세의 열로 내보내고 행만 거른다 |

## 시험

```powershell
$env:PYTHONUTF8 = '1'
python -m pytest tests -q
```

가중치·torch·pywebview 없이 돈다. 취소와 요청 ID 폐기가 실제로 지켜지는지
보려고, 경계의 규칙을 일부러 깨면 이 중 다섯 개가 깨지는 것을 확인했다.
