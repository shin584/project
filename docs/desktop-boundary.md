# Dashboard v3 화면 호출 경계 (#194)

v3 화면(JS)과 파이썬 사이의 **호출 경계**다. 부모 티켓은
[#191](https://github.com/shin584/project/issues/191)이다.

기존 #177 프로그램과 독립된 v3 경계다. 함수·슬롯은 계승하고 모든 이벤트에
`generation`을 추가한다. 변경할 때 이 문서, `desktop/api.py`,
`desktop/computation.py`, 실행 가능한 가짜 구현과 해당 검증을 함께 갱신한다.
기존 프로그램이나 완료된 이슈의 계약은 변경하지 않는다.

실행 가능한 형태는 `desktop/api.py`, 가중치 없이 도는 가짜 구현은
`desktop/fake.py`에 있다.

## 구조

```
JS ──(pywebview js_api)──▶ DesktopApi ──▶ ProcessDispatcher(스레드 1개)
                                              │ JSON pipe
                                              ▼
                                 계산 프로세스 1개 ──▶ Renderer
JS ◀──(evaluate_js)────── window.xaiDesktop.onEvent(event)
```

- **Renderer** — 계산 프로세스에서 예측·XAI·PNG/HTML을 만드는 쪽.
  `ComputationRenderer`와 가짜 `FakeRenderer`가 같은 프로토콜을 따른다.
- **ProcessDispatcher** — 요청 ID·프로세스 세대를 매기고 직렬화한다.
  이전 요청의 진행·결과와 이전 세대의 모든 이벤트를 버린다.
  `Dispatcher`는 모델 없는 인프로세스 단위 검증에도 사용하는 스케줄러다.
- **DesktopApi** — pywebview에 `js_api`로 넘기는 객체. JS가 부르는 것은 이것뿐이다.

파이썬이 HTML 조각을 만들고 JS는 끼워 넣기만 한다. CDN도, 로컬 HTTP 서버도
쓰지 않는다.

선택 전 PNG/HTML과 숫자 배열 전송 비교, JSON 메시지 형식·복사 버퍼·프로세스
종료·캐시 동일성·수명은 [계산 프로세스 결정](computation-process.md)에 있다.
UI import 경로는 `desktop.renderer`, `desktop.fake`나 모델 런타임을 가져오지 않는다.

## 1. JS가 부르는 함수

`window.pywebview.api.<이름>`으로 부른다. 모든 함수는 **즉시**
`{"request_id": <정수>}`를 돌려주고, 결과는 반환값이 아니라 이벤트로 온다.

| 함수 | 하는 일 | 결과 |
|---|---|---|
| `prepare()` | 모델 올리기 | `ready` 이벤트 |
| `select_family(family)` | 두 Cas의 계열 전환 후 준비 | 단계별 `progress`와 `ready` 이벤트 |
| `scan_sites(dna)` | 사이트 탐색 | 조각 `sites` |
| `predict(dna)` | 예측 | 조각 `best`, `variants`, `detail` |
| `explain(dna, site_key, head, method, params, size)` | XAI 한 기법 | 조각 `xai` |
| `save_context()` | 저장 대화상자의 세대·취소 상태 고정 | `{generation, revision}` |
| `save_csv(dna, filters, generation, revision)` | CSV 저장 | 조각 `notice` |
| `cancel()` | 진행 중·대기 중 요청 취소 | 없음 |

- `dna` — 사용자가 넣은 81bp 서열 그대로. 검증은 파이썬이 한다 (실패는 `error` 이벤트).
- `family` — `"non_nt"` 또는 `"nt"`. 데스크톱은 매 실행 `non_nt`로 시작한다.
  전환은 이전 세대 무효화 → 자식 종료 확인 → 선택 계열 준비 순서다.
  반환값에는 `request_id`, `generation`, `family`, `changed`가 있다.
  같은 계열은 기존 자식과 캐시를 유지하고 준비 완료 신호만 다시 보낸다.
  다른 계산/준비 호출의 반환값에도 `generation`·`family`를 포함한다.
- `site_key` — `"<Cas 종류>|<가닥>|<PAM 시작>"` (예: `"Cas12a|+|16"`).
  조각의 각 행에 `data-site-key` 속성으로 들어 있다.
- `head` — 설명 대상 헤드. Cas12a는 `"Cas12a"`, SpCas9은 변종 이름(예: `"SpCas9(WT)"`).
  조각의 각 행에 `data-head` 속성으로 들어 있다.
- `method` — `"ig"` | `"shap"` | `"rollout"`.
- `params` — `ig`: `{"steps": 정수}` · `shap`: `{"nsamples": 정수, "n_background": 정수}` ·
  `rollout`: `{}`. 빠진 키는 Streamlit 화면의 기본값(`XAI_DEFAULTS`)을 쓴다:
  **IG 32단계, SHAP 샘플 64 × 배경 4** (`app.py:456-461`). 이 숫자가 어긋나면
  #177 게이트 2의 수치 비교가 깨지므로, 바꾸려면 두 화면을 같이 바꾼다.
- `size` — XAI 그래프 크기(%). 40~100, 기본 80 (`app.py:134`의 슬라이더와 같다).
  범위를 벗어나면 `error` 이벤트가 간다.

`params`와 `size`를 **따로 받는 것이 핵심이다.** `params`는 계산 설정이라 XAI
캐시 키에 들어가고, `size`는 그리기 설정이라 들어가지 않는다. 섞으면 슬라이더를
움직일 때마다 SHAP이 다시 돌아 기능이 퇴행한다(#177). 크기는 그림 자체를 바꾸지
않고 바깥 `<figure>`의 너비만 바꾸므로, 같은 계산이면 같은 data URI가 나온다.

`save_csv`는 저장 대화상자를 먼저 띄운다. 사용자가 닫으면 `{"request_id": null}`을
돌려주고 **아무 작업도 맡기지 않는다** — 진행 중인 계산도 건드리지 않는다.
CSV는 Streamlit 화면과 같이 **전체 상세 한 장뿐**이고, 파일 이름도 같다
(`crispr_prediction_results.csv`). `cancel()`도 `{"request_id": null}`이다.
대화상자를 열었던 세대가 바뀌면 저장 요청을 폐기한다. 계산 중 전환·취소된 CSV도
파일 쓰기 직전에 세대·요청 ID를 검사해 게시하지 않는다.
화면은 저장 버튼을 눌렀던 `generation`을 함께 보내며, 이미 이전 세대라면
대화상자를 열지 않는다. 이 인자를 생략한 기존 호출은 API 진입 시점의 세대를 쓴다.
JS는 저장 대화상자 응답을 기다리는 동안에도 전환·취소 호출을 전달한다.
대화상자 중 취소나 새 계산이 생긴 경우도 제출 시점의 요청 변경 여부를 검사해
저장을 폐기한다.
먼저 직렬 호출 큐에서 빠른 `save_context()` 응답을 기다려 `revision`을 고정한 뒤
대화상자 호출만 분리한다. 취소·새 분석이 저장 API보다 먼저 Python에 도착하더라도
오래된 저장이 새 요청의 revision을 채택할 수 없다. `save_context`는 작업을 제출하지
않으며 대화상자 취소도 기존처럼 진행 중 계산을 취소하지 않는다.

- `filters` — '전체 상세 보기'가 **검색해 둔** 조건. 생략하면 전부 통과다(= 화면의
  기본값). `{"cas_names": [이름...], "eff_range": [최소, 최대], "dist_range": [최소, 최대]}`
  이고, 범위는 양끝을 포함하며 화면의 슬라이더 한계(점수 0~100, 거리 0~15) 밖이면
  `error` 이벤트가 간다. Streamlit의 다운로드 버튼이 **거른 표**를 내려주므로
  (`app.py:385-389`) 데스크톱도 같은 조건을 다시 적용하고 순위를 1부터 다시 매긴다.
  화면이 거른 행 대신 조건만 보내는 이유는 양쪽이 같은 술어를 쓰게 하려는 것이다
  (`desktop/detail_view.py`의 `DetailFilter`). 화면은 각 행의 `data-cas-name`·
  `data-eff`·`data-dist`로 같은 비교를 하고, 뒤의 둘은 **반올림 전** 값이라
  경계에서도 결과가 갈라지지 않는다. 필터가 보는 것은 `data-cas-name`이지
  `data-head`가 아니다 — 지금은 값이 같지만 헤드는 모델 출력의 인덱스이고
  `cas_name`은 예측 행의 이름이다 (`CONTEXT.md`).

상세 행의 `data-raw`는 반올림 전 원출력이다. v3 검증은 이 값과 `data-eff`를
보존된 고정 사례의 원출력·표시 점수와 비교한다 (`rtol=1e-5`, `atol=1e-8`).

저장한 파일은 UTF-8 BOM으로 시작하고 줄 끝은 CRLF다 (`desktop/api.py`의
`csv_bytes`). 둘 다 `os.linesep`에 맡기지 않고 못 박았다 — #181 기준 파일은 고정된
산출물이고 게이트 2가 바이트로 비교한다.

#196이 맨 끝에 `모델`·`모델 버전` 두 열을 더했으므로, 저장한 파일 **전체**가
Streamlit의 다운로드와 바이트까지 같지는 않다. 바이트 비교의 대상은 **기존
9열**(`desktop/detail_view.py`의 `BASE_COLUMNS`)이다. 자르는 자리는
`detail_view.base_columns_only` 하나가 정하고, 보존된 기준과의 대조는
`detail_view.csv_against_baseline`이 한다 — `desktop/validation.py`,
`docs/equivalence-baseline/compare_baseline.py`, `tests/test_nt_process.py`가
그 함수를 함께 쓴다. 보고 항목은 `csv_base_columns_byte_equal`(게이트가 보는 값),
`csv_whole_file_byte_equal`, `csv_model_columns`다. 기존 열의 **값·순서와 필터·정렬
의미는 바뀌지 않았다.**

`prepare()`는 창을 띄운 직후에 부른다. 화면은 "모델 준비 중"을 보여주다가
`ready`를 받고 분석 버튼을 연다. 여러 번 불러도 안전하다 — 로딩은 한 번만
일어나고 그 뒤의 호출은 `ready`만 다시 보내므로, 화면은 언제 물어도 답을 받는다.

## 2. 돌려주는 HTML 조각

결과 이벤트의 `fragments`는 `{슬롯 이름: HTML 문자열}`이다. 슬롯은 여섯 개뿐이고
(`SLOTS`), 그 밖의 이름을 쓰면 파이썬 쪽에서 오류가 난다.

| 슬롯 | 내용 |
|---|---|
| `sites` | 후보 사이트 표 |
| `best` | Best Candidates 보기 |
| `variants` | 변종 비교 보기 |
| `detail` | 전체 상세 보기 |
| `xai` | XAI 그림 (`<figure>` + PNG/SVG data URI 이미지) |
| `notice` | 짧은 알림 한 줄 (CSV 저장 결과 등) |

조각은 **자족적이어야 한다**: 바깥 주소(`http://`, `https://`), `<script>`,
`data:`가 아닌 `src`가 들어가면 안 된다.

차트는 **언제나 data URI 하나**다 (#177). 모양은 이것으로 고정한다:

```html
<figure class="xai" style="width:80%">
  <img src="data:image/png;base64,…" alt="…">
  <figcaption>…</figcaption>
</figure>
```

진짜 구현은 matplotlib PNG를 넣고 가짜는 SVG를 넣지만, 화면이 보는 모양은 같아서
둘을 구별할 필요가 없다. `style`의 너비가 `size`다. 인라인 `<svg>`를 조각에 직접
넣지 않는다 — 두 가지 방식이 섞이면 화면이 둘 다 다뤄야 한다.

data URI 안에 SVG를 넣을 때는 `xmlns`가 **있어야 한다**. 인라인이 아니라 독립
문서이기 때문이고, base64로 감싸이므로 조각 본문에는 그 주소가 드러나지 않는다.

JS가 할 일은 슬롯 이름에 맞는 자리에 조각을 끼워 넣는 것뿐이다.

**예외 하나: '전체 상세 보기'의 검색이다** (#185). 거기서 JS는 조각이 이미 들고 온
행을 **숨기고 번호만 다시 매긴다** — HTML을 새로 만들지도, 값을 계산하지도 않는다.
검색마다 파이썬에 다녀오면 예측을 통째로 다시 돌려야 하기 때문이다. 대신 거르는
기준은 양쪽에 똑같이 있어야 해서, 저장할 때는 화면이 **거른 행이 아니라 조건**을
보내고 파이썬이 같은 술어로 다시 거른다 (§1의 `filters`).

### 표의 열

열 이름과 순서는 Streamlit 화면을 그대로 따라간다. #177 게이트 2가 두 화면을
**고정 입력 하나로 수치 비교**(CSV는 diff)하므로, 여기서 어긋나면 동등성 점검이
깨진다. 가이드 서열은 표시용으로 RNA(T→U)로 바꿔 넣는다.

| 보기 | 열 | 출처 |
|---|---|---|
| Best Candidates | 순위 · 최적 Cas 변종 · PAM · 가이드 서열 (RNA) · 가닥 · 거리 (bp) · 예측 점수 (%) · 원출력 | `app.py:221-229` |
| 변종 비교 | 가이드 × 변종 **피벗** (평평한 목록이 아니다). 열은 그 입력에 나타난 변종만, `SPCAS9_VARIANTS` 순서 | `app.py:260-270` |
| 전체 상세 보기 | 순위 · Cas 변종 · PAM · 가이드 서열 (RNA) · 가닥 · 거리 (bp) · 원출력 · 예측 점수 (%) · 점수 기준 · **모델 · 모델 버전** | `app.py:353-364` + #196 |

`예측 점수 (%)`의 뜻은 `점수 기준` 열이 정한다 (`절단 효율` 또는
`학습 분포 백분위`). **기준이 다른 행끼리는 점수를 직접 비교할 수 없다.**

끝의 두 열은 #196이 더했다. 값은 예측 행이 들고 온 모델 식별에서 나오며
(`CONTEXT.md`의 '모델 식별'), **행마다** 그 행을 낸 모델을 가리킨다 — 한 표에 Cas가
섞이면 섞인 대로 맞다. 예측기를 거치지 않은 행(미리보기)은 모델을 주장하지 않고
`detail_view.UNKNOWN_MODEL`을 적는다.

## 3. 진행률·취소 이벤트

파이썬이 `window.xaiDesktop.onEvent(event)`를 부른다. 화면은 이 함수 하나만
정의하면 된다. 모든 이벤트에 `type`, `request_id`, `generation`이 있다.

```js
{"type": "progress", "generation": 1, "request_id": 7, "done": 12, "total": 32}
{"type": "result", "generation": 1, "request_id": 7, "fragments": {"xai": "<figure>…</figure>"}}
{"type": "ready", "generation": 1, "request_id": 1, "html": "<p class=\"status\">모델 준비 완료</p>"}
{"type": "error", "generation": 1, "request_id": 7, "html": "<p class=\"error\">…</p>"}
{"type": "restarting", "reason": "cancel_grace", "generation": 2, "family": "non_nt", "request_id": 9}
```

- `progress` — 배치 계산에서 `done`은 뒤로 가지 않고, 끝나면 `done == total`에 닿는다.
  Rollout처럼 순전파 1회로 끝나는 계산은 진행률을 보내지 않는다. 계열을 전환할 때는
  이전 프로세스 종료 확인 후 `done=1, total=3`, 새 프로세스 시작 후 `done=2, total=3`을
  보낸다. 화면은 전환 시작을 0%, `ready`를 100%로 표시한다.
- `result` — 요청당 최대 한 번. 취소되면 오지 않는다.
- `ready` — `prepare()` 또는 `select_family()`가 끝났다. 화면은 다른 이벤트와
  마찬가지로 현재 요청 ID·세대·계열과 일치할 때만 분석 버튼을 연다.
- `error` — 입력 검증 실패 등. `html`은 이미 이스케이프돼 있다.
- `restarting` — 강제 재시작이 시작됐다 (#218). `generation`은 새 세대이고
  `request_id`는 그 세대의 준비 작업이다. 화면의 현재 요청 ID와 상관없이 받으며,
  화면은 세대를 갱신하고 "계산 프로세스를 다시 시작하는 중..."을 보인 채 그
  준비의 `ready`까지 분석 버튼을 잠근다. 이전 세대의 요청은 응답하지 않는다.
  `reason`은 `"cancel_grace"`(취소에 응답 없음) 또는 `"no_progress"`(무진행 한도,
  #219)다.

취소는 진행률 콜백이 `Cancelled` 예외를 던지는 방식으로 전해진다 (ADR 0001의
코어 API를 바꾸지 않는다). **모델 로딩과 Rollout은 중간에 취소하지 않는다** —
`_ensure_loaded()`는 끊을 수 없고 Rollout은 짧다. 이 구간은 끝까지 진행하고
결과만 버린다.

자식 종료/통신 단절은 오류로 보고하고 다음 제출에서 새 세대로 재시작한다.

**취소 유예**(#218): 취소 또는 새 요청으로 진행 중인 요청이 접히면, 부모는 다음
진행률을 기다리지 않고 곧바로 자식에 취소를 알린다. 자식이 유예
(`computation.CANCEL_GRACE`, 53초, `ProcessDispatcher(cancel_grace=)`로 주입)
안에 그 요청으로 `cancelled`·`progress` 등 아무 응답도 하지 않으면 작업자 스레드가
**같은 계열로 강제 재시작**한다: 이전 세대 무효화(뒤에 줄 선 요청 포함) → 멈춘 자식
종료 확인 → 새 세대의 준비 작업을 줄 세움 → `restarting` 이벤트(그 준비의 `ready`보다
항상 먼저). 모델을 다시 적재하고 XAI 메모리 캐시를 잃는다. 응답이 오면 유예는 그 시점부터 다시 잰다. 준비 작업은 취소되지
않으므로 유예가 적용되지 않는다. 끊지 않는 Rollout은 정상 배치 하나로 보아 유예 안에
끝나야 한다 (실제 NT에서 취소 후 0.47초, #218 검증). 파이썬은 아무 요청도 다시 돌리지
않는다 — 화면이 재시작 당시 기다리던 분석·설명을 `ready` 뒤 다시 요청하고, 기다리던
CSV 저장은 거부 오류로 보인다. 재시작 중에는 CSV 저장도 받지 않는다.

**무진행 한도**(#219): 아무도 취소하지 않은 요청이 제출 또는 마지막 진행률 보고 뒤
한도(`computation.NO_PROGRESS_LIMIT`, 105초, `ProcessDispatcher(no_progress_limit=)`로
주입) 동안 아무것도 보내지 않으면 **멈춘 계산**으로 보고 위와 같은 강제 재시작을
한다(`reason: "no_progress"`). 진행률이 올 때마다 다시 재므로 작업 전체 시간에는 상한이
없다 — Cas12a SHAP처럼 오래 걸려도 보고가 이어지면 걸리지 않는다. Rollout처럼 진행률이
없는 계산은 결과까지의 시간이 한도 안이어야 한다. non-NT는 `warm_up`이 아니라 첫
예측 안에서 모델을 적재하므로(실제 16.9초 무보고) 그 적재 시간도 이 한도가 덮는다. 화면은 "계산이 응답하지 않아 계산
프로세스를 다시 시작했습니다" 오류를 띄우고 `ready`까지 분석을 잠근다. 멈춘 요청은
**화면도 다시 요청하지 않는다**(결정적으로 멈추는 입력이 재시작을 끝없이 반복하지
않도록). 그 뒤 `모델 준비 다시 시도`도 사용자가 새로 분석을 시작하기 전까지 분석을
다시 돌리지 않는다. 설명이 멈췄으면 설명 패널의 `다시 계산` 버튼만 보인다.

**적재 한도**(#219): 준비 작업(`warm_up`, 강제 재시작 뒤의 재적재 포함)은 적재 도중
진행률이 없으므로 전체 시간에 한도(`computation.LOAD_LIMIT`, 165초,
`ProcessDispatcher(load_limit=)`로 주입)를 둔다. 넘으면 자식을 종료하고 준비 오류
("모델 적재가 적재 한도(…초)를 넘겨 … 모델 준비를 다시 시도하세요")를 보낸다. 화면은 기존 준비 오류처럼 `모델 준비
다시 시도`(`retry-prepare`)를 보이고, 그 재시도가 종료된 자식을 같은 계열의 새 세대로
교체한다.

**세 한도의 확정값**(#220): 개발 PC에서 실제 모델로 오프라인 측정해 정했다. 전시장
PC는 개발 PC와 사양·환경이 거의 같으므로 같은 값을 쓴다. 코드 상수이며 설정 파일·
환경변수로 노출하지 않는다(테스트·프로브만 `ProcessDispatcher` 인자로 주입).

- 측정 방법: `tests/limit_measurement_probe.py`가 세 한도를 닿지 않는 값으로 주입하고
  계열(NT·non-NT)마다 새 자식 2개의 `warm_up` 전체 시간, 각 작업의 제출→첫 진행률·
  진행률 사이·마지막 진행률→결과 간격, 첫 진행률 직후 취소했을 때 `cancelled`까지의
  시간을 잰다. 작업은 예측(cold·warm·사이트 밀집 입력), CSV(캐시 없음·있음), IG/SHAP
  (화면 기본값과 화면 최대값 IG 256·SHAP 1024/32, Cas12a·SpCas9 헤드 각 1개), NT
  Rollout이다. 기록: `.packaging/issue220-limits/run01/report.json`.
- 측정 범위: 2026-11-22 run01은 3라운드 예정이었으나 라운드 1의 NT 최대 SHAP(Cas12a)이
  프로브의 대기 한도 3600초를 넘겨(한도 발동 아님, `restarting` 0건) 중단됐다. 라운드
  0 전체와 라운드 1의 non-NT 전체, NT의 그 작업 전까지를 썼다. 빠진 것은 라운드 1 NT의
  최대 SHAP(Cas12a)·Rollout(Cas12a)·취소·두 번째 적재와 라운드 2다. 측정·무진행 프로브의
  작업당 대기는 4시간으로 늘렸다.
  개발 PC(8 GiB, 여유 약 3 GiB)의 메모리 압박으로 라운드 간 시간이 최대 2배쯤 달랐다.
- 측정값(최대): 적재 NT 54.6초 · non-NT 1.9초. 무보고 간격은 cold 첫 예측의 첫 구간이
  가장 길다 — NT 35.1초 · non-NT 27.8초(지연 적재). 그 밖에는 IG 7.3초(마지막 진행률→
  결과) · SHAP 9.3초(NT 최대값) · Rollout 0.8초 · warm 예측·CSV 0.8초 이하. 첫 진행률
  뒤 취소는 NT 2.0초 · non-NT 0.14초 안에 `cancelled`, 진행 중 Rollout 취소는 0.15초.
  작업 전체 시간은 화면 최대값 SHAP이 가장 길다 — NT Cas12a 3374초(라운드 1은 3600초
  초과) · NT SpCas9 1855–2300초 · non-NT 26–66초. 진행률은 계속 오므로 무진행 한도와
  무관하다.
- 최대 배치: 취소는 진행률 보고에서만 받으므로, 끊을 수 없는 최장 구간인 cold 첫
  예측의 첫 구간 35.1초를 최대 배치로 본다. 이것을 빼면 재시작 직후 첫 분석을 취소할
  때 정상 취소가 강제 재시작으로 번진다. (프로브의 "cold 첫 예측 즉시 취소"는 요청이
  자식에 닿기 전 대기 중에 접혀 `superseded` 0초로 끝나 이 구간을 재지 못했다. 그래서
  첫 구간 간격으로 대신한다.)
- 배수와 확정값: 무진행 한도 = 35.1초 × 3 ≈ **105초**, 적재 한도 = 54.6초 × 3 ≈
  **165초**, 취소 유예 = 35.1초 × 1.5 ≈ **53초**.
- 확정값 검증(주입 없음, 2026-11-22): `tests/cancel_grace_probe.py` 통과 — 예측·IG·SHAP·
  Rollout의 정상 협력 취소가 모두 `cancelled`(NT 최대 0.94초, non-NT 0.08초)로 끝나고
  같은 자식·세대 유지, 캐시 적중, `restarting` 0건, 남은 자식 0개
  (`.packaging/issue220-limits/cancel/report.json`). `tests/stall_limit_probe.py`는 개발
  PC 메모리 부족으로 중지되어 확정값으로는 **미검증**이다. 대신 run01에서 화면 최대값
  SHAP의 최대 무보고 간격(NT 9.3초)이 무진행 한도 105초의 1/10 아래이고, 측정 적재 최대
  54.6초가 적재 한도 165초 안임을 근거로 둔다. 메모리를 확보할 수 있으면
  `python -X utf8 tests/stall_limit_probe.py .packaging/issue220-limits/stall`로 확인한다.

#219 개발 PC 실측(오프라인, 화면 기본 설정, Cas12a 헤드, 2회 중 최대): 최대 무보고
간격은 NT 첫 예측 29.0초 · non-NT 첫 예측 16.9초(지연 적재) · IG 8.9초 · SHAP 1.9초 ·
Rollout 1.1초, NT `warm_up` 34.1초 · non-NT 0.5초. 실제 non-NT 자식을 SHAP 도중
정지시키면 주입한 5초 한도 뒤 5.1초에 재시작, 5.6초에 `ready`, 멈춘 요청은 다시
돌지 않고 종료 후 남은 자식 0개였다.

준비 오류는 창의 `모델 준비 다시 시도` 버튼으로 같은 창에서 재시도한다.
`ProcessDispatcher.restart()`는 이전 세대를 먼저 무효화하고 자식 종료를 확인한다.
`close()`도 자식을 종료·join하고 이후 제출을 거절한다. 모델/메모리 캐시 수명은
자식 프로세스 수명과 같고, 출처가 확인되지 않은 디스크 SHAP 캐시는 사용하지 않는다.

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

끝내 취소되지 않고 끝까지 돌아버린 작업(Rollout 등)의 결과도 3번에서 버려진다.

준비 작업은 스케줄러에서 생애 주기 작업으로 분리해 같은 작업자에서 우선 실행한다.
스케줄러의 최신 계산 ID 관문 밖이지만 프로세스 세대 관문은 적용한다. #198의
화면은 준비 중 계산을 제출하지 않고, 준비 신호도 현재 요청 ID로 검사한다.
따라서 교체 전 준비 신호나 이전 준비 재시도의 신호가 버튼을 열 수 없다.
JS는 호출 응답 전에 도착한 이벤트를 버퍼링하고 응답의 ID·세대로 다시 검사한다.
전환 즉시 입력·창은 유지하고 네 보기, 설명 선택·이미지, 진행률, CSV 검색 상태를
비운다. 준비가 끝나면 현재 입력을 다시 탐색·예측한다.

각 요청의 끝은 `Dispatcher.outcome(request_id)`로 볼 수 있다.

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
`dataclasses`만 import하므로 가중치도 torch도 끌고 오지 않는다. 탐색
규칙을 베껴 두면 `pam_scanner`가 바뀔 때 조용히 어긋나고, 팀원들은 존재하지 않는
`site_key`에 화면을 묶게 된다. 코어는 현재 폴더의 `pam_scanner.py`를 사용하며,
형제 체크아웃 없이 `desktop.fake`를 import할 수 있다.

```powershell
python -m desktop --preview
```

가짜 구현도 `ProcessDispatcher(..., preview=True)`를 통해 별도 프로세스에서 돈다.
직접 실행 파일을 만들 때는 프로세스가 창을 재귀 생성하지 않도록
`if __name__ == "__main__":` 안에서 시작하고 frozen entry에서
`multiprocessing.freeze_support()`를 먼저 호출한다 (`desktop_entry.py`).

## 6. 결정 기록

2026-11-16, #177 작성자가 아래 네 가지를 승인해 경계에 넣었다. 초안에서는 빠져
있던 것들이고, 하위 티켓 4·7이 바로 이것들을 필요로 한다.

| 결정 | 어떻게 들어갔나 |
|---|---|
| **로딩 완료 이벤트를 추가** | `prepare()` + `ready` 이벤트. 요청 번호 관문 밖의 생애 주기 작업이다 (§4의 예외) |
| **그래프 크기 인자를 추가** | `explain`의 `size` 인자. `params`와 **따로** 받아 캐시 키에 들어가지 않게 했다 |
| **차트는 URI 방식으로 통일** | 인라인 SVG를 없애고 `<img src="data:…">` 하나로 고정했다 (§2) |
| **CSV는 단일 파일 저장** | `save_csv(dna)`에서 `view`를 뺐다. Streamlit과 같이 전체 상세 한 장이다 |

2026-11-16, #185 작업 중 아래 하나를 더 승인받아 넣었다.

| 결정 | 어떻게 들어갔나 |
|---|---|
| **CSV에 필터 조건을 전달** | `save_csv(dna, filters)`와 `Renderer.csv_table(dna, on_batch, filters)`에 **선택** 인자를 더했다. 빠뜨리면 전부 통과라 기존 호출은 그대로 돈다. Streamlit이 **거른 표**를 내려주는데 `save_csv(dna)`만으로는 전체 행밖에 저장할 수 없어, 기본 필터가 아닌 상태에서 게이트 2의 CSV diff가 깨진다 |

위 #177 기록은 계승한 경계의 역사다. v3 변경은 #191/#194 범위에서 독립적으로
진행하며 `tests/test_computation_process.py`가 세대·종료·재시작 경계를 검증한다.

## 시험

```powershell
$env:PYTHONUTF8 = '1'
python -m pytest tests/test_desktop_api.py tests/test_computation_process.py -q
```

가중치·torch·pywebview 없이 돈다. 실제 NT/PNG/CSV 고정 사례는 별도 실행한다:
`python -X utf8 -m pytest tests/test_nt_process.py -q -s`.
취소와 요청 ID 폐기가 실제로 지켜지는지
보려고, 경계의 규칙을 일부러 깨면 이 중 다섯 개가 깨지는 것을 확인했다.
