# #198 — non-NT 기본 시작과 공통 계열 전환

[요구사항](https://github.com/shin584/project/issues/198)의 작업 위치는
`dashboard_v3` 워크트리/브랜치다. 작업 시작 기준은 `d3a8737`이며 기존
`xai_dashboard_desktop`과 상위 작업 폴더는 변경하지 않았다.

## 동작

데스크톱은 매 실행 non-NT로 시작한다. 공통 선택창 하나로 SpCas9과 Cas12a를
함께 바꾼다. 기존 코어/Streamlit의 NT 기본값은 유지하며 데스크톱 계산 소유자가
선택 계열의 렌더러를 명시한다. UI 프로세스는 모델·토크나이저·수치/차트 런타임을
import하지 않는다. 선택하지 않은 계열의 가중치·토크나이저도 준비하지 않는다.

전환 순서는 이전 세대/요청 무효화 → 이전 계산 프로세스 terminate/join 및 종료
확인 → 새 계열 렌더러 준비다. 창과 DNA 입력은 유지한다. 네 보기의 결과와 설명
이미지·선택, 진행 상태, CSV 검색 조건을 즉시 비우고 준비 후 현재 입력을 다시
탐색·예측한다. 선택창은 준비 중에도 사용할 수 있다. 화면은 다시 불러오는 상태와
모델에 따라 예측 점수·후보 순위가 달라질 수 있음을 안내한다. non-NT의 점수 안내는
두 Cas 모두 절단 효율로 표시한다.

준비 완료를 포함한 모든 이벤트는 활성 요청 ID·세대·계열로 검사한다. 호출 응답보다
먼저 도착한 이벤트도 버퍼링 후 같은 검사를 거친다. 요청 ID와 세대·계열의 반환은
제출 잠금 안에서 함께 만든다. 저장 대화상자는 열었던 세대에 귀속되며, 실제 CSV
쓰기 역시 요청/세대 무효화 잠금 안에서 검사한다. JS는 저장 대화상자의 응답을
기다리지 않고 전환·취소를 전달한다. 화면에서 전달한 세대가 이미 지난 경우
대화상자를 열지 않으며, 대화상자 중 취소·새 계산이 생겨도 저장을 폐기한다.
대화상자 호출을 분리하기 전에 직렬 큐에서 `save_context`로 요청 변경 상태를
고정하므로 취소·새 분석의 Python 호출이 앞질러 도착하는 경우도 이전 저장을 거부한다.
이전 계열에서 시작한 저장이
다음 계열로 넘어가거나 뒤늦게 파일을 쓰지 않는다.

같은 계열을 다시 고르면 화면은 호출을 보내지 않는다. API로 재선택해도 기존 자식과
예측기·설명 캐시는 유지한다. 탭과 그래프 크기는 화면에서 처리한다. 계열을 바꾸면
모든 계산 캐시는 자식과 함께 사라지고, 같은 계열 안의 checkpoint·설명 프로토콜
식별은 기존 검증된 캐시 키를 사용한다.

non-NT 설명 계산 연결은 #199/#200이다. 이 작업은 기존 예측·선택을 유지하며 화면에
설명 계산 준비 중임을 표시한다. 기존 NT 기준을 검사하는 `--validate`와 실제 NT
프로세스 시험은 이제 NT를 명시적으로 선택한다.

## 검증 범위와 명령

`AGENTS.md`에 따라 공개 API/프로세스·화면의 수용 기준과 직접 영향받는 캐시·보기
경계만 선택했다. #194가 문서화한 API/렌더러 경계를 재사용했다. 저장 대화상자
전환 사례는 먼저 실패를 확인하고 수정 후 통과했다.

```powershell
python -m pytest tests/test_family_switch.py tests/test_computation_process.py tests/test_desktop_api.py tests/test_desktop_shell.py -q
```

56 통과 / 0 실패 / 0 건너뜀. 준비·예측·설명·CSV 계산 중 전환, 빠른 연속 선택,
같은 계열 재선택, 잘못된 계열 거부, 이전 자식의 실제 종료, 종료 시 자식 정리,
취소·새 요청·계산/준비 실패 후 재시도, 오래된 네 종류 이벤트 폐기, 입력 유지,
화면 비우기, 저장 대화상자가 열린 동안 전환 완료·이전 파일 미변경, 대화상자 중
취소, 오래된 보기의 저장 거부 및 실제 WebView2 창의 네 보기·CSV를 검사했다.

```powershell
python -m pytest tests/test_worker_cache.py test/test_cache_identity.py test/test_explain_cache.py tests/test_prediction_views.py tests/test_detail_view.py tests/test_desktop_xai.py -q
```

128 통과 / 0 실패 / 0 건너뜀. 기존 보기와 CSV, 설명 캐시·checkpoint/프로토콜
구분, 기법 재방문과 크기 변경의 캐시 유지에 대한 회귀다. 작은 고정 선형 모델을
쓰는 기존 SHAP 시험에서 특이행렬 경고 3건이 발생했다.

```powershell
python -m pytest tests/test_non_nt_process.py::test_the_non_nt_path_loads_no_nt_weights_or_tokenizer -q
```

1 통과 / 0 실패 / 0 건너뜀. 실제 non-NT 예측 후 NT 토크나이저 캐시가 비어 있고
`transformers`·`safetensors`가 import되지 않았으며 NT 예측기가 생성되지 않았음을
별도 프로세스에서 확인했다. 메모리 영향 때문에 실제 모델 실행과 가벼운 회귀는
별도 Python 프로세스로 실행했다. 저장소 전체 스위트는 실행하지 않았다.

실제 창 검증 명령:

```powershell
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
python -X utf8 tests/family_window_probe.py .packaging/evidence/issue198-accepted
```

이 검증은 실제 두 모델/계열로 non-NT → NT → non-NT를 선택해 현재 DNA를
재분석하고 네 보기와 CSV를 확인한다. NT CSV의 기존 열은 보존된 기준과 바이트
비교하며, non-NT 복귀 CSV는 첫 non-NT 실행과 바이트 비교한다. 매 전환 후 이전
PID의 종료, 최종 창 종료 시 자식 정리, UI의 모델 런타임 import 부재, 같은 계열
재선택·탭/크기 변경에 계산 요청이 발생하지 않음도 확인한다.

최종 실제 창 결과는 `docs/benchmarks/issue198-family-window.json`에 기록한다.
2026-11-19 개발 PC에서 **통과**했다. non-NT/NT/non-NT의 계산 PID는 각각
18900/23732/22208로 달랐고 이전 PID 종료 및 최종 자식 정리를 확인했다.
분석·보기·CSV·재선택 확인을 포함한 각 구간은 18.449/73.732/37.578초였다.
NT의 기존 CSV 열 일치와 non-NT 복귀 CSV 전체 바이트 일치가 모두 참이다.
보고서의 소스 SHA256도 최종 구현 파일과 대조했다. 위 pytest 합계는
**185 통과 / 0 실패 / 0 건너뜀**이며, 이 실제 창 검증은 별도로 통과했다.
개발 PC에서 실제 모델·창·프로세스 왕복을 검증하는 범위다. 전체 프로세스 트리
피크 RSS는 전환 직전 이전 계열의 사용량도 포함하므로 non-NT의 안정 상태 RAM이나
절약률로 읽지 않는다. 기존 NT 앱과의 정식 RAM 비교는 #203, 새 배포본과 깨끗한
오프라인 Windows 검증은 #204, non-NT 설명 연결은 #199/#200 범위로 남는다.

## 독립 코드 리뷰

`code-review` 스킬의 Standards와 Spec 두 축을 독립 에이전트가 검토했다.
비교 기준은 `d3a8737...33aa9be`이며 후속 수정도 Standards 에이전트가 확인했다.

### Standards

- 계약 위반 1건: 요청 제출 후 세대/계열을 읽으면 동시 전환에서 오래된 ID와 새
  세대 정보가 짝지어질 수 있었다. `ProcessDispatcher.submit_request`가 제출 잠금
  안에서 ID·세대·계열을 함께 반환하도록 수정했다.
- 판단 사항 1건: 검증 보고서의 `after_close_tree_rss_bytes`는 UI RSS만 측정했다.
  실제 측정 범위를 나타내는 `after_close_ui_rss_bytes`로 이름을 바로잡았다.

후속 확인: 두 수정 모두 확인됐고 남은 조치 사항은 없다.

### Spec

최초 비교에서는 확인된 지적 0건이었다. 후속 저장 대화상자 보완에서 P2 1건을
발견했다: 비동기 저장 호출보다 취소·새 분석의 Python 호출이 먼저 도착하면 저장이
새 요청 상태를 채택할 수 있었다. 직렬 큐에서 `save_context()` 응답을 받은 뒤
대화상자를 분리하고 그때의 revision을 검증하도록 수정했다. 두 도착 순서의 API
시험은 실패→통과를 확인했고, 리뷰어도 수정 후 남은 사항 0건으로 확인했다.
non-NT 기본값, 공통 선택창, 교체 순서, 입력 유지·자동 재분석, 화면/CSV 상태
비우기, 오래된 이벤트 거부와 캐시 유지에서 다른 누락은 없었다. 리뷰는 정적
검토이며 위 실행 검증과 구분한다.
