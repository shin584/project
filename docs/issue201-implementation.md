# #201 — NT XAI 단위 교정과 세 기법 검증

[#201](https://github.com/shin584/project/issues/201)의 2026-11-19 전시회 표시 방침을
적용했다. 구현과 검증 실행은 완료했으며, SHAP의 seed 간 안정성은 통과하지 않았다.
유한한 결과의 그래프 표시는 유지하고 품질 경고 배너·팝업은 추가하지 않았다.

작업 기준은 `dashboard_v3`, `0052cb076fe43bddb383717417618badcde3f142`다.
기존 프로그램·checkpoint와 #205 측정 자료는 보존했다. 같은 워크트리의 병행
#199 변경은 작업 트리에 유지하고 #201 커밋에서 분리했다.

## 변경

- SpCas9의 헤드별 학습 통계와 출처를 확인했다. 훈련 CSV 29,134행에서 float32
  타깃 ÷100의 `nanmean`·`nanstd(ddof=0)`을 재계산했으며 등록된 9개 평균·표준편차와
  fraction 단위 절대 오차 `1e-7` 안에서 일치했다. 헤드 이름 대응과 파일 해시는
  [출력 근거](nt-xai/output-provenance.json)와 [SpCas9 보고서](nt-xai/evidence/SpCas9.json)에 있다.
- 예측·기준값은 `(raw × std + mean) ×100`, 기여도·오차는 `raw_delta × std ×100`이다.
  Cas12a는 학습 데이터의 비율 출력 근거와 checkpoint SHA256 연결을 확인했다.
  원출력과 원기여도 배열을 보존하고 화면에서 단위를 바꾼다. 통계가 없으면 원출력
  단위로 표시한다. 표의 0~100% 제한과 설명의 제한 전 함수를 구분한다.
- IG는 서열 word embedding만 0으로 만들고 특수 토큰을 고정한다. 일반 input-id
  forward의 word-embedding 경계에 입력을 주입해 ESM 전처리를 그대로 거친다.
  설치된 Transformers에서 `inputs_embeds`가 전처리를 건너뛰던 불일치를 없앴다.
  공유 잠금으로 예측·Rollout과 격리하고 예외에도 임시 hook을 제거한다.
- 프로토콜을 `nt-affine-units-fixed-specials-esm-input-v2`로 바꿨다. 구 단위/경로의
  메모리·디스크 캐시를 재사용하지 않는다. SHAP 저장·조회는 raw 값과 표시 단위
  메타데이터를 함께 보존한다. 새 고정 사례 캐시는 검증 폴더에만 두었다.
- 기본 IG 32 / SHAP 64·배경 4개, 기존 설정 변경·진행률·취소를 유지했다. 정밀
  설정은 선택 기능이며 자동으로 기본값을 올리지 않는다. 계산 예외·결과 없음·
  NaN/inf는 오류로 처리한다. 유한한 수치 품질 실패는 그래프를 유지하고 내부
  `%LOCALAPPDATA%/dashboard-v3/logs/nt-xai.jsonl`과 검증 보고서에 기록한다.

## 실제 모델의 수치 결과

실행 전에 [contract.json](nt-xai/contract.json)의 서열·설정·허용 오차를 동결했다.
Cas마다 고정 윈도우 1개, SpCas9 9개 헤드와 Cas12a 1개 헤드를 검증했다. IG는
32·64·256단계, SHAP은 64/4·128/8에서 배경 seed와 부분집합 seed를 각각 바꿨다.
허용 오차와 실패 목록은 [분리 집계](nt-xai/evidence/summary.json)에 있다.

| 구분 | SpCas9 통과/실패/건너뜀 | Cas12a 통과/실패/건너뜀 |
|---|---:|---:|
| 수치 검사 | 334 / 39 / 0 | 37 / 8 / 0 |
| 그래프 표시 검사 | 27 / 0 / 0 | 3 / 0 / 0 |
| 출처·캐시 검사 | 12 / 0 / 0 | 2 / 0 / 0 |

전체는 **415 통과 / 47 실패 / 0 건너뜀**이다. 수치 검사만 보면 **371 통과 /
47 실패**다. 실패는 모두 seed 간 SHAP 위치 기여도 안정성 비교다. 기본 64/4는
40개 비교 중 35개, 128/8은 별도 40개 중 12개가 실패했다. 두 설정 모두 안정성이
검증된 설정으로 부르지 않는다. 동일 seed 고정과 가법성 통과를 안정성의 증거로
대신하지 않았다. 두 모델 검증 명령의 종료코드는 각각 1이며 실패를 통과로 바꾸지 않았다.

원함수 일치·유한값·IG 완결성/단계 안정성·SHAP 가법성/동일 seed 재현성·Rollout은
고정 범위에서 통과했다. SpCas9의 IG 완결성 최대 오차는 32/64/256단계 각각
0.025741 / 0.0033195 / 0.00018633%p, Cas12a는 0.0044393 / 0.0059910 /
0.00035868%p였다. IG와 일반 예측의 원출력 차이는 고정 10개 헤드 모두 0이었다.
보존된 고정 두 예측의 반올림 전 raw 값도 `atol=1e-8, rtol=1e-5`로 일치했다.

SHAP harness는 같은 서열의 실제 전방 출력 벡터를 헤드·seed 비교 사이에 메모한다.
따라서 동일 seed 반복 통과는 이 결정적 출력 위에서의 추정기 재현성이며, 모든
모델 추론을 독립적으로 다시 수행한 변동 측정이라는 뜻은 아니다.

## 실행 명령과 테스트

실제 실행 진입점은 다음과 같다. 로컬 계획 파일로 첫 실행은
`regression, SpCas9, Cas12a, process`, 최종 실행은 `audit, display, regression,
shared_ig`를 선택했다. 각 단계는 별도 Python 3.13 프로세스다.

```powershell
py -3.13 -X utf8 docs/nt-xai/run_validation.py
```

최종 pytest 범위는 아래 명령들이다. 합계 **193 통과 / 0 실패 / 0 건너뜀**이며
전체 저장소 스위트는 실행하지 않았다.

```powershell
py -3.13 -X utf8 -m pytest test/test_nt_explain.py test/test_predict.py test/test_explain.py test/test_explain_cache.py test/test_cache_identity.py test/test_shap_rng.py test/test_core_facade.py tests/test_desktop_xai.py tests/test_nt_cancellation.py tests/test_nt_xai_view.py tests/test_progress_callback.py tests/test_shap_cache.py tests/test_worker_cache.py -q -p no:cacheprovider
# 189 passed, 0 failed, 0 skipped, 89.48s
py -3.13 -X utf8 -m pytest test/test_cas12a_ig.py::test_selected_handle_explains_the_composed_model_through_the_existing_loop test/test_cas12a_ig.py::test_path_protocol_prevents_cross_path_memory_and_disk_cache_hits test/test_cas12a_ig.py::test_selected_path_keeps_progress_and_callback_cancellation -q -p no:cacheprovider
# 3 passed, 0 failed, 0 skipped, 28.33s
py -3.13 -X utf8 -m pytest tests/test_nt_process.py -q -p no:cacheprovider
# 1 passed, 0 failed, 0 skipped, 209.64s
```

첫 범위는 변경한 단위·실제 ESM 경로·캐시·렌더러와 직접 연결되는 기존 공개 경계다.
공용 IG 기준 계산을 바꿨으므로 Cas12a 복합/이중 입력·경로 캐시·취소의 기존 3개
케이스까지 확장했다. 작은 선형 시험 모델에서 기존 SHAP 특이행렬 경고 4건이
발생했다. 개발 중에는 작은 ESM fixture 자료형 오류로 1실패/92통과, 이후 원함수
불일치로 1실패/7통과를 확인했고, 수정 후 집중 범위는 8통과했다.

실제 모델·증거 검증 명령:

```powershell
py -3.13 -X utf8 docs/nt-xai/verify.py SpCas9
# 373 pass / 39 fail / 0 skip, 1415.943s, exit 1
py -3.13 -X utf8 docs/nt-xai/verify.py Cas12a
# 42 pass / 8 fail / 0 skip, 1012.993s, exit 1
py -3.13 -X utf8 docs/nt-xai/audit_evidence.py
# artifact/source/raw hashes, frozen policy, internal logs, fixed raw parity: pass, exit 0
py -3.13 -X utf8 docs/nt-xai/render_evidence.py
# final renderer over saved real arrays: 6 pass / 0 fail / 0 skip, exit 0
```

실제 process 시험에서 기존 CSV 열의 바이트 일치, 두 모델에서 세 기법의 화면,
크기 변경/재방문 캐시, 실제 진행률, UI의 계산 라이브러리 import 부재와 자식 종료를
확인했다. [process 증거](nt-xai/evidence/process.json)는 #194의 기존 시험을 #201에서
다시 실행한 원시 결과를 감싼 것이며 과거 결과를 재사용한 것이 아니다. 표시 성공은
수치 안정성 통과와 별개다. 최종 PNG는 `.packaging/evidence/issue201/final-display/`에
생성하고 IG·SHAP 워터폴·Rollout 이미지를 직접 열어 확인했다.

## 증거와 남은 범위

[validation.json](nt-xai/evidence/validation.json)에 실제 명령·종료코드·집계를 연결했다.
모델·계약·환경 식별, 보고서·raw 배열 SHA256 및 출처가 있는 새 캐시를 보존한다.
측정 당시 정확한 소스는 `nt-xai/evidence/measured-source.zip`에 있다. 측정 중
소스가 바뀌지 않았음을 양 모델에서 확인했다. 병행 #199의 공용 UI 추가분도 당시
스냅샷에는 포함되지만 NT에서는 빈 참고 문자열을 반환하는 분기다. 제품 변경
커밋은 #201에 해당하는 수정만 분리한다.

측정 후 변경은 미사용 UI 타입 import 제거, 차트 단위 설명 문구, 동일한 동결
숫자를 계약에서 읽도록 한 verifier 정리 및 후속 명시적 재실행의 소스 ZIP 자동
생성이다. 새 소스 스냅샷 생성은 임시 디렉터리에서 해시를 검증했다. 변경 전/후 해시를 summary에
기록했고 독립 리뷰가 값의 동등성을 확인했다. 모델 계산은 바뀌지 않았다.

SHAP 두 설정의 안정성 실패는 남아 있다. 고정 사례 외 서열 전체의 품질은 검증하지
않았다. 기존 NT Cas12a의 TTTV 0–3번과 PAM 색 4–7번 불일치는 보존된 한계이며,
이번 결과는 생물학적 구간 색/PAM별 기여도 검증 완료를 뜻하지 않는다.
새 배포본·깨끗한 Windows 오프라인 검증과 정식 RAM 비교는 #203/#204 범위다.
#191이나 다른 티켓을 자동 종료하지 않는다.

## Standards

초기 규칙 위반 1건: UI의 직접 `xai.units` import가 core facade 경계를 우회했다.
미사용 import와 기본 객체를 제거했다. 판단 사항 1건: 허용 오차 리터럴 중복을
계약 참조와 실행 정책 일치 검사로 정리했다. 후속 읽기 전용 검토에서 남은 지적 0건이다.

## Spec

최종 확인된 구현 결함 0건이다. 리뷰어는 계약·소스 ZIP·보고서·원시 배열의 해시,
표시/수치/출처 분리 집계, 사후 기준 완화가 없음을 독립 확인했다. SHAP 안정성
미달과 기존 Cas12a 구간 색 한계를 숨기지 않았으며, 최신 전시회 표시 방침과
일치한다고 확인했다. Standards 0건 / Spec 0건이 남아 있다.
