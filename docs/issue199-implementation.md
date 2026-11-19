# #199 — non-NT SpCas9 IG·SHAP 연결과 기존 그래프 표시

작업 기준은 `dashboard_v3`의 #201 완료 커밋
`cf7f0c6cd8712ac3a0fae8df53c7bfdfee8b3601`이다. #199와 #201의 병행 작업 중
중단했던 변경을 이 커밋 위에서 마무리했다. 기존 프로그램과 #205 자료는 보존했다.

## 구현

- `NonNTSpCas9Predictor.explain_handle`이 채택된 Keras 3/torch 런타임의
  forward와 입력 gradient를 기존 `integrated_gradients`/`kernel_shap` 루프에 연결한다.
  같은 모델 인스턴스를 쓰고, 변종 이름을 동결된 9개 출력 층 순서에 대응시킨다.
- 설명의 계산 식별은 실제 적재 checkpoint SHA256, 계열, 설명 프로토콜,
  런타임 버전·backend·dtype·스레드 수다. 헤드·윈도우·단계/샘플/배경 수·두 seed는
  기존 결과 캐시 키에 포함된다. 크기는 계산 키에 포함되지 않는다. 프로토콜이나
  계산 설정이 바뀌면 재계산하고 같은 설정 재선택·크기 변경은 재사용한다.
- 기본값은 **IG 32 / SHAP 64샘플·배경 4개**다. IG 기준은 30×4 영벡터이며
  SHAP은 seed 0의 무작위 ACGT 배경과 부분집합 추출을 사용한다. 화면 기술적
  참고에 두 seed와 PAM도 교체될 수 있음을 표시한다. 설정을 높인 정밀 재계산은
  사용자가 선택하며, 기존 진행률·취소를 유지한다.
- SpCas9 원출력은 `published_percentage/100`이다. 근거는
  [#196 단위 확인](issue196-implementation.md)과 보관된 export manifest/
  test_eval 산출물이다. 원출력·기여도 배열은 그대로 보존하고, 화면은 제한 전
  예측에 ×100(%), 기여도에 ×100(%p)을 적용한다. 제한 전 출력이 0~1 밖이면 표의
  제한된 점수와 설명 대상이 다름을 안내한다. 모델 출력의 분해가 생물학적
  인과관계라는 뜻은 아님을 명시한다.
- non-NT Rollout은 기존 메뉴 위치에서 비활성화하고 attention 부재를 설명한다.
  미지원 Cas12a를 선택하면 이전 SpCas9 작업을 취소하고 윈도우·지표·그래프를
  지워 늦게 도착한 결과가 다른 사이트의 설명으로 표시되지 않게 한다.
  Cas12a 설명 연결은 #200 범위다.
- 공용 품질 기록에서 non-NT SpCas9의 완결성·가법성은 동결 절대 오차
  **0.005 raw = 0.5%p**를 사용한다. NT의 #201 기준 `0.5%p + 0.05×signal`은
  유지한다. 내부 기록은 `%LOCALAPPDATA%/dashboard-v3/logs/non_nt-xai.jsonl`과
  `nt-xai.jsonl`로 나누어 남긴다. non-NT의 길이 불일치와 NaN/inf는 오류다.
  유한한 품질 실패 결과는 기존 그래프로 표시하고 별도 경고를 추가하지 않는다.

## 실제 모델 수치 검증 — 실패 보존

[새 수치 보고서](benchmarks/issue199-xai-20261119-resumed01/report.json)는
`docs/runtime/contract.json`의 S1–S4와 허용 오차를 그대로 사용한다. 원 계약의
잘못된 SpCas9 z-score 설명은 #196의 정정을 따르며, 동결 숫자를 바꾸지 않았다.

4개 사례 × 9개 헤드에서 IG 32/256, SHAP 64/4의 seed 0/1/2와 seed 0 독립 반복을
계산했다. SHAP 512/32 비교는 #193에 등록된 **SpCas9-NG 헤드의 4개 사례만**
수행했다. 다른 8개 헤드의 512/32 비교까지 완료했다는 뜻은 아니다. seed 변경은
기존 동결 계약대로 배경과 부분집합 추출을 함께 바꾼다.

| 수치 검사 | 통과 | 실패 | 건너뜀 |
|---|---:|---:|---:|
| alpha 0/0.5/1의 설명 forward/gradient 출력 일치 | 108 | 0 | 0 |
| 예측 forward와 설명 대상 일치 | 36 | 0 | 0 |
| gradient 유한값 | 108 | 0 | 0 |
| IG 완결성 32단계 | 15 | 21 | 0 |
| IG 완결성 256단계 | 34 | 2 | 0 |
| IG 결과·항 유한값 | 72 | 0 | 0 |
| IG 32/256 안정성: 상관/최대 차이/강한 기여도 부호 | 82 | 26 | 0 |
| SHAP 결과·항 유한값 | 144 | 0 | 0 |
| SHAP 가법성 | 36 | 0 | 0 |
| SHAP 동일 seed 재현성 | 36 | 0 | 0 |
| SHAP seed 간 상관/최대 차이 | 0 | 216 | 0 |
| SHAP 기본값/512·32 상관 (SpCas9-NG만) | 0 | 4 | 0 |
| **합계** | **671** | **269** | **0** |

함수 일치 최대 오차는 **0**이다. IG 완결성 최대 raw 오차는 32단계
`0.0410337811`(4.103378%p), 256단계 `0.00719121591`(0.719122%p)다.
정밀 IG도 전체 헤드에서 품질 기준을 통과한 설정으로 부르지 않는다. IG 안정성은
상관 34/36, 최대 차이 12/36, 강한 기여도 부호 36/36이 통과했다.
SHAP 가법성 최대 raw 오차는 `3.72529030e-8`이며 동일 seed 값은 재현됐다.
이 통과는 seed 간 안정성을 뜻하지 않는다. 기본값/512·32의 상관은 S1–S4 각각
`0.8190 / 0.7410 / 0.5002 / 0.6429`로, 동결 기준 0.9에 미달한다.

수치 명령은 **종료코드 1**, 실행 시간 328.41초다. 656개 원시 배열은 모두
유한하다. 계산 중 소스 변경은 없었고, 보고서의 원시 NPZ SHA256과 소스 ZIP을
독립 리뷰에서 확인했다. 측정 당시 소스는
[measured-source.zip](benchmarks/issue199-xai-20261119-resumed01/measured-source.zip)에 있다.
측정 후 계산 소스의 유일한 변경은 `non_nt_runtime._features_for`의 `derive is None`
방어다. SpCas9의 `derive=None` 경로에서는 이 특징 헬퍼를 호출하지 않으므로
SpCas9 수치 계산은 바뀌지 않는다. Cas12a 직접 영향 경계 테스트를 추가 실행했다.

## 화면·회귀 검증 — 수치 품질과 별도

[최종 WebView2 보고서](benchmarks/issue199-xai-20261119-resumed01/window/report.json)는
**12 통과 / 0 실패 / 0 건너뜀**이다. 실제 계산 자식 프로세스에서 IG 로고와 SHAP
막대·워터폴, 제한 전 단위/seed/인과관계 참고, 별도 품질 경고 미표시를 확인했다.
크기 변경은 같은 PNG를 재사용했다. 정밀 SHAP의 진행률·취소, 취소 후 기본 결과
재선택, 계산 중 미지원 사이트 선택 시 이전 요청 취소·표시 제거도 통과했다.
UI 프로세스에 모델 라이브러리가 없고 종료 후 자식이 닫힌 것을 확인했다.
저장된 [IG](benchmarks/issue199-xai-20261119-resumed01/window/ig-0.png),
[SHAP 막대](benchmarks/issue199-xai-20261119-resumed01/window/shap-0.png),
[워터폴](benchmarks/issue199-xai-20261119-resumed01/window/shap-1.png)을 직접 열어 확인했다.

선택한 회귀 범위는 예측기·공용 런타임 handle·설명 캐시·데스크톱 그래프·단위
경계다. 공용 로그와 타입을 변경했으므로 NT 기준 보존 및 Cas12a 기존 경로·취소와
실제 non-NT 예측/CSV 프로세스까지 확인했다. 전체 저장소 스위트는 실행하지 않았다.

```powershell
py -3.13 -X utf8 -m pytest test/test_non_nt_spcas9.py test/test_non_nt_runtime.py test/test_explain_cache.py test/test_cache_identity.py tests/test_non_nt_xai.py tests/test_desktop_xai.py tests/test_worker_cache.py tests/test_nt_xai_view.py test/test_nt_explain.py::test_quality_failure_stays_displayable_and_nonfinite_result_is_rejected -q -p no:cacheprovider
# 137 passed / 0 failed / 0 skipped, 134.31s; 기존 작은 SHAP fixture의 특이행렬 경고 3건
py -3.13 -X utf8 -m pytest test/test_cas12a_ig.py::test_selected_handle_explains_the_composed_model_through_the_existing_loop test/test_cas12a_ig.py::test_path_protocol_prevents_cross_path_memory_and_disk_cache_hits test/test_cas12a_ig.py::test_selected_path_keeps_progress_and_callback_cancellation tests/test_non_nt_process.py -q -p no:cacheprovider
# 11 passed / 0 failed / 0 skipped, 65.05s
py -3.13 -X utf8 -m pytest tests/test_non_nt_xai.py::test_non_nt_quality_uses_frozen_bound_and_persists_internal_failure -q -p no:cacheprovider
# 최종 NT 기준 보존 단언 추가 후 2 passed / 0 failed / 0 skipped, 16.49s
py -3.13 -X utf8 docs/runtime/verify_issue199.py docs/benchmarks/issue199-xai-20261119-resumed01
# 수치 671 pass / 269 fail / 0 skip, exit 1
py -3.13 -X utf8 tests/non_nt_xai_window_probe.py .packaging/evidence/issue199/window-final01
# 화면 12 pass / 0 fail / 0 skip, exit 0
py -3.13 -m pyright non_nt_predict.py non_nt_runtime.py desktop/worker_renderer.py desktop/xai_view.py desktop/screen.py xai/quality.py docs/runtime/verify_issue199.py tests/non_nt_xai_window_probe.py
# 0 errors / 0 warnings
py -3.13 -m ruff check --select F,E9 non_nt_predict.py non_nt_runtime.py desktop/worker_renderer.py desktop/xai_view.py desktop/screen.py xai/quality.py docs/runtime/verify_issue199.py tests/test_non_nt_xai.py tests/non_nt_xai_window_probe.py test/test_non_nt_spcas9.py
# All checks passed
git diff --check
```

## 독립 리뷰

- **Standards:** 문서화된 규칙 위반·수정이 필요한 냄새 0건.
- **Spec:** 최초 1건(미지원 Cas12a 선택 시 이전 SpCas9 요청 잔류)을 수정했다.
  실제 화면 재검증과 읽기 전용 재리뷰 후 남은 구현 지적 0건.

기능·표시 검증 성공과 수치 기준 미달은 서로 다른 결론이다. 위 결과는 고정 사례의
검증이며 모든 서열의 품질이나 생물학적 정확성을 보장하지 않는다. 새 배포본·깨끗한
오프라인 Windows 검증과 정식 RAM 비교는 #203/#204 범위다. #191이나 다른 티켓을
자동 종료하지 않는다.
