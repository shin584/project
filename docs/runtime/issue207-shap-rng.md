# #207 동시 SHAP 호출의 RNG 복원과 두 seed 독립 제어 — 검증 기록

작성일: 2026-11-18. 부모: #205.

## 바뀐 것

- `xai/explain.py`: 모듈 잠금 `_SHAP_RNG_LOCK`이 kernel_shap 호출끼리
  NumPy 전역 RNG 저장 → `seed(subset_seed)` → 표본추출·평가 → 복원을 직렬화한다.
  이 함수를 거치는 호출만 보호한다. 전역 RNG를 직접 쓰는 관련 없는 코드까지
  보장하지 않는다. shap이 표본추출과 모델 평가를 한 호출에서 이어 하므로
  잠금은 계산 전체를 덮는다. on_batch도 잠금 안에서 불린다.
- `kernel_shap(..., background_seed=None, subset_seed=None)`: 주지 않은 축은
  `seed`를 따른다. 그래서 기존 단일 seed 호출의 의미가 그대로다. `SHAPResult`에
  `background_seed`·`subset_seed` provenance가 붙는다.
- 사전계산 SHAP 캐시는 두 seed가 모두 0일 때만 조회한다. 적중한 결과에는 0/0을
  기록한다. 캐시 키, 진행률·취소 계약, 기본값(`seed=0`), 런타임·checkpoint는
  바꾸지 않았다.
- 배경 수를 늘려도 앞쪽 배경은 같은 난수열 prefix를 공유한다(기존 동작). 이번에
  공개 API에서 검증했다.

## 검증 (`test/test_shap_rng.py`, 실제 `shap.KernelExplainer` 0.52.0)

- 동시 호출: 테스트가 운영 잠금을 먼저 쥔 상태로 두 호출을 출발시킨다. 두 호출이
  잠금 밖 explainer 초기화를 끝내고 획득을 시도한 것을 확인한 다음 잠금을 놓는다.
  직렬화 구간 안에는 대기를 두지 않는다. 결과를 직렬 결과와 대조하고,
  획득·해제 순서가 겹치지 않는지 확인한다. 같은 seed와 다른 seed 두 경우를 본다.
- 복원: 정상 완료, 그리고 부분집합 표본추출 **뒤** 표본 평가 중 `on_batch`가 던지는
  예외·협력 취소. 던지는 시점에 seed가 걸려 있었고 난수가 소비됐는지 확인한다.
  끝난 뒤 RNG 상태와 잠금 해제도 확인한다.
- 축 독립성: subset_seed만 바꾸면 배경과 기준값은 같고 표본 마스크와 값이 바뀐다.
  background_seed만 바꾸면 표본 마스크는 같고 배경과 기준값이 바뀐다. 단일 seed는
  두 축에 같은 값으로 대응한다. 배경 수를 늘려도 prefix는 같다.
- 변이 확인: 잠금을 빼면 동시 테스트 2개가 실패한다. 복원을 빼면 5개가 실패한다.

실행 명령과 결과:

```
python -m pytest test/test_shap_rng.py -q -p no:cacheprovider
→ 14 passed, 0 failed, 0 skipped (6회 반복 모두 동일)

python -m pytest test/test_shap_rng.py test/test_explain.py test/test_explain_cache.py \
  tests/test_shap_cache.py tests/test_progress_callback.py tests/test_nt_cancellation.py \
  tests/test_dashboard_integration.py -q -p no:cacheprovider -rs
→ 94 passed, 0 failed, 0 skipped
```

선택 근거: kernel_shap·SHAPResult·사전계산 캐시를 직접 쓰는 테스트, 그리고
진행률·취소 계약을 검증하는 테스트(AGENTS.md 범위 규칙).

## 남은 것

- 실제 모델(채택 non-NT 런타임)로 동시 호출을 실행하지는 않았다. 보장은 공용
  함수 경계에서 검증했다.
- 잠금이 SHAP 계산 전체를 직렬화하므로, 동시 요청이 늘면 처리량이 줄 수 있다.
