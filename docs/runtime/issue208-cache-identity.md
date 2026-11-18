# #208 계산 식별이 일치하는 설명 캐시만 재사용 — 검증 기록

작성일: 2026-11-18. 부모: #205. 선행: #207 (`68437ba`).

## 바뀐 것

- `xai/identity.py` (새 파일): `ExplanationIdentity(family, model, checkpoints,
  protocol, configuration)`는 모델 쪽 계산 식별이다. `verified`는 다섯 조건을 모두
  요구한다. 알려진 계열(`nt`, `non_nt`)이고, checkpoint sha256이 1개 이상이며
  형식이 맞아야 한다. 모델 이름과 프로토콜도 비어 있지 않아야 한다.
  `from_files`는 경로가 아니라 파일 바이트를 해시한다.
- XAI 메모리 캐시 (`xai/explain_cache.py`): 키는 (기법, 입증된 식별, 헤드,
  윈도우, 기법별 설정)이다. SHAP은 여기에 `background_seed`·`subset_seed`를
  더한다. `kernel_shap`이 #207 seed API를 받는다. 단일 `seed`는 두 축에 같은
  값으로 대응하므로 `seed=3`과 `(3, 3)`은 같은 항목이다. 식별은 둘 중 하나로
  얻는다. 하나는 `model_key` 자리에 식별을 직접 넘기는 방식(데스크톱 worker)이다.
  다른 하나는 이름 문자열로 부를 때 예측기의 `explanation_identity()`를 쓰는
  방식(Streamlit)이다. 이때 이름과 식별의 model이 같아야 한다. 입증할 수 없으면
  매번 새로 계산하고 결과를 저장하지 않는다.
- 사전계산 SHAP 캐시 (`xai/shap_cache.py`): 키는 식별·헤드·윈도우·설정·두 seed를
  담은 정규 JSON이다. 파일에는 `schema=shap-precomputed-v2`와 전체 키가
  들어간다. 다음 경우는 거부(미적중)한다.
  - 키를 입증할 수 없는 경우
  - `schema`가 없는 legacy 파일
  - 저장된 키가 요청과 다른 경우
  - 값의 모양이 윈도우와 다르거나 비유한 값이 있는 경우

  입증할 수 없는 키나 seed가 어긋난 결과는 `save`가 거부한다.
- `explain.kernel_shap`: handle의 `model_key`가 입증된 식별일 때만 디스크를
  조회한다. #207의 "0/0일 때만 조회" 대신 두 seed를 키에 넣는다.
- NT: `predict.NT_EXPLANATION_PROTOCOL`(값은 기존
  `nt-legacy-calculation-v1-units-pending-201`)과 `nt_explanation_identity(spec)`.
  이 함수는 checkpoint 바이트, spec, dtype, 배치 크기를 담는다.
  해시는 (설정, 프로토콜, 파일 경로·크기·mtime) 키로 memo하므로 파일이나
  프로토콜이 바뀌면 다시 해시한다. `NTClassifierPredictor.explanation_identity()`와
  데스크톱 worker가 모두 이 함수를 쓰고, NT handle의 `model_key`가 이 식별이다.
  SHAP seed 해석은 `explain.resolve_seeds` 한 곳에서 한다.
- `make_precomputed_shap.py`: 식별 키로 저장하도록 고쳤다. 실행하지는 않았다.

## 바꾸지 않은 것

- `data/precomputed_shap/`의 legacy 산출물 4개는 지우거나 다시 만들지 않았다.
  거부만 한다. NT 단위 교정, 품질 승인, 재생성은 #201에 인계한다. #201이
  `NT_EXPLANATION_PROTOCOL`을 올리면 이전 결과는 메모리·디스크 어디서도
  재사용되지 않는다(테스트로 확인).
- 배포 기본값(IG 32, SHAP 64/4, seed 0), 런타임, checkpoint는 그대로다.
  추가 모델 캐시도 없다.
- 데스크톱 worker는 계속 `use_disk_cache=False`다. 출처 없는 디스크 산출물을
  읽지 않는다는 기존 보장(`tests/test_worker_cache.py`)은 그대로 통과한다.
- `docs/equivalence-baseline/capture_baseline.py`(#181 역사 기록 스크립트)는
  손대지 않았다. 다시 돌리면 이름 키 `save`에서 `ValueError`로 멈춘다. 출처 없는
  산출물을 새로 만들지 않게 의도한 동작이다.

## 검증

선택 근거(AGENTS.md 범위 규칙): 아래 테스트만 골랐다.

- 변경한 공개 경계: `ExplainCache`, `shap_cache`, `kernel_shap`,
  `predict` 식별
- 그 소비 경로: worker, 데스크톱 base renderer, Streamlit `xai/view.py`
- #207 seed·RNG 계약
- 진행률·취소 계약

새 테스트 `test/test_cache_identity.py`(47개)는 실제 `shap.KernelExplainer`를 쓰고
모델만 작은 numpy 함수로 바꾼다. 확인하는 것은 다음과 같다.

- 모델 쪽 식별 항목(checkpoint, 프로토콜, 런타임 설정, 모델) 각각의 차이를 양
  계열(nt, non_nt)에서 거부한다. 계열만 다른 경우도 거부한다.
- 요청 쪽 항목(헤드, 윈도우, nsamples, n_background, background_seed,
  subset_seed, 단일 seed, IG steps) 각각의 차이를 거부한다.
- 거부 뒤 결과는 캐시를 거치지 않은 fresh 계산과 같다.
- 허용된 메모리 적중(SHAP, IG)과 디스크 적중은 추론 없이 나온다. 그 값·기준값·
  예측·오차·seed provenance는 fresh 계산과 정확히 같다.
- 입증할 수 없는 식별 7종은 절대 적중하지 않는다: 이름만, 짧은 해시,
  checkpoint 없음, 모르는 계열, 빈 프로토콜, 예측기가 다른 모델을 댐, 예측기
  식별 미입증.
- 이름 키라도 예측기가 식별을 입증하면 재사용한다. 그 checkpoint가 바뀌면
  재사용하지 않는다.
- legacy 형식 파일이 정확히 그 경로에 있어도 거부한다. 비유한 값이나 모양이
  틀린 파일도 거부한다.
- 저장소의 실제 legacy 4개는 예전 키로도, 같은 윈도우의 입증된 키로도 적중하지
  않는다. 실행 전후 바이트가 같다.
- `nt_explanation_identity`는 파일 바이트와 프로토콜을 따른다.
- Streamlit 이름 키 경로(`NTClassifierPredictor`)도 프로토콜이나 checkpoint
  파일이 바뀌면 예전 결과를 재사용하지 않는다.

worker 프로토콜 무효화 테스트는 `tests/test_worker_cache.py`에 추가했다.

기존 테스트 중 바뀐 계약에 맞춰 수정한 것은 아래와 같다. 시나리오와 의도는
유지하고, 이름 문자열 대신 입증된 식별을 쓴다.

- `test/test_explain_cache.py`, `tests/test_shap_cache.py`,
  `tests/test_dashboard_integration.py`, `tests/test_desktop_xai.py`,
  `tests/test_xai_cache_view.py`: 가짜 예측기·handle이 식별을 댄다.
- `test/test_shap_rng.py`: `test_disk_cache_is_only_consulted_when_both_seeds_are_zero`
  를 `test_disk_cache_keys_carry_both_seeds`로 바꿨다. 이름 키는 조회하지 않고,
  입증된 키는 두 seed를 담아 조회한다는 내용이다.

변이 확인: 다음 다섯 가지 변이에서 새 테스트가 각각 실패했다.

| 변이 | 실패 수 |
| --- | --- |
| schema 검사 제거 | 1 |
| 메모리 키를 이름으로만 잡기 | 8 |
| 이름·식별 model 대조 제거 | 1 |
| 해시 형식 검사 제거 | 6 |
| seed를 메모리 키에서 빼기 | 3 |
| 식별 memo 키에서 프로토콜 빼기 | 2 (예측기 경로 1, worker 1) |

실행 명령과 결과:

```
python -m pytest test/test_cache_identity.py -q -p no:cacheprovider
→ 47 passed, 0 failed, 0 skipped

python -m pytest test/test_cache_identity.py test/test_explain_cache.py \
  tests/test_shap_cache.py tests/test_worker_cache.py test/test_shap_rng.py \
  test/test_explain.py tests/test_progress_callback.py tests/test_nt_cancellation.py \
  tests/test_dashboard_integration.py tests/test_desktop_xai.py \
  tests/test_xai_cache_view.py test/test_predict.py test/test_core_facade.py \
  -q -p no:cacheprovider -rs
→ 182 passed, 0 failed, 0 skipped
```

pyright: 변경한 줄에서 새 오류는 없다. `xai/identity.py`, `xai/shap_cache.py`,
`xai/explain_cache.py`, `desktop/worker_renderer.py`는 0 errors다. `predict.py`와
`xai/explain.py`의 기존 오류는 변경 줄 밖이다. ruff(F, E9)는 변경 파일에서 통과했다.

코드 리뷰(standards·spec 두 축)에서 다음을 반영했다.

- 예측기의 식별 memo가 프로세스 수명 동안 굳던 문제를 고쳤다.
- 쓰이지 않던 worker의 `EXPLANATION_PROTOCOL` 별칭을 지웠다.
- 중복된 seed 해석과 중복된 설정 조립을 한 곳으로 모았다.
- 타입 힌트(`ModelKey`)와 "캐시" 수식어, CONTEXT.md 항목 위치를 바로잡았다.

반영하지 않은 것도 있다. 디스크 파일의 모양·유한성 검사와 `non_nt` 계열 상수는
유지했다. 앞의 것은 비용이 거의 없고 비유한 산출물 재사용을 막는다. 뒤의 것은
양 계열 무효화 기준에 필요하다.

## 남은 한계

- 실제 NT·non-NT 가중치로 캐시 적중 = fresh를 실행하지 않았다. 보장은 실제
  KernelSHAP과 작은 결정적 모델로 공용 캐시 경계에서 검증했다.
- non-NT 계열은 아직 설명 소비자가 없다(#195·#199). 식별과 무효화 규칙은 계열
  공통 API에서 양 계열로 검증했다. non-NT 프로토콜 문자열과 식별 생성 지점은
  그 소비자가 정한다. `non_nt_runtime.explain_handle(model_key=...)`에 식별을
  넘기면 같은 규칙이 적용된다.
- Streamlit 화면은 NT 예측기가 첫 설명 요청 때 checkpoint를 한 번 해시한다
  (Cas당 약 1.9GB 파일 1개, 요청 경로에서 동기로). 이후에는 memo를 쓴다.
- `make_precomputed_shap.py`는 새 키로 저장하도록 고쳤지만, #201 전에 실행하지
  말라는 경고는 docstring에만 있다.
- 식별은 checkpoint 바이트와 선언한 프로토콜·설정까지다. 코드 자체(예: 설명
  함수 구현)의 변경은 프로토콜 문자열을 올려야 반영된다.
