# #196 — non-NT SpCas9 예측을 기존 보기와 CSV까지 연결

[요구사항 #196](https://github.com/shin584/project/issues/196). 선행 티켓은
런타임·계약을 동결한 [#193](https://github.com/shin584/project/issues/193)과
계산 프로세스를 분리한 [#194](https://github.com/shin584/project/issues/194)이며 둘 다 종료됐다.

이 티켓은 **예측과 그 결과의 화면·CSV 연결**까지다. 설명(IG·SHAP) 계산은 #199,
non-NT Cas12a는 #197, 기본 시작 계열과 공통 선택창은 #198이다. 이 문서의 어떤
결과도 v3 전체의 기본 모드·선택창 완료를 뜻하지 않는다.

## 1. 모델 계열을 코드에 들여온 방식

`(모델, Cas)` 키와 `VariantResult.model`("A")은 그대로 두고, **계열**을 그 위의
축으로 더했다. 각 Cas의 `("A", <Cas>)` 자리에 그 계열의 예측기 하나가 앉는다.

| 이름 | 역할 |
| --- | --- |
| `predict.FAMILIES` | `("nt", "non_nt")`. `xai.identity.FAMILIES`와 같은 이름을 쓴다 |
| `predict.DEFAULT_FAMILY` | 지금은 `"nt"`. 기본 시작 계열 변경은 **#198** |
| `predict.MODEL_REGISTRY` / `NON_NT_REGISTRY` | 계열별 `ModelSpec`. 기존 이름은 유지했다 |
| `predict.FAMILY_WARMUP` | 계열별 워밍업 키. non-NT는 `()` — 아무것도 데우지 않는다 |
| `predict.load_predictors(warmup, family)` | 그 계열 **하나만** 만든다 |

`load_predictors`는 인자를 정규화한 뒤 캐시한다. `@cache`를 바로 걸면
`load_predictors()`와 `load_predictors(family="nt")`가 다른 키가 되어, 한
프로세스가 같은 계열의 예측기를 두 벌 만들고 NT 가중치도 두 번 올린다.

non-NT 예측기는 `FAMILY_PREDICTOR_CLASSES`에 **"모듈:이름" 문자열로** 적혀 있다.
그 계열을 고르지 않은 프로세스가 import만으로 Keras 런타임을 끌어오지 않게 하기
위해서다. NT 항목은 기존 모듈 전역 `PREDICTOR_CLASSES`를 호출 시점에 읽는다
(`predict.predictor_classes`) — 그 이름을 갈아 끼워 지연 로딩을 검사하는 기존
시험이 계속 유효하도록 사본을 두지 않았다.

### 모델/checkpoint 식별이 예측 행과 함께 다닌다

`pam_scanner.ModelIdentity`를 더하고 `VariantResult.identity`에 실어 보낸다.
CSV의 모델·버전 열, 화면의 원출력 단위 안내, 계산 프로세스의 예측 캐시 키가
**이 한 값**을 쓴다. 세 곳이 따로 세면 저장된 CSV가 가리키는 모델과 화면이 보여준
모델이 조용히 갈라질 수 있다. 예측기를 거치지 않은 행(미리보기)은 `None`이고,
그때 화면은 모델 이름이나 단위를 주장하지 않는다.

checkpoint 해시를 얻는 방법은 계열마다 다르다.

- non-NT — `model_source.NON_NT_FILES`에 **고정된** sha256. `model_source.resolve`가
  적재 때 실제 바이트를 그 값과 대조하므로, 어긋난 파일로는 추론 자체가 안 된다.
  모델을 올리지 않고도 식별을 댈 수 있다 (예측 캐시 키는 예측 **전에** 만든다).
- NT — 실제 파일을 해시한 값. `nt_explanation_identity`가 파일 상태별로 memo한다.

설명 캐시 식별(`xai.identity.ExplanationIdentity`)은 이것과 별개다. 설명
프로토콜까지 포함해야 하고 그 프로토콜은 계열이 소유한다. `ModelSpec`에 더한
표시용 항목(`name`·`asset_id`·`raw_unit`)은 **계산 식별에 들어가지 않는다** —
열 이름을 고치는 것만으로 사전계산 SHAP 캐시가 전부 미스가 되지 않도록
`nt_explanation_identity`가 계산에 영향을 주는 항목만 담는다
(`test/test_cache_identity.py::test_display_only_spec_fields_do_not_change_the_explanation_identity`).

## 2. 30bp 인코딩·헤드 순서·단위 — 고정 사례에서 원 모델과 비교

측정 결과: `docs/benchmarks/issue196-spcas9-parity.json`.
허용 오차는 이 티켓이 새로 정하지 않고 #193에서 동결한
`docs/runtime/contract.json`의 `criteria.forward_parity`(abs 1e-4 / rel 1e-3)를 썼다.

| 항목 | 결과 |
| --- | --- |
| 입력 | `In_Seq` [30, 4], 채널 A·C·G·T. export manifest의 `input`과 같고 `CAS_CONFIGS`의 `model_input_len`과도 같다 |
| 헤드 순서 | 적재한 `output_names` 9개가 export manifest의 `index → layer`와 같고, 그 `protein` 순서가 `SPCAS9_VARIANTS`와 같다 |
| 고정 사례 전방 | 비교 19행(변종 9개 전부 포함), **최대 절대 오차 6.0e-8 / 상대 5.1e-7** → 통과 |
| 보관 예측 재현 | `test_eval/predictions.csv` 8행 × 9헤드, **최대 절대 오차 1.3e-7** → 통과 |

기준값은 이 작업이 다시 계산한 값이 아니라 보관된 산출물에서 읽었다.
`export-manifest.json`의 `validation.batches`는 학습 원본(.keras, TensorFlow
backend)과 export가 같은 값을 냈다는 #98의 기록이고, `test_eval/predictions.csv`는
같은 모델로 뜬 테스트셋 예측이다.

한 윈도우로 헤드 9개를 다 덮을 수는 없다 — VRQR는 NGA를, 나머지 여섯은 NGG를
인식하므로 한 PAM이 둘을 동시에 만족하지 못한다. 고정 윈도우들의 합집합으로 덮었다.

### 원출력 단위와 효율 변환

**출처가 확인됐으므로 효율로 표시한다.** 세 산출물이 같은 말을 한다.

| 출처 | 내용 |
| --- | --- |
| `models/non_nt/spcas9-9hydra-v2/export-manifest.json` | `output.scale` = `published_percentage/100`, `activation` = `linear` |
| 같은 폴더 `test_eval/results.json` | `scale` = `"published percentage / 100 (the scale 9-Hydra v2 was trained on)"`. `*_true`와 `*_pred`를 그 척도에서 직접 비교한다 |
| `models/non_nt/manifest.json` | 학습 원본 → export의 출처·SHA 연결 |

따라서 `output_mode="raw"`다. 표시 점수는 NT와 같은 `predict._to_display_score`
하나를 지나 **0~1로 자른 뒤 ×100**이고, `score_kind`는 `"efficiency"`다. linear
출력이라 음수가 나올 수 있으며 0%로 눌러 보여주고 원출력은 그대로 보존한다.

#### `docs/runtime/contract.json`의 `raw_unit` 표기 오류

그 파일의 `models.spcas9_non_nt.raw_unit`은 이 모델을
`"z-score (training-distribution standardized)"`라고 적었다. 위 세 산출물과
어긋나는 **설명 문구의 오류**다. 보관된 값 자체가 반증한다 — 기록된 예측은 모두
0 근방의 양수 비율이고(evoCas9 0.0255 등) `*_true`와 같은 척도에서 직접 비교됐다.
z-score라면 평균 0 양쪽으로 퍼져야 하고, `*_true`와 직접 비교할 수도 없다.

그 필드는 #193의 **수치 판정 기준이 아니라 설명 문구**이므로, 동결된 허용 오차와
#193의 측정 결과(`issue212-20261119-required01`, 605 통과 / 43 실패, 최종 `fail`)는
이 정정으로 바뀌지 않는다. 계약 파일은 #193의 동결 산출물이므로 고치지 않고,
정정 근거를 여기와 `predict.NON_NT_REGISTRY`의 주석에 남긴다.

**이 단위 확인은 예측값의 품질 승인이 아니다.** 배포 모델의 test 성능·출처는 #202다.

## 3. 지연 로딩과 NT 비접촉

- `NonNTSpCas9Predictor.predict_batch([])`는 checkpoint를 읽지 않고 끝난다. 서열에
  SpCas9 PAM이 없으면 이 모델은 메모리에 올라가지 않는다.
- 적재 시 입력 모양·헤드 순서·고정 해시를 대조하고, 어긋나면 **멈춘다**. 조용히
  변종만 뒤바뀐 표가 나오는 것을 막기 위한 것이다.
- non-NT 계열은 워밍업이 없다 (`FAMILY_WARMUP["non_nt"] == ()`).
- 실제 측정: `tests/test_non_nt_process.py::test_the_non_nt_path_loads_no_nt_weights_or_tokenizer`
  가 별도 프로세스에서 예측과 CSV를 끝낸 뒤 `predict.load_nt_tokenizer`의 캐시가
  비어 있고 `transformers`·`safetensors`가 `sys.modules`에 없음을 확인한다.

## 4. 화면 네 보기와 XAI 선택

| 보기 | 연결 |
| --- | --- |
| Best Candidates | 사이트별 대표 행. 원출력 단위 안내를 **행이 들고 온 식별에서** 읽는다 |
| 변종 비교 | 가이드 × 변종 히트맵. 이 계열의 예측 행이 그대로 들어간다 |
| 전체 상세 | 표·필터·CSV. 모델·버전 열이 끝에 붙는다 |
| XAI 사이트/헤드 선택 | `explain_heads`가 예측 행과 **같은 집합**을 돌려준다 |

원출력 단위 안내는 더 이상 코드에 적혀 있지 않다. 전에는 "SpCas9은 z-score라…"가
고정 문구였는데, 그 문장은 non-NT 계열에서 틀린 설명이 된다. 이제
`ModelSpec.raw_unit`에 **출처가 확인된 단위만** 적고, 비어 있거나 식별이 없는 행에
대해서는 아무 단위도 주장하지 않는다. '모델 출력 해석 상태' 블록도 레지스트리 키가
아니라 행의 식별을 읽는다.

설명 계산은 연결하지 않았다. `NonNTSpCas9Predictor.explain_handle`과
`NonNTComputationRenderer._model_key`는 #199를 가리키며 `NotImplementedError`로
멈춘다 — 미구현 기법이 성공 결과로 표시되지 않는다. Attention Rollout은 non-NT에서
제공되지 않으며 그 메뉴 처리도 #199의 범위다.

## 5. CSV

기존 9열을 **순서·값·필터·정렬 의미 그대로** 두고 맨 끝에 `모델`·`모델 버전`을 더했다.

```
순위, Cas 변종, PAM, 가이드 서열 (RNA), 가닥, 거리 (bp), 원출력, 예측 점수 (%), 점수 기준, 모델, 모델 버전
```

- `desktop.detail_view.BASE_COLUMNS` + `MODEL_COLUMNS` = `DETAIL_COLUMNS`. 표와 CSV가
  같은 `_cells` 하나에서 나오므로 두 쪽이 열을 따로 세지 않는다.
- 모델 열은 **행마다** 그 행을 낸 모델을 가리킨다. 한 CSV에 Cas가 섞여도 섞인 대로 맞다.
- 필터(`DetailFilter`)와 정렬은 손대지 않았다. 순위는 거른 뒤 1부터 다시 매긴다.
- 현재 계열의 결과만 저장된다 — 계산 프로세스가 계열 하나만 들기 때문이다.

### 보존된 기준 CSV와의 대조

열을 더했으므로 저장 파일이 `docs/equivalence-baseline/crispr_prediction_results.csv`와
바이트로 같지는 않다. 기준이 증거로 남긴 것은 **기존 열의 값과 순서**이므로 그
부분만 떼어 바이트로 비교한다. 자르는 자리는 `detail_view.base_columns_only` 하나가
정하고, `desktop/validation.py`와 `tests/test_detail_view.py`가 그것을 함께 쓴다.
`validation.py`의 보고 항목은 `csv_byte_equal` → `csv_base_columns_byte_equal`로
바뀌었고 `csv_whole_file_byte_equal`·`csv_model_columns`를 함께 남긴다.

## 6. 검증 실행

고른 범위는 #196이 바꾼 공개 인터페이스(계열별 예측기 적재, 예측 행의 모델 식별,
상세 표·CSV 열, 계산 프로세스 렌더러)와 그 경계의 기존 회귀다.

| 실행 | 결과 |
| --- | --- |
| `pytest test/test_non_nt_spcas9.py -q` (실제 checkpoint) | **15 통과 / 0 실패 / 0 건너뜀** |
| `pytest tests/test_non_nt_process.py -q` (실제 checkpoint, 실제 자식 프로세스) | **8 통과 / 0 실패 / 0 건너뜀** |
| 리뷰 수정 전 회귀 (21개 파일) | 302 통과 / 0 실패 / 1 건너뜀 |
| **최종 회귀 — 아래 23개 파일 한 번에** | **320 통과 / 0 실패 / 0 건너뜀**, 경고 4건 |
| `pytest tests/test_nt_process.py -q` (실제 NT checkpoint 2개, 자식 프로세스) | **1 통과 / 0 실패** (168초) |

리뷰 수정 전의 건너뜀 1건은
`test/test_predict.py::test_unconnected_models_claim_no_checkpoint`였다 — non-NT
Cas12a가 같은 워크트리에서 #197로 연동되어 영구히 건너뛰게 됐으므로,
`UnconnectedPredictor`를 직접 검사하도록 고쳐 건너뜀을 없앴다.

최종 회귀에는 같은 커밋에 든 #197의 시험 2개 파일(`test/test_non_nt_cas12a.py`,
`tests/test_non_nt_cas12a_process.py`)도 함께 넣었다.

```
python -X utf8 -m pytest \
  test/test_core_facade.py test/test_core_loading.py test/test_pam_scanner.py \
  test/test_non_nt_runtime.py test/test_cache_identity.py test/test_explain.py \
  test/test_explain_cache.py test/test_predict.py test/test_non_nt_spcas9.py \
  tests/test_desktop_api.py tests/test_desktop_shell.py tests/test_detail_view.py \
  tests/test_prediction_views.py tests/test_integrated_layout.py \
  tests/test_worker_cache.py tests/test_computation_process.py \
  tests/test_dashboard_integration.py tests/test_progress_callback.py \
  tests/test_desktop_xai.py tests/test_xai_cache_view.py \
  tests/test_non_nt_process.py -q -rs
```

이 브랜치에는 타입 검사기·린터 설정이 없다 (`.pre-commit-config.yaml`은 의도적으로
비어 있다). 대신 바꾼 모듈 전부를 `py_compile`과 실제 import로 확인했다.

`tests/test_nt_process.py`는 실제 NT checkpoint 2개(약 3.8GB)를 올려 기존 NT 경로가
그대로인지 본다. CSV 비교를 기존 열 기준으로 고친 뒤 실행해 통과했다 — **보존된
기준 CSV와 기존 9열이 바이트로 같고**, 더해진 모델 열이 NT 계열의 모델을 가리킨다.
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`로 돌렸다.

저장소 전체 스위트는 실행하지 않았다 (`AGENTS.md`의 #191 범위 규칙).

## 7. 독립 리뷰 (Standards · Spec 2축)

두 축을 각각 독립 에이전트로 돌렸다. 고친 것과 남긴 것을 나눠 적는다.

고친 것:

- `docs/desktop-boundary.md`·`docs/computation-process.md`가 문서화한 CSV 계약을
  함께 갱신하지 않았다 → 열 표와 "바이트까지 같다" 문단을 고쳤다. (문서화된 규칙 위반)
- `csv_against_baseline`이 **양쪽을** 다시 직렬화해 바이트 비교가 아니라 왕복
  비교였다 → 보존된 기준의 원래 바이트와 맞춘다. 저장 파일만 다시 직렬화한다.
- 미리보기 행이 `family="nt"`를 주장하고 `UNKNOWN_MODEL` 경로를 죽였다 → 미리보기
  행의 식별을 `None`으로 두어, 시험하는 경로와 실제 경로가 같아졌다.
- `forward_all`이 `forward`의 본문을 복제했다 → 전방 루프를 `forward_all` 하나로
  두고 `forward`는 그 결과의 한 열을 고른다. head 범위 검사는 유지했다.
- 비교 함수와 CSV 디코드가 `validation.py`와 `compare_baseline.py`에 각각 복사돼
  있었다 → `detail_view`의 `csv_rows`·`csv_against_baseline` 하나로 모았다
  (`base_columns_only`의 "함수는 하나만 둔다"와 같은 자리).
- `_rows_html`의 `cells[7]`·`cells[8:]` 같은 위치 상수 → `BASE_COLUMNS.index(...)`.
- '모델 출력 해석 상태' 제목이 더 이상 해석 모드를 보여주지 않았다 → '모델과 출력 범위'.
- 진단 블록이 첫 행에 식별이 없으면 그룹 전체를 `?`로 적었다 → 식별이 있는 행을
  찾고, 없으면 모델 이름을 **적지 않는다**.
- 시험 정리: 리터럴을 단정하던 줄 제거, 영구히 건너뛰던
  `test_unconnected_models_claim_no_checkpoint`를 `UnconnectedPredictor` 직접
  검사로 바꿈(건너뜀 0), Cas12a 행에 SpCas9 식별을 주던 fixture 수정, 쓰지 않는
  fixture 제거, 단위 출처 재현을 `split == "test"` 행으로 한정.
- 동결 계약과의 불일치가 코드 주석에만 있었다 → `docs/runtime/README.md`에 근거를
  갖춘 정정 절을 추가했다. `contract.json` 자체는 ADR 0002대로 고치지 않았다.

남긴 것과 이유:

- **`NonNTComputationRenderer`가 설명 캐시를 물려받고 `_model_key`를 막는다
  (Refused Bequest).** #199가 non-NT 설명을 붙이면 그 상속이 맞게 된다. 지금
  갈라 두면 #199에서 되돌려야 한다.
- **계열이 맨 문자열이고 표가 세 개다 (`FAMILY_REGISTRIES`·
  `FAMILY_PREDICTOR_CLASSES`·`FAMILY_WARMUP`).** 한 레코드로 접으면 NT 예측기 표를
  **호출 시점에** 읽어야 하는 요구(기존 시험이 `PREDICTOR_CLASSES`를 갈아 끼운다)를
  깨뜨린다. 그래서 `predictor_classes()`만 두고 표는 셋으로 남겼다.
- **고정 사례의 기준값이 보관된 숫자이고 학습 원본 `.keras`를 다시 돌리지 않았다.**
  backend 간 전방 parity는 #193이 자체 기준 런타임으로 이미 측정·동결했다. 이
  티켓이 다시 재는 것은 그 범위의 재실행이다. 문서와 증거 JSON에 출처를 명시했다.
- **`compare_baseline.py`가 `desktop.detail_view`를 import한다.** 자르는 자리를
  복제하지 않으려면 공유가 필요하다. 앱의 `BASE_COLUMNS`가 바뀌면 이 비교가
  알아채는 것이 맞는 방향이다.

## 8. 같은 워크트리의 #197

이 작업 중에 다른 에이전트가 같은 워크트리에서 #197(non-NT Cas12a)을 구현했다.
사용자 결정에 따라 분리하지 않고 공유 커밋으로 둔다. 그래서 이 커밋에는 #196이
만들지 않은 다음이 함께 들어 있다.

- `predict.NON_NT_REGISTRY[("A", "Cas12a")]`와
  `FAMILY_PREDICTOR_CLASSES["non_nt"]`의 Cas12a 항목.
- `desktop/worker_renderer.py`의 `NonNTComputationRenderer._scan`
  (`non_nt_cas12a_predict.MANIFEST_CAS_CONFIGS`를 읽는다).

**따라서 #196은 단독으로 되돌릴 수 없다.** #197의 모듈·시험·문서는 이미 별도
커밋(`ef1ff17`)에 있다.

### 아직 하지 않은 것

- 실제 Windows 창에서의 non-NT 화면 확인. 지금 데스크톱 앱은 여전히 NT 계열로
  시작하므로(`DEFAULT_FAMILY == "nt"`), non-NT를 창에서 띄우는 것은 선택창과 기본
  계열을 붙이는 **#198**의 검증이다. 이 티켓은 실제 계산 자식 프로세스를 띄워
  네 보기 조각과 저장된 CSV로 확인했다.
- 전체 프로세스 트리 RAM 실측(#203), 새 오프라인 배포본(#204), 모델 성능·출처(#202).
- non-NT 설명 수치 검증(#199/#200)과 NT 단위 교정(#201).
