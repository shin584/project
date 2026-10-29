# Cas12a 모델 리뷰

- **일자**: 2026-10-28
- **대상**: `cas12a/model/Cas12a_model_final.ipynb`, `cas12a/dataset/kim_2018_cas12a_dataset.xlsx`
- **브랜치**: `model_analysis`
- **결론**: 데이터는 깨끗하고 모델도 실제로 신호를 학습한다. 단, 노트북에 평가 지표가 전혀 없어
  **"모델이 좋은지 판단할 수 없는 상태"**다. 모델 결함보다 검증 부재가 핵심 문제다.

## 1. 데이터 검증 — 통과

`kim_2018_cas12a_dataset.xlsx` 원본 15,002행 → 타깃 결측 2행 제외 후 **15,000행**.

| 항목 | 결과 | 판정 |
|---|---|---|
| 서열 길이 | 15000/15000이 정확히 34bp | OK |
| non-ACGT 문자 | 0건 (`one_hot_encode`의 `[0,0,0,0]` 폴백 미발동) | OK |
| 중복 34bp 서열 | 0건 (고유 15,000) | OK |
| 중복 23bp protospacer | 0건 (고유 15,000) | OK |
| PAM 위치 | 위치 4:7이 100% `TTT` | OK |

### 1.1 누출 없음

중복 서열이 0건이므로 `KFold(n_splits=5, shuffle=True, random_state=42)`의 **행 단위 분할에도
누출이 발생하지 않는다.** SpCas9 9-head에서 확인된 PAM-library protospacer 중복(테스트셋 25%)
문제는 Cas12a에 해당하지 않는다. SaCas9의 cell-line replicate 누출과도 무관하다.

### 1.2 인덱싱 정확

컬럼명이 선언한 레이아웃(`4 bp + PAM + 23 bp protospacer + 3 bp`)과 실제 데이터가 일치한다.
4 + 4 + 23 + 3 = 34이고 위치 4:7이 전 행 `TTT`이므로, `extract_seq_features`의
`protospacer = seq[8:31]`은 **올바른 23bp protospacer를 집는다.**

참고로 PAM 4번째 염기 분포는 `TTTC` 5068 / `TTTG` 4897 / `TTTT` 3635 / `TTTA` 1400 이다.
24%가 `TTTT`이므로 이 라이브러리의 PAM은 TTTV가 아니라 **TTTN**이다 (Kim 2018 설계대로이며 버그 아님).

## 2. 성능 — 학습은 되지만 선형모델 수준

노트북이 지표를 전혀 남기지 않으므로, 동일 피처·동일 분할(`seed=42` 5-fold)로
기준선을 직접 산출해 비교했다.

| 모델 | MSE | MAE | R² | Spearman |
|---|---|---|---|---|
| 평균값 예측 (기준선) | 0.1039 | 0.2837 | 0.00 | — |
| **RidgeCV (선형)** | **0.0481 ± 0.0008** | 0.1705 | **0.537** | **0.738** |
| 노트북 CNN+BiLSTM (5 epoch) | 0.0439 ~ 0.0481 | 0.1556 ~ 0.1641 | ~0.54 ~ 0.58 | 미측정 |

### 해석

- 모델은 **분명히 신호를 학습한다** — 평균 예측기 대비 MSE가 절반 이하(0.1039 → 0.044).
  "모델이 망가졌다"는 아니다.
- 그러나 3-scale CNN + BiLSTM이 **단순 ridge와 사실상 동일하다.** 최악 fold(0.0481)는
  ridge 평균값과 같다. 현재 구조가 복잡도를 정당화하지 못한다.
- 원논문 DeepCpf1은 같은 데이터에서 Spearman ≈ 0.77을 보고한다. 현 상태는 그 아래로 추정되나,
  **노트북이 Spearman을 계산하지 않아 확정할 수 없다.**

### 기준선 재현 방법

- 피처: `one_hot(34×4=136)` + `[gc_total, gc_seed]` + `dinuc(22×16=352)` = 490차원
  (노트북 `extract_seq_features`/`one_hot_encode`와 동일한 산식)
- 타깃: `clip(indel_background_subtracted / 100, 0, 1)` (노트북과 동일)
- 모델: `RidgeCV(alphas=np.logspace(-2, 3, 12))`, 예측값을 `[0,1]`로 클립
- 분할: `KFold(n_splits=5, shuffle=True, random_state=42)` (노트북과 동일)

## 3. 수정 항목

### 🔴 P1 — 평가 지표가 하나도 없음

`train_5fold`가 `fold_results`를 빈 리스트로 초기화한 뒤 **아무것도 append하지 않고 반환한다.**
`mean_absolute_error`, `r2_score`, `pearsonr`를 import하지만 호출하지 않는다.
이 분야 표준 지표인 **Spearman은 import조차 없다.**

→ fold별 Spearman / Pearson / R² / MAE를 기록하고 평균±표준편차를 출력할 것.
이것만 넣으면 §2의 ridge 기준선을 실제로 이기는지 즉시 판정된다.

### 🔴 P2 — 학습 부족 + 베스트 체크포인트 없음

`epochs=5`가 전부이고, `EarlyStopping`/`ReduceLROnPlateau`를 import했으나
`model.fit(callbacks=...)`에 넘기지 않는다. `restore_best_weights`가 없으므로
**마지막 epoch의 가중치가 그대로 남는다.**

실제 피해: fold 4는 epoch 4에서 `val_loss 0.0445`를 봤으나 epoch 5의 `0.0481`로 종료됐다.
보고된 수치가 사실상 우연이다.

→ `EarlyStopping(monitor='val_loss', patience=…, restore_best_weights=True)` 연결,
epoch 상한 증대.

### 🟠 P3 — `bean_input` 죽은 코드

`create_cas12a_model` 마지막 부분:

```python
model = Model(inputs=[seq_input, bean_input] if 'bean_input' in locals() else [seq_input, feat_input], ...)
```

`bean_input`은 어디에도 정의되지 않으므로 항상 else 분기를 탄다. 즉 **우연히 동작하는 코드**다.
`inputs=[seq_input, feat_input]`로 단순화할 것.

### 🟠 P4 — 산출물 없음 / 재현 불가

- `fold_models`는 메모리에만 존재하고 디스크에 저장되지 않는다. 파일명이 `_final`인데
  재사용 가능한 산출물이 하나도 남지 않는다.
- KFold 시드는 고정했으나 TF/NumPy 가중치 초기화 시드는 미고정 → 재실행 시 수치가 달라진다.
- 별도 test 분할이 없다. 5-fold CV만으로 보고할지, held-out test를 둘지 정해야 한다.

### 🟡 P5 — 타깃 클리핑이 문서화되지 않음

`np.clip(y, 0, 1)`이 background 차감 후 음수인 **861행(5.7%)을 0으로 누른다.**
그 결과 **정확히 0인 타깃이 10.5%(1,570행)**가 된다 (원본 범위 -0.08 ~ 1.00, 1 초과는 0건).

sigmoid 출력 + MSE 손실 조합에서 zero-inflated 분포가 되므로, 의도된 선택이라면 근거를 남기고
아니라면 음수 유지 또는 별도 처리를 검토할 것.

### 🟡 P6 — dinucleotide 피처 중복

352차원 dinuc 원-핫은 CNN이 이미 입력으로 받는 one-hot 서열에서 **완전히 유도 가능하다.**
별도 Dense 분기(354차원 → 64)가 새 정보를 주지 않는다. ridge가 동일 성능을 내는 원인일 수 있다.

### 🟡 P7 — 테스트 없음

`cas12a/test/` 디렉터리가 비어 있다. 다른 모델 디렉터리와 달리 테스트가 전무하다.
최소한 데이터 로더(서열 길이 34 고정, PAM 위치, 타깃 범위)에 대한 테스트가 필요하다.

### 미사용 import

`EarlyStopping`, `ReduceLROnPlateau`, `mean_absolute_error`, `r2_score`, `pearsonr`,
`matplotlib.pyplot` — 전부 import되지만 사용되지 않는다. P1/P2를 처리하면 대부분 해소된다.

## 4. 권고 순서

1. **P1 + P2를 먼저** — fold별 Spearman/Pearson/R² 기록, EarlyStopping 연결.
   이 두 개로 "ridge 0.738을 이기는가"가 결정되고, 나머지 판단의 전제가 된다.
2. 이긴다면 → P3/P4(정리·저장·시드) 처리 후 확정.
3. 못 이긴다면 → P6(중복 피처) 및 구조 재검토. 현재 GlobalAveragePooling1D가
   BiLSTM 출력의 위치 정보를 평균으로 지워버리는 점도 함께 볼 것.

## 5. #91 이후 상태 — 알려진 잔존 항목

#91 로 피처·로더 정본이 `cas12a/features.py`·`cas12a/data.py` 로 분리됐고,
`cas12a/model/Cas12a_model_final.ipynb` 는 그 모듈을 import 한다 (§P7 의 테스트 부재는
`cas12a/test/` 로 해소). 실데이터 15,000행에 대해 추출 전후 `X_seq`/`X_feat`/`y` 가
완전히 동일함을 확인했다.

**아직 남은 중복**: `e_system/predict.py:60` (`one_hot_encode_dna`)·`:68`
(`extract_seq_features`) 는 정본과 글자 그대로 같은 코드를 여전히 들고 있다. `e_system`
수정이 #91 범위 밖이라 연결하지 않았다 — 즉 **한쪽만 고치면 조용히 어긋나 틀린 예측을 낸다**는
본문의 지적은 유효하다. 연결은 `e_system` 추적 방침(#89)이 정해질 때 별도 이슈로 수행한다.
의존 방향은 `e_system` → `cas12a` 단방향이며, `cas12a` 는 `e_system` 을 import 하지 않는다.

**데이터셋 출처**: `cas12a/dataset/kim_2018_cas12a_dataset.xlsx` 는 Kim HK et al.,
*Deep learning improves prediction of CRISPR-Cpf1 guide RNA activity*,
Nat Biotechnol 36:239-241 (2018) 의 보충 표(HEK293T 측정 34bp 타깃 15,002행)다.
2MB로 커밋하지 않고 `.gitignore` 에 등록했으며, 공용 사본은 Google Drive
`project_shared/data/` 에 있다.

## 6. #93 이후 상태 — 학습 경로 포팅

학습 경로가 노트북에서 `cas12a/model.py`(`create_cas12a_model`)·`cas12a/train.py`
(`train_5fold` + 진입점)로 분리됐다. `python -m cas12a.train` 으로 로컬 CPU 에서
끝까지 실행되며, Drive 마운트와 `/content/...` 경로는 사라졌다. 노트북은
그 진입점을 호출하는 Colab 래퍼로 남았다(삭제하지 않음 — 근거는 ADR 0002).

**§3 의 수정 항목 중 바뀐 것은 P3 하나뿐이다.** `bean_input` 죽은 분기는 #91 이
이미 제거했다(린트 F821 때문이며 동작·파라미터 수 영향 없음). 나머지는 그대로다:
P1(지표 없음), P2(epochs=5, 콜백 미연결), P4(산출물·시드), P5, P6 — **전부 #92 범위**다.
포팅과 설정 수정을 같은 커밋에 섞으면 성능 변화의 원인을 분리할 수 없기 때문에
의도적으로 손대지 않았고, `cas12a/test/test_train.py`·`test_model.py` 가 이 값들을
핀으로 고정해 #92 의 변경이 눈에 보이게 만든다.

**검증 범위**: 학습은 확률적이고 CPU/GPU 커널이 달라 bit-exact 비교가 불가능하다.
결정적인 부분만 검증했다 — 전처리(#91 에서 실데이터 15,000행 bit-exact 확인, 재측정 안 함),
분할(fold 인덱스 골든), 구조(21개 레이어 테이블 + 파라미터 96,513개). 노트북 보고치
`val_loss 0.0481`(fold 4, epoch 5)은 참고용이며 어디에도 단정으로 쓰지 않는다.

**CPU 비용**: 1 epoch 약 10초, 5-fold 전체 257.5초(4분 18초). 상세는
`cas12a/docs/cas12a-training-cost.md`. #92 를 로컬에서 돌릴지
Colab 으로 되돌릴지의 판단 근거다.
