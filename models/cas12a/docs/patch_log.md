cas12a 문제점과 #89~#93 개선 정리

리뷰 기준일은 2026-10-28, 대상은 포팅 전 cas12a/model/Cas12a_model_final.ipynb 입니다. 리뷰의 총평은 "모델 결함보다 검증 부재가 핵심" — 데이터는 깨끗하고 모델도 신호를 학습하지만, 지표가 하나도 없어 좋은지 판단할 수 없는 상태였습니다.

---

1. 발견된 문제 (리뷰 §3)

🔴 P1
문제: 평가 지표 전무
내용: train_5fold 가 fold_results 를 빈 리스트로
초기화 후 아무것도 append하지 않고 반환.
mean_absolute_error·r2_score·pearsonr 를 import만
하고 호출 안 함. 표준 지표 Spearman은  import조차
없음
상태: ✅ #92 해소
────────────────────────────────────────
🔴 P2
문제: 학습 부족 + 베스트 체크포인트 없음
내용: epochs=5 가 전부.
EarlyStopping/ReduceLROnPlateau 를 import했으나
model.fit(callbacks=...) 에 미연결 →
restore_best_weights 없어 마지막  epoch 가중치가
남음. 실피해: fold 4가 epoch 4에서 val_loss 0.0445
 를 봤는데 epoch 5의 0.0481 로 종료
상태: ✅ #92 해소
────────────────────────────────────────
🟠 P3
문제: bean_input 죽은 코드
내용: inputs=[seq_input, bean_input] if 'bean_input'
 in locals() else [...] — bean_input 이 정의된
곳이 없어 항상 else 분기. 우연히 동작하는  코드
상태: ✅ #91 제거
────────────────────────────────────────
🟠 P4
문제: 산출물 없음 / 재현 불가
내용: fold_models 가 메모리에만 존재. 파일명이
_final 인데 재사용 가능한 산출물 0개. TF/NumPy
가중치 초기화 시드 미고정. 별도 test 분할 없음
상태: ✅ #92 해소
────────────────────────────────────────
🟡 P5
문제: 타깃 클리핑 미문서화
내용: np.clip(y,0,1) 이 background 차감 후 음수인
861행(5.7%)을 0으로 누름 → 정확히 0인 타깃이
10.5%(1,570행). sigmoid+MSE 조합에서 zero-inflated
 분포
상태: ⚠️ 잔존 (ADR 0001이 데이터 쪽 결정만 기록)
────────────────────────────────────────
🟡 P6
문제: dinucleotide 피처 중복
내용: 352차원 dinuc 원-핫은 CNN 입력 one-hot에서
완전히 유도 가능. 별도 Dense 분기(354→64)가 새
정보를 안 줌. ridge가 동일 성능인 원인 후보
상태: ⚠️ 잔존 (반박도 입증도 안 됨)
────────────────────────────────────────
🟡 P7
문제: 테스트 없음
내용: cas12a/test/ 디렉터리가 비어 있었음
상태: ✅ #91 해소 (현재 81개)
────────────────────────────────────────
문제: 미사용 import
내용: EarlyStopping, ReduceLROnPlateau,
mean_absolute_error, r2_score, pearsonr,
matplotlib.pyplot 전부
상태: ✅ P1/P2로 해소

데이터 검증은 전부 통과했습니다 (§1): 15,002행 → 타깃 결측 2행 제외 15,000행, 길이 34bp 100%, non-ACGT 0건, 중복 34bp 서열 0건, 중복 23bp protospacer 0건, 위치 4:7이 100% TTT. 중복이 0건이므로 행 단위 KFold 분할에도 누출이 없습니다 — SpCas9 9-head의 PAM-library 중복(테스트셋 25%)이나 SaCas9 replicate 누출과 달리 Cas12a는 깨끗합니다.

---

2. 이슈별 수행 내용

#89 — 세 모델 출력 단위 정의 [CLOSED, needs-info]
리뷰 과정에서 발견한 모델 결함보다 우선하는 문제였습니다. e_system/predict.py 의 MODEL_SCORE_SCALE 이 SpCas9·SaCas9엔 ×100, Cas12a엔 ×1 → Cas12a 예측 0.85(=85%)가 "0.85%"로 표시되고 SpCas9의 30점 뒤로 밀립니다. 지표를 개선해도 이 단위가 어긋난 채로는 사용자에게 도달하지 않습니다. (e_system 은 이번 세션 범위 밖)

#90 — SaCas9 앙상블 기록용 이슈 [CLOSED, needs-triage]
cas12a 리뷰 중 발견한 SaCas9 문제로, 즉시 조치 대상이 아닌 기록용입니다. ①주석은 "5-Fold 앙상블"인데 실제 로드는 fold{1,2,3,4}.pth 4개. ②k-fold 평균 앙상블의 성능은 그 k-fold CV로 측정 불가 — 각 fold 모델이 다른 fold의 validation을 학습에 포함하므로 보고된 CV 수치가 배포된 앙상블을 설명하지 않습니다. → Cas12a는 같은 문제를 피하려고 단일 fold 배포로 결정했습니다.

#91 — 정본화와 공유 피처 모듈 추출 [CLOSED, ready-for-agent] · 동작 변경 없음
- cas12a/ 전체가 git 미추적이었고, models/Cas12a_model_final.ipynb 와 코드가 완전 동일했던 상태를 정리
- features.py(one_hot_encode, extract_seq_features) / data.py(load_cas12a_dataset) 추출 — 피처 정의가 리포지토리에 한 벌만 존재하게
- 테스트 신설(P7) + 합성 fixture 20행 — 실데이터로는 one_hot_encode 의 [0,0,0,0] 폴백과 길이 분기가 전부 죽은 코드라 방어 분기를 하나도 실행할 수 없었기 때문
- P3 bean_input 제거 (린트 F821 사유, 동작·파라미터 수 영향 없음)
- 루트 CONTEXT.md 신설 + ADR 0001 (TTTN 학습 / TTTV 질의)
- 2MB xlsx는 커밋하지 않고 출처 문서화

#93 — 학습 경로 노트북 분리 [CLOSED, ready-for-agent] · 동작 변경 없음, 포팅만
- model.py(create_cas12a_model) / train.py(train_5fold + 진입점) 추출, google.colab.drive 마운트와 /content/... 경로 제거 → python -m cas12a.train
- #91의 "학습 코드는 노트북에 남긴다" 결정을 뒤집었습니다 → ADR 0002
- P1·P2·P4를 의도적으로 고치지 않았습니다 — 포팅과 설정 수정을 같은 커밋에 섞으면 성능 변화가 포팅 탓인지 설정 탓인지 분리할 수 없기 때문. test_train.py·test_model.py 가 깨진 값들을 핀으로 고정해 #92의 변경이 눈에 보이게 만들었습니다
- 패리티는 결정적인 부분만 검증 (전처리 bit-exact / fold 인덱스 골든 / 레이어 21개·파라미터 96,513개). 수렴값은 패리티 기준에서 제외
- 비용 측정: 1 epoch ≈10초, 5-fold 257.5초(4분 18초) → #92의 ready-for-human(Colab 마운트) 근거 해소

#92 — 지표 기록·학습 설정 수정·산출물 저장 [CLOSED, ready-for-agent]
이 모델의 성능 수치가 처음으로 존재하게 됐고, 채택 gate를 통과했습니다.

┌───────────┬───────┬───────┬───────┬─────────┐
│   모델    │  MSE  │  MAE  │  R²   │ Spearma │
│           │       │       │       │    n    │
├───────────┼───────┼───────┼───────┼─────────┤
│ 평균값    │ 0.103 │ 0.283 │ 0.00  │ —       │
│ 예측      │ 9     │ 7     │       │         │
├───────────┼───────┼───────┼───────┼─────────┤
│ RidgeCV   │ 0.048 │       │       │         │
│ (선형,    │ 1 ± 0 │ 0.170 │ 0.537 │ 0.738   │
│ gate      │ .0008 │ 5     │       │         │
│ 기준선)   │       │       │       │         │
├───────────┼───────┼───────┼───────┼─────────┤
│ CNN+BiLST │ 0.045 │ 0.156 │ 0.566 │ 0.7509  │
│ M (#92)   │ 0 ± 0 │ 8 ± 0 │ 8 ± 0 │ ±       │
│           │ .0012 │ .0038 │ .0107 │ 0.0085  │
├───────────┼───────┼───────┼───────┼─────────┤
│ DeepCpf1  │       │       │       │         │
│ (참고선,  │ —     │ —     │ —     │ ≈0.77   │
│ gate      │       │       │       │         │
│ 아님)     │       │       │       │         │
└───────────┴───────┴───────┴───────┴─────────┘

gate 판정: 0.7509 - 0.738 = +0.0129 ≥ 0.0085(fold 표준편차) → 통과 (1.52배)

- P1: metrics.py 가 fold별 Spearman/Pearson/R²/MAE/MSE + 보정(slope·intercept·bias) 기록
- P2: 콜백 실제 연결, epoch 상한 100 (실제로는 13~19에서 정지 → 상한은 구속 조건 아님). 5개 fold 전부 best epoch이 마지막 epoch이 아니었습니다 — 리뷰의 지적이 fold 4의 사고가 아니라 구조적 문제였음이 확인됐습니다
- P4: 최고 fold .keras + 지표 JSON을 같은 디렉터리에, 시드 고정 + 라이브러리 버전 기록. enable_op_determinism() 미적용 / held-out 미분리 → ADR 0003

---

3. 남은 것

1. P5 / P6 는 그대로입니다. 특히 P6는 이번 측정으로 반박되지도 입증되지도 않았습니다 — gate는 넘었지만 96,513 파라미터가 선형 모델 대비 얻은 것이 Spearman +0.013이므로 "별도 Dense 분기가 새 정보를 주지 않는다"는 가설이 여전히 살아 있습니다.
2. GlobalAveragePooling1D 가 BiLSTM 출력의 위치 정보를 평균으로 소거하는 문제도 미해결. P6와 함께 별도 이슈여야 합니다.
3. 다음 구조 변경부터는 held-out 분리가 선행 조건입니다 (ADR 0003). 이번 수치는 구조를 고르는 데 쓰이지 않아 아직 오염되지 않았지만, P6나 pooling 교체를 fold 지표로 판정하기 시작하면 이 수치는 성능 추정치가 아니게 됩니다.
4. e_system/predict.py:60,68 에 정본과 글자 그대로 같은 피처 코드가 아직 남아 있습니다 — 한쪽만 고치면 조용히 어긋나 틀린 예측. e_system 추적 방침(#89)이 정해질 때 별도 이슈로 연결합니다.
5. gate를 넘었으므로 분기상 구조를 손대지 않고 새 산출물로 cas12a_model_fold_4.keras 교체가 다음 단계입니다 (#92 범위 밖).