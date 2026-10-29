cas12a/
├── __init__.py                 패키지 계약 명시: features/data 는 순수 NumPy·pandas,
│                               model/train 부터 TF. e_system 임포트 없음
│
├─ ─ ─ 데이터 경로 (TF 없이 임포트 가능) ─ ─ ─
├── dataset/
│   └── kim_2018_cas12a_dataset.xlsx     원본 Kim 2018 테이블 (2.0MB, usable 15,000행)
├── data.py          (100줄)   xlsx/csv 읽기 → 시퀀스 정규화 → load_cas12a_dataset()
│                               이 반환하는 (X_seq, X_feat, y) 가 훈련 입력
├── features.py       (92줄)   34bp 윈도우 규격. one-hot + GC 2개 + 22×16 다이뉴클
│                               → FEATURE_DIM. PAM=slice(4,8), protospacer=slice(8,31)
├── units.py          (47줄)   점수 단위 변환(clip/percent/penalty). 훈련과 무관,
│                               예측 점수를 쓰는 쪽에서 사용
│
├─ ─ ─ 모델·훈련 경로 (TF 필요) ─ ─ ─
├── model.py          (85줄)   create_cas12a_model() — 3-커널 병렬 CNN(4/8/15bp)
│                               + BiLSTM, 파라미터 96,513개, Adam lr=5e-4
├── train.py         (14KB)    ★ 진입점: python -m cas12a.train
│                               make_kfold() / train_5fold() / save_artifacts()
│                               KFold(5, shuffle, seed=42) 고정, EPOCHS=100 상한
│
├─ ─ ─ 평가 경로 ─ ─ ─
├── metrics.py       (188줄)   fold_metrics() 폴드 채점 → aggregate() 집계
│                               → gate_verdict() 판정 → format_summary() 출력
│                               게이트: BASELINE_SPEARMAN=0.738 (RidgeCV, 같은 split)
│                               참고선: REFERENCE_SPEARMAN=0.77 (DeepCpf1, 게이트 아님)
│
├─ ─ ─ 산출물 ─ ─ ─
├── artifacts/
│   ├── cas12a_5fold_metrics.json        git 추적 ○ — 재현성 기록(숫자가 작음)
│   └── cas12a_best_fold4.keras          git 추적 ✗ — .gitignore (가중치는 큼)
│
├─ ─ ─ 테스트 (81개) ─ ─ ─
├── test/
│   ├── test_data.py      17개      test_metrics.py   20개
│   ├── test_features.py  14개      test_train.py     16개
│   ├── test_model.py      6개      test_units.py      8개
│   └── fixtures/synthetic_cas12a.csv    실제 xlsx 없이 돌리는 합성 입력
│
└─ ─ ─ 문서 / 원본 ─ ─ ─
    ├── model/Cas12a_model_final.ipynb   포팅 전 원본 노트북 (이제 참조용)
    └── docs/
        ├── cas12a-model-review.md       §2 ridge 베이스라인, §3 P1·P2·P4 리뷰 지적
        ├── cas12a-5fold-metrics.md      #92 측정 결과 보고
        └── cas12a-training-cost.md      #93 CPU 비용 측정 (epoch당 ~10s)