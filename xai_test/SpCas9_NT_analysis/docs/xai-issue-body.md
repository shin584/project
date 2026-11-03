9-Hydra v3(SpCas9 NT 파인튜닝 모델)의 설명가능성 분석. **단독 분석**이며 v1·v2와의 비교는 범위 밖이다.

설계 정본: `xai_test/SpCas9_NT_analysis/docs/xai-design.md`
관련 ADR: `xai_test/docs/adr/0006-xai-attribution-at-the-encoder-input-embedding.md`, `xai_test/docs/adr/0007-ism-carries-pam-resolution-not-ig.md`

이 이슈는 계획 전체를 담고, 아래 "미결 항목"이 닫힌 뒤 하위 티켓으로 분할한다.

## 분석이 답하는 질문

1. v3이 30bp 창의 **어느 위치**에서 효율을 읽는가 (protospacer / PAM / flank).
2. 공유 trunk가 **head별로 PAM을 다르게 쓰는가** — `SpCas9-NG`는 NGA를, `SpCas9`는 NGG를 인식하므로 PAM 민감도 프로파일이 분리되어야 한다. 분리되지 않으면 그 자체가 결과다.
3. v3이 **어디서 틀리는가**, 그리고 설명이 그 실패를 기술하는가.

범위 밖: v3 재학습·split 변경·튜닝, 그리고 물리적 DNA/단백질 결합에 대한 주장. attribution은 모델 내부의 정보 사용을 기술할 뿐이다.

## 확정된 범위

**모집단** — 9-Hydra Testset 5,186행. `SpCas9/published_split/`를 직접 측정한 결과 test는 train+val과 protospacer가 **완전히 분리**된다(0/5,186행, 30-mer 중복 0). 따라서 `Unseen target / PAM-variant overlap / Exact duplicate` 삼분법은 v1 전용 유물이며 **v3에 적용하지 않는다**. 대신 `in_pam_library`로 두 층을 나누고 **절대 합산하지 않는다**:

| 층 | 행 | distinct protospacer |
|---|---|---|
| 비library | 3,889 | 2,421 |
| PAM library | 1,297 | **5** |

library 층 결과는 "타깃 5개에서 관찰됨"으로 보고하며 1,297행을 독립 표본으로 읽지 않는다.

**head** — `SpCas9`(주축) + `SpCas9-NG`, `evoCas9`(대비). head 간 비교는 세 head가 모두 측정된 **3,851행 교집합**에 고정. 단일 head 절대 수치는 자기 커버리지(5,021 / 3,982 / 5,161)를 쓰고 artifact에서 이름을 구분한다.

**기법** — Integrated Gradients(50 steps), ISM(30위치 × 3염기), Attention Rollout, 위치/PAM 프로파일링, **head 간 attribution 일치도**(다중 head 모델에만 존재하는 축, 비교 모델 불필요).

**비적용(재논의 방지용 명시)** — Tree SHAP / Token-grouped SHAP(v3에 트리 모델·테이블 특성 없음), feature ablation(물리 특성 채널 없음, ADR 0005), paired bootstrap 델타 CI(두 번째 모델 없음).

**토큰 해상도** — NT는 비중첩 6-mer 토크나이저이고 30bp는 정확히 5토큰(`0-5|6-11|12-17|18-23|24-29`)이다. 마지막 토큰이 **PAM(24-26)과 우flank(27-29)를 한 덩어리로** 묶으므로, IG는 **5토큰 해상도까지만** 보고하고 해당 토큰을 `pam_plus_right_flank`로 명명한다. **염기 단위 PAM 결과는 ISM이 담당**한다 (ADR 0007).

**예산** — 한 번의 forward가 9개 head를 모두 반환하므로 head 수는 비용을 곱하지 않는다.

| 분석 | 추론 | 모집단 |
|---|---|---|
| ISM | 3,851 × 91 ≈ 350K forward | **전수 3,851** |
| Attention Rollout | 3,851 forward | **전수 3,851** |
| Integrated Gradients | 400 × 50 × 3 ≈ 60K forward+backward | **층화 400행** |

IG 부분집합은 층당 200행, 층 내에서 NGG/비NGG와 `SpCas9` 오차 사분위로 균형을 맞추고, 시드와 체크섬을 붙인 고정 명단으로 디스크에 쓴다.

**실행 분할** — 경계는 "모델 forward가 필요한가"다. Colab(GPU, Drive의 체크포인트): ISM·IG·rollout 산출. 로컬(체크포인트 불필요, 테스트 포함): IG 부분집합 선정, 케이스 선정, 모든 집계·층화·검증·보고. Colab 드라이버는 `CODE_REF`와 **`transformers` 버전을 핀으로 고정**하고 메타데이터에 기록한다(SaCas9에서 IG가 전부 0이 된 원인이 이 라이브러리 내부였다).

**케이스 선정** — SaCas9의 Concordant/Discordant는 두 모델 정의이므로 승계하지 않는다. 대체: 층별로 ① `SpCas9` 오차 사분위(최상/최하), ② **head 순서 역전**(실측 head 순서와 예측 순서 불일치). attribution 이상치는 선정에서 제외하고 사후 기술에만 쓴다(설명 대상으로 설명 대상을 뽑는 순환 회피).

**검증** — ① 구현: IG 완결성(기여도 합 ≈ f(x) − f(baseline))과 비영 검사를 **테스트 내부에서 만든 작은 랜덤 초기화 9출력 모델**로 로컬 유닛 테스트. ② 해석: head별 IG 토큰 프로파일과 같은 5토큰으로 집계한 ISM 프로파일의 순위 상관을 층별로 보고. 불일치는 버그가 아니라 토큰 수준 attribution의 한계로 보고한다.

**산출물** — `xai_test/SpCas9_NT_analysis/` 독립 서브트리(모듈 + `test/` + `colab_*.ipynb` + `artifacts/`). 루트 SaCas9 분석 모듈은 import하지 않는다(36bp·단일출력·SaCas9 토크나이저 전제). 기법은 재사용하고 구현은 분리한다. 모든 artifact에 체크포인트 digest, 소비한 split 체크섬, `transformers`/`torch` 버전, 분석 head, 층, 모집단/부분집합 명단 digest를 담은 사이드카를 붙인다.

## 수용 기준

- [ ] AC1 — v3 체크포인트와 per-head Testset 예측이 디스크에 고정되고, 소비한 split의 체크섬이 `SpCas9/published_split/split_checksum.json`과 일치함이 검증된다. 불일치 시 분석은 시작하지 않는다.
- [ ] AC2 — 분석 교집합 3,851행이 코드로 재현되고, 세 head의 커버리지(5,021 / 3,982 / 5,161)와 구분된 이름으로 기록된다.
- [ ] AC3 — 두 층(비library 3,889 / library 1,297)이 코드로 재현되고, 어떤 산출물도 두 층을 합산하지 않는다. library 층 보고에는 distinct protospacer 수(5)와 타깃별 분해가 함께 나온다.
- [ ] AC4 — 토크나이즈가 30bp → 5토큰이고 경계가 `0-5|6-11|12-17|18-23|24-29`임을 고정하는 테스트가 있다. IG 산출물에 `pam` 이름의 염기 단위 PAM 키가 존재하지 않는다.
- [ ] AC5 — IG가 인코더 **입력 임베딩**에서 계산되며, 완결성 검사와 비영 검사가 로컬 유닛 테스트로 통과한다. 체크포인트 없이 실행된다.
- [ ] AC6 — ISM(3,851 × 30 × 3)과 Attention Rollout(3,851 × **5 × 5**, 토큰 해상도)이 전수로 산출되고 사이드카를 가진다. rollout을 염기 좌표 30 × 30으로 펼치지 않는다 — 어텐션은 토큰 간 양이고, 펼치면 ADR 0007이 금지한 PAM/flank 혼합이 재발한다(SaCas9는 36 × 36으로 펼쳤으나 이 선택은 승계하지 않는다).
- [ ] AC7 — IG 400행 부분집합이 시드·체크섬 붙은 고정 명단으로 존재하고, 모든 IG 산출물이 같은 명단을 가리킨다.
- [ ] AC8 — head 간 attribution 일치도가 3,851행 교집합에서, 층별로 산출된다.
- [ ] AC9 — 케이스가 층별로 오차 사분위와 head 순서 역전 두 기준으로 선정되고, 선정 로직은 모델 없이 로컬 테스트로 검증된다.
- [ ] AC10 — IG↔ISM 순위 상관이 층별·head별로 보고된다.
- [ ] AC11 — Colab 드라이버가 `CODE_REF`와 `transformers` 버전을 핀으로 고정하고, 두 값이 모든 산출물 사이드카에 기록된다.
- [ ] AC12 — 테스트는 `python -m pytest xai_test/SpCas9_NT_analysis/test -q`로 돌고 통과한다.

## 미결 항목 (티켓 분할 전에 닫는다)

- [ ] v3 per-head 예측이 표준화 스케일인지 역표준화된 라벨 스케일(퍼센트/100)인지. `mu_sd.npy`는 `result/`에 `(2, 9)` float32로 **이미 로컬에 있으므로** 어느 쪽이든 변환은 가능하다 — `SpCas9` head의 mu 0.249가 v2의 퍼센트/100 규약과 일치한다. 남은 건 예측 CSV 자체이고, 그것이 AC1을 막고 있다. (`result/`에는 체크포인트 메타데이터만 있고 가중치와 예측 CSV는 없다. 가중치는 로컬에 필요 없다.)
- [ ] `.npz` 키 스키마와 배열 shape.
- [ ] 층별 케이스 예산과 `case_id` 체계.
