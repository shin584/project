# Model B 설명 대시보드 계획

> 용어는 `xai_test/CONTEXT.md`를 따른다. 범위 결정의 근거는 `docs/adr/0002-dashboard-explains-model-b-only.md` 참고.

**1. 요구 사항**

* 예측 점수가 블랙박스로 남지 않도록, Model B가 점수를 내린 과정과 근거(서열 위치별 기여도, 열역학 지표의 기여 등)를 사용자가 직관적으로 이해할 수 있게 시각화해야 합니다.
  * 대시보드의 목적은 **Model B 단독의 판단 근거 설명**이며, Model A와의 비교 분석은 범위에 포함하지 않습니다.
  * 대상 단백질은 **SaCas9 전용**입니다 (XAI 파이프라인이 SaCas9에만 존재).

* 5억 개 이상의 파라미터를 가진 파운데이션 모델(NT regression checkpoint)의 구동 부하를 극복하고 원활한 시연이 가능해야 합니다. 이를 두 단계로 나누어 대응합니다.
  * **1단계 (현재, 초안)**: Testset에 대해 Colab GPU에서 사전 계산한 결과만 렌더링하여 실시간 NT 구동을 회피합니다.
  * **2단계 (추후)**: 학교 서버 GPU에서 실시간 추론/XAI 계산을 수행하고, 대시보드의 데이터 소스 구현체만 교체합니다. 2단계의 세부 아키텍처(추론 서버 분리 여부 등)는 서버 환경 확인 후 결정합니다.


**2. 시스템 설계 (합성 및 분석)**

* `xai_test/visualization/`에 독립 Streamlit 앱으로 작성합니다. `e_system/app.py`의 레이아웃·탭·session state 뼈대만 차용하고 코드는 공유하지 않습니다.

* **입력**: 1단계에서는 81bp 서열 입력 대신 사이드바의 `sample_id` 선택기(Testset 0–513)를 사용합니다. 2단계에서 서열 입력으로 교체합니다.

* **데이터 소스 경계**: UI는 "샘플 하나의 XAI 묶음"을 반환하는 데이터 소스 인터페이스에만 의존합니다.
  * 1단계 구현체: `final_analysis_result/` 및 추가 export 파일(아래)을 읽는 사전 계산 데이터 소스.
  * 2단계 구현체: 학교 서버 실시간 계산 데이터 소스.
  * `true_score`와 오차는 Testset에서만 존재하는 optional 필드로 다루며, 없으면 UI에서 자동으로 숨깁니다.
  * Integrated Gradients는 Case Study 15개에만 존재하므로 optional 필드이며, "없음"이 명시적으로 표현되어야 합니다.
  * Concordant/Discordant 분류는 Model A와의 비교로 정의되므로 UI에 노출하지 않습니다. Case Study 15개는 "IG 사용 가능 샘플"로만 활용합니다.

* **선행 데이터 작업**: Testset 514개 전체에 대한 Model B 예측값과 물리 피처 4개(MFE, ΔG, Tm, GC) 값을 Colab에서 export합니다 (현재는 Case Study 15개에만 존재).

* **다중 탭 구조**
  1. **최고 후보군**: Testset을 Model B 예측 점수순으로 정렬한 랭킹.
  2. **위치별 히트맵**: 선택 샘플의 ISM delta (36 위치 × 3 대체 염기) 히트맵. PAM(위치 25–30), Seed(17–24), Distal(0–7) 구간을 표시. 아래에 복합 미스매치 15개 시나리오의 점수 변화 막대그래프. (Single mismatch delta는 ISM과 중복되므로 제외)
  3. **예측 상세 표**: `sample_id`, 서열, Model B 예측값, (Testset 전용) 정답·오차, 물리 피처 4개 값.
  4. **XAI 분석 뷰** (선택 샘플, 위치 x축을 공유하도록 세로 정렬)
     * **Attribution Logo**: 실제 염기 글자 높이 = 부호 있는 IG attribution. Deterministic Projection Rule 가정을 화면에 명시. IG가 없는 샘플은 "IG 미계산" 안내를 표시.
     * **Attention Rollout 위치 프로파일**: 위치별 정보 의존도를 합산한 36칸 막대가 기본이며, 원본 36×36 히트맵은 펼쳐 보기로 제공. "모델 내부 정보 의존성이며 물리적 결합이 아님" 주석 표시.
     * **Token-grouped SHAP waterfall**: 임베딩 SHAP을 7개 토큰(6-mer 6개 + `[CLS]` = 전역 문맥)으로 부호 있는 합산, 물리 피처 4개는 개별 표시 — 총 11개 묶음.


**3. 구현**

* 웹 프론트엔드 및 대시보드: Streamlit, Pandas.
* 시각화: Matplotlib (ISM 히트맵, 복합 미스매치 막대, Rollout 프로파일), `logomaker` (Attribution Logo, 신규 의존성), SHAP (`shap.Explanation` + `shap.plots.waterfall`로 Token-grouped SHAP 렌더링).
* 인터랙티브 차트 라이브러리(plotly 등)는 사용하지 않습니다.


**4. 시험평가**

* **자동 테스트**
  * 데이터 소스 계약 테스트: 514개 전체 샘플에 대해 배열 shape, NaN 부재, optional 필드(IG, 정답)의 "없음" 표현을 검증.
  * Streamlit `AppTest` 렌더링 테스트: 대표 샘플(IG가 있는 Case Study 샘플, IG가 없는 샘플, 경계 `sample_id` 0/513)에 대해 각 탭이 예외 없이 렌더링되고 필수 요소가 존재하는지, IG 미계산 안내가 표시되는지 검증.
* **수동 체크리스트** (시연 전): 각 시각화가 비전문가에게 직관적으로 읽히는지, 주석·가정 문구가 노출되는지 확인.
