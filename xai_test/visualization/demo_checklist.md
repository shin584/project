# 시연 전 수동 체크리스트 (Model B 설명 대시보드)

실행: `cd xai_test/visualization && streamlit run app.py`
(사전 조건: `xai_test/test_metadata.csv`, `xai_test/final_analysis_result/` 3개 파일)

자동 테스트(`pytest visualization/`)가 통과한 뒤, 아래 항목을 **비전문가 관점**에서 직접 확인합니다.
대표 샘플: IG 있는 샘플(예: 216), IG 없는 샘플(예: 1), 경계값 0 / 513.

## 공통
- [ ] 화면 어디에도 Model A 예측값, Concordant/Discordant, case_id 문구가 보이지 않는다
- [ ] 사이드바에서 "IG 있음" 표시만 보고 IG 계산 샘플을 찾을 수 있다
- [ ] 상단 지표(예측값·실측값·오차)가 한눈에 읽힌다

## 탭 1 최고 후보군
- [ ] 예측 점수 내림차순이고, 선택 샘플의 순위 안내가 맞다

## 탭 2 위치별 히트맵
- [ ] PAM(25–30) / Seed(17–24) / Distal(0–7) 구간 표시가 위치 눈금과 정확히 맞는다
- [ ] 빨강=상승, 파랑=하락 설명과 색이 일치하고, 진한 칸의 글자도 읽힌다
- [ ] 복합 미스매치 막대 15개의 구간 색 범례가 이해된다

## 탭 3 예측 상세
- [ ] 예측값, 실측값, 오차, MFE/ΔG/Tm/GC 값과 단위 설명이 보인다

## 탭 4 XAI 분석
- [ ] Attribution Logo: 양수는 축 위, 음수는 축 아래로 보이고 글자가 잘리지 않는다
- [ ] Deterministic Projection Rule 가정 문구가 Logo 바로 위에 보인다
- [ ] IG 없는 샘플에서 "IG 미계산" 안내가 보인다
- [ ] Rollout 프로파일과 Logo의 위치 x축이 세로로 정렬된다
- [ ] "물리적 결합이 아님" 주석이 보이고, 36×36 히트맵은 펼쳐 보기 안에만 있다
- [ ] SHAP waterfall에 11개 묶음이 모두 보이고 `[CLS]`가 전역 문맥으로 표기된다
- [ ] SHAP 합산값 f(x)와 실제 예측값 차이 안내 문구가 이해된다
