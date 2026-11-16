# 동등성 기준 (#181)

게이트 2는 "데스크톱 화면이 현재 Streamlit 화면과 같은가"를 판정한다. 그 **기준**이
이 폴더다. Streamlit 시연 브랜치를 동결하기 전, 아무도 화면을 건드리기 전에 떴다.

두 가지가 있다.

| | 무엇 | 왜 |
|---|---|---|
| 그림 | `screenshots/` 4~6장 | 문구·표 열·색 규칙·필터·버튼처럼 **보이는 것** |
| 수치 | `candidates.json` · `crispr_prediction_results.csv` · `xai-*.npz` | 점수 소수점·정렬 동점·XAI 배열처럼 **눈으로 안 보이는 것** |

스크린샷만으로는 부족하다. 사람 눈으로는 점수 소수 둘째 자리나 동점일 때의 순서가
보이지 않는다. 그래서 같은 입력의 산출물을 기계 비교가 되는 형식으로 함께 떠 둔다.

## 고정 입력

```
CAAAGTAGCCGGTCGCTTTAAGAATAAATAAAAGCGCCACAGGATTGACTAAACCACAGCCGGCGCCCTGGCCCCCTACCG
```

81bp. `capture_baseline.py`의 `FIXED_INPUT`이 정본이고, 스크린샷도 같은 서열로 찍는다.
#179 게이트 1 스모크가 쓴 서열과 같다.

이 서열을 고른 이유:

- **NGG PAM(AGG·TGG)이 있어 SpCas9이 실제로 지연 로딩된다.** Cas12a만 있는 서열로는
  SpCas9 경로가 안 돌아 기준이 되지 못한다.
- Cas12a TTTV 사이트가 양쪽 가닥에 하나씩 있다.
- SpCas9 변종 9종의 히트맵이 채워지고, 변종이 인식하지 않는 PAM 자리가 `None`으로
  남는 칸도 함께 나온다.
- 사이트 13개 · 변종 예측 38건. '전체 상세 보기'에서 한 사이트를 두 변종이 인식해
  **노란 줄이 2개** 칠해지는 경우까지 들어간다 (1행과 29행).

덮이지 않는 것도 적어 둔다. 지금은 두 Cas 모두 `score_kind="efficiency"`라
Best Candidates가 **한 표**로 나온다. 점수 단위가 섞일 때 표가 Cas별로 갈라지는
가지는 이 입력으로 실행되지 않는다. `MODEL_REGISTRY`에서 SpCas9이 백분위로 돌아가면
기준을 다시 떠야 한다.

## 파일

| 파일 | 내용 |
|---|---|
| `capture_baseline.py` | 기준을 뜨는 스크립트. 코어를 직접 부른다 |
| `candidates.json` | 사이트 목록(정렬 포함) · Best Candidates 표 · 변종 히트맵 · 전체 상세 표 |
| `crispr_prediction_results.csv` | '전체 상세 보기'의 CSV 다운로드와 **바이트 단위로 같은** 파일 |
| `xai-<Cas>-<변종>.npz` | IG 기여도 · SHAP 값 · Rollout 프로파일/행렬 (`np.allclose` 비교용) |
| `capture.json` | 입력·설정·환경·스칼라 값(pred, base_value, 오차) 기록 |
| `compare_baseline.py` | 데스크톱 산출물을 이 기준과 대조한다 |
| `shoot_screenshots.py` | 스크린샷 자동 촬영 (playwright + 설치된 Edge) |
| `screenshots/*.png` | 4개 보기 + XAI 3기법, 6장 (폭 1600px) |
| `screenshots/CHECKLIST.md` | 각 장에 무엇이 보여야 하는지 |

`candidates.json`에는 화면에 보이는 반올림 값과 함께 `_efficiency_pct_full` ·
`_raw_score_full`을 넣어 두었다. 반올림 때문에 동점으로 보이는 행의 **실제 정렬 근거**를
비교하기 위한 것이다.

## 설정

화면의 '계산 설정' 기본값을 쓴다. 바꾸면 SHAP 캐시 키가 달라진다.

- IG 적분 단계 32
- SHAP 샘플 64 · 배경 서열 4
- XAI 대상: 점수가 가장 높은 사이트(화면 기본 선택) + 다른 Cas 종류의 최고 사이트

## SHAP의 재현성

`shap.KernelExplainer`는 전역 `np.random`을 쓰고 seed를 받지 않는다. 그래서 같은 입력을
두 번 계산해도 값이 조금 다르다. 기준을 뜨면서 계산한 SHAP을
`data/precomputed_shap/`에 저장했다 — `kernel_shap()`이 계산 전에 조회하는 그 저장소다.
키는 `(model_key, head, window_seq, nsamples, n_background)` 다섯 개이므로,
두 화면이 같은 사이트·같은 설정으로 SHAP을 열면 **둘 다 같은 캐시를 읽는다.**

그래서 SHAP 비교는 `np.allclose`가 아니라 완전 일치여야 한다. 틀어진다면 캐시를
안 읽은 것이고, 그 자체가 동등성 결함이다. (캐시를 지우고 다시 계산하면 SHAP만
~1e-3 수준으로 흔들린다. IG와 Rollout은 결정적이다.)

## 쓰는 법 (게이트 2)

데스크톱 화면에서 같은 고정 입력으로 같은 이름의 파일을 만들어 한 폴더에 넣고:

```powershell
python docs/equivalence-baseline/compare_baseline.py <데스크톱_산출물_폴더>
```

XAI 배열은 `np.allclose(rtol=1e-5, atol=1e-8)`, CSV는 바이트 비교 후 행 diff,
`candidates.json`은 재귀 비교한다. 종료 코드 0이면 수치 항목은 전부 통과다.
보이는 것은 `screenshots/CHECKLIST.md`로 따로 대조한다.

## 기준을 다시 뜰 때

모델 가중치나 `pam_scanner`·`predict`의 계산이 바뀌면 기준도 다시 떠야 한다.
그때는 **왜 바뀌었는지**를 함께 적는다. 기준이 조용히 바뀌면 게이트 2는 아무것도
판정하지 못한다.

```powershell
python docs/equivalence-baseline/capture_baseline.py
python docs/equivalence-baseline/shoot_screenshots.py
```
