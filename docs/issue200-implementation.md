# #200 — non-NT Cas12a 설명 연결

Cas12a 예측 행에서 DNA 외 추가 입력 없이 IG와 SHAP을 계산한다. #195에서 채택한
`F(x)=M(x,D(x))` 합성 경로를 그대로 사용하며, 기본값은 IG 32단계와 SHAP
64샘플·배경 4개다. 계산 가능한 결과는 기존 로고·막대·워터폴로 표시한다.
수치 기준 미달은 내부에 남기며 별도 품질 경고나 재시도 화면을 추가하지 않는다.

## 연결과 결과 의미

- `NonNTCas12aPredictor.explain_handle("Cas12a")`는
  `non_nt_runtime.cas12a_explain_handle`의 채택 경로를 반환한다. 예측과 설명이
  동일한 실제 적재 checkpoint를 사용하며, 화면/캐시의 헤드는 `Cas12a`다.
- IG는 서열 영벡터와 `D(0)=0`에서 시작한다. GC는 보간 계수에, 인접 염기쌍은
  그 제곱에 비례한다. 34개 위치의 signed 기여도에는 파생 특징의 chain rule
  경로가 포함된다. 특징 기여도를 따로 염기에 나누거나 중요도 비율로 만들지 않는다.
- SHAP은 모든 변경 서열에서 GC 2개와 인접 염기쌍 352개를 다시 계산한다.
  실제 원핫의 특징·출력은 학습 특징 추출기를 사용한 원 모델과 대조한다.
- 결과와 캐시는 raw 확률을 보존한다. 화면은 예측/기준을 %, 기여도/잔차를 %p로
  표시한다. 제한 전 효율과 원출력을 함께 안내하고 표의 0~100% clipping을 설명한다.
- 기존 기술 참고에 합성 경로, 영벡터/soft 입력이 실제 DNA가 아니라는 점,
  무작위 ACGT 배경의 분포 한계와 PAM 변경 가능성을 명시했다. 기여도는
  기준·경로·배경에 의존하며 생물학적 인과관계를 입증하지 않는다.
- `ExplanationIdentity`는 실제 SHA·runtime 설정과 #195의 합성 프로토콜
  `cas12a-cnn-ig-composite-gc-dinuc-zero-midpoint-raw-v1`을 사용한다. 화면 조각과
  계산 결과 캐시는 경로가 달라지면 재사용하지 않는다. 그래프 크기는 키에서 제외한다.
- 기존 계산 설정을 올려 선택적으로 재계산할 수 있고 진행률·취소를 유지한다.
  non-NT Rollout은 attention 층이 없다는 사유와 함께 비활성화한다.

`xai.quality`는 Cas12a IG 완결성/SHAP 가법성에 동결 절대 오차 `0.001 raw`
(0.1 %p)를 적용한다. NT의 상대 허용 오차를 Cas12a에 적용하지 않는다.
유한값·길이 검사는 유지하며 수치 판정은
`%LOCALAPPDATA%/dashboard-v3/logs/non_nt-xai.jsonl`에 기록한다. 유한한 품질 실패는
캐시에 보존하고 그래프를 표시한다. 계산 예외·NaN/inf는 기존 오류 처리를 따른다.

## 검증의 근거와 범위

수치 검증기는 [#195 계약](runtime/cas12a-ig-contract.json)의 C1–C4·합성 경로와
[#193 계약](runtime/contract.json)의 SHAP 기준을 사용한다. 기준을 완화하지 않았다.
측정 전에 관련 소스와 계약을 zip으로 보존하고 SHA를 등록한다. IG32/256의
완결성·해상도, soft 지점의 원함수 일치·chain rule, SHAP 기본값의 seed0/1/2·동일
seed 반복 및 512샘플·배경32 비교를 수행한다. SHAP이 실제 모델에 넘긴 모든
변경 서열의 354개 특징과 원함수 일치를 검사하고 원시 입력·결과 배열도 보존한다.

화면 검증은 숨겨진 실제 WebView2와 non-NT 계산 자식 프로세스를 사용한다.
이 검증의 성공은 수치 품질의 통과와 구분한다. 소스는
[수치 검증기](runtime/verify_issue200.py)와
[화면 검증기](../tests/non_nt_xai_window_probe.py)다.

회귀 범위는 기존 Cas12a 특징/합성 handle·예측기, 설명 캐시, 데스크톱 화면,
실제 계산 프로세스/CSV 경계다. 공통 품질 기록을 변경했으므로 SpCas9 설명과
NT 단위/품질 판정 보존도 포함했다. 저장소 전체 테스트는 실행하지 않는다.

새 배포본·깨끗한 오프라인 Windows 검증과 정식 RAM 비교는 #203/#204 범위다.
이번 수치·화면 결과를 #205 검증 묶음과 합치거나 기존 보관 자료를 변경하지 않는다.
held-out 예측 성능과 생물학적 정확성은 이 작업의 검증 대상이 아니다.

## 실행 기록

각 실행의 수를 별도로 기록하며, 반복 실행한 테스트를 합산하지 않는다.
구현 전 기준선은 다음 명령으로 **21 통과 / 0 실패 / 0 건너뜀**이었다.

```powershell
py -3.13 -X utf8 -m pytest test/test_cas12a_ig.py test/test_non_nt_cas12a.py -q -p no:cacheprovider
```

새 예측기 연결 검사는 아래 node ID에서 기존 `NotImplementedError`로
**0 통과 / 1 실패 / 0 건너뜀**을 확인한 뒤 구현했다.

```powershell
py -3.13 -X utf8 -m pytest tests/test_cas12a_xai.py::test_predictor_uses_adopted_single_sequence_handle_and_identity -q -p no:cacheprovider
py -3.13 -X utf8 -m pytest tests/test_cas12a_xai.py -q -p no:cacheprovider
```

구현 후 새 파일은 **9 통과 / 0 실패 / 0 건너뜀**이다. 최종 회귀와 실제 프로세스
검증 명령은 다음과 같다.

```powershell
py -3.13 -X utf8 -m pytest test/test_cas12a_ig.py test/test_non_nt_cas12a.py test/test_non_nt_runtime.py test/test_explain_cache.py test/test_cache_identity.py tests/test_cas12a_xai.py tests/test_non_nt_xai.py tests/test_desktop_xai.py tests/test_worker_cache.py tests/test_nt_xai_view.py test/test_nt_explain.py::test_quality_failure_stays_displayable_and_nonfinite_result_is_rejected -q -p no:cacheprovider
# 151 passed / 0 failed / 0 skipped; 213.76s
py -3.13 -X utf8 -m pytest tests/test_non_nt_cas12a_process.py -q -p no:cacheprovider
# 2 passed / 0 failed / 0 skipped; 77.26s
py -3.13 -m pyright non_nt_cas12a_predict.py desktop/worker_renderer.py desktop/xai_view.py desktop/screen.py xai/quality.py docs/runtime/verify_issue200.py tests/non_nt_xai_window_probe.py
# 0 errors / 0 warnings
py -3.13 -m ruff check --select F,E9 non_nt_cas12a_predict.py desktop/worker_renderer.py desktop/xai_view.py desktop/screen.py xai/quality.py docs/runtime/verify_issue200.py tests/test_cas12a_xai.py tests/test_non_nt_xai.py tests/test_non_nt_cas12a_process.py tests/non_nt_xai_window_probe.py test/test_non_nt_cas12a.py
# All checks passed
git diff --check
```

회귀 실행의 경고 3건은 기존 `test_worker_cache.py`의 작은 SHAP fixture에서 발생한
특이행렬/최소제곱 경고다. 수치 기준을 바꾸거나 경고를 숨기지 않았다.

실제 화면 실행은 다음과 같다.

```powershell
py -3.13 -X utf8 tests/non_nt_xai_window_probe.py .packaging/evidence/issue200/window01 Cas12a
# 15 passed / 0 failed / 0 skipped; exit 0
```

[화면 보고서](benchmarks/issue200-xai-20261119-run01/window/report.json)는
**15 통과 / 0 실패 / 0 건너뜀**이다. 단일 헤드·manifest PAM 4–7 좌표,
IG 로고와 SHAP 막대·워터폴, 기본값, 크기 변경 시 PNG 재사용, 별도 품질 경고 미표시,
예측 표 유지, 사이트 전환 시 이전 계산 취소, 정밀 SHAP 진행률·취소·기본 결과 재선택을
확인했다. UI 프로세스에 모델 라이브러리가 없으며 종료 후 계산 자식도 닫혔다.
저장한 [IG](benchmarks/issue200-xai-20261119-run01/window/ig-0.png),
[SHAP 막대](benchmarks/issue200-xai-20261119-run01/window/shap-0.png),
[워터폴](benchmarks/issue200-xai-20261119-run01/window/shap-1.png)을 직접 열어 확인했다.

## 실제 모델 수치 결과

```powershell
py -3.13 -X utf8 docs/runtime/verify_issue200.py docs/benchmarks/issue200-xai-20261119-run01
# numerical: 85 passed / 32 failed / 0 skipped; exit 1; 332.85s
```

[수치 보고서](benchmarks/issue200-xai-20261119-run01/report.json)는
**85 통과 / 32 실패 / 0 건너뜀**이다. pytest의 153개 회귀 통과나 화면의 15개
통과와 합산하지 않는다. 실패는 IG 완결성 4건, SHAP seed 간 안정성 24건,
기본값과 정밀 설정의 상관 비교 4건이다. 동결 기준을 만족한 설명이라고 보고하지 않는다.

| 검사 | 결과 |
| --- | --- |
| 실제 모델 IG/raw/원함수 일치, alpha 0·0.25·0.5·1 | 16/16 통과 |
| 실제 모델 chain rule, alpha 0.5·1, 최대 오차와 cosine 각각 검사 | 16/16 통과 |
| SHAP 변경 입력의 354개 특징·원함수 일치 | C1–C4 모두 최대 절대 오차 0 |
| IG32→256 해상도 안정성: 상관·최대 차이·유의 위치 부호 | 12/12 통과 |
| SHAP 가법성: seed0/1/2·동일 seed 반복·정밀 설정 | 20/20 통과 |
| SHAP 동일 seed 반복 | 4/4 통과, 최대 차이 0 |
| SHAP seed 간 상관/최대 차이 | 0/24 통과 |
| SHAP 기본64/4 대 정밀512/32 상관 | 0/4 통과 |
| 저장된 출력·gradient·기여도·진단 배열 유한값 | C1–C4 모두 통과 |

SHAP 특징/원함수 대조는 **70,036개 모델 전달 행**(중복 포함)을 검사했다.
각 사례에서 17,509행을 처리했으며 모든 변경 입력의 원핫 배열은
[raw.npz](benchmarks/issue200-xai-20261119-run01/raw.npz)에 보존했다.

IG 완결성 절대 오차는 다음과 같다. 모든 칸의 기준은 `<=0.001 raw`다.

| 사례 | IG32 | IG256 |
| --- | ---: | ---: |
| C1 | 0.000590845 | 0.000069462 |
| C2 | **0.003990365** | **0.001227337** |
| C3 | **0.001888127** | 0.000099079 |
| C4 | **0.001309912** | 0.000020229 |

기본 IG는 1/4, 256단계는 3/4만 완결성을 통과했다. 위치별 해상도 상관은 모두
0.999985 이상이지만 완결성 통과와 같은 의미가 아니다. SHAP의 동일 seed
재현성과 가법성이 통과했어도 seed 간 상관은 -0.126532~0.682726으로 기준 0.9에
미달했다. 기본값과 정밀 설정의 상관도 0.223376~0.741399로 모두 미달했다.
따라서 정밀 설정을 보편적으로 검증된 설정으로 승격하지 않는다.

수치 보고서의 소스 불변 검사와 다음 보관 자료의 SHA-256 대조를 통과했다.
[등록 파일](benchmarks/issue200-xai-20261119-run01/registration.json),
[측정 소스·계약](benchmarks/issue200-xai-20261119-run01/measured-source.zip),
원시 배열의 checksum을 보고서가 연결한다. `.gitattributes`는 이 디렉터리의
정확한 바이트를 유지한다. #195의 기존 특징 parity/soft gradient 회귀도 통과했으며,
#205 보관 묶음은 변경하지 않았다.

## Standards

작업 시작점 `fcf9210` 대비 독립 리뷰: 문서화된 규칙 위반 **0건**,
수정이 필요한 코드 냄새 **0건**. 기존 예측기·캐시·화면 경계를 유지하며 코어에
UI 의존성을 추가하지 않는다. #205 보존 자료를 수정하지 않는다.

## Spec

독립 리뷰: #200 요구사항의 누락·오구현·범위 확장 지적 **0건**.
합성 경로와 특징 재계산, 출력 단위, 기본 설정, 캐시 식별, 계산 가능한 결과의
표시 유지와 선택 정밀 계산이 명세와 일치한다. 수치 검증의 32개 실패는 위에 별도로
보고하며 구현·표시 성공으로 대체하지 않는다.
