# XAI dashboard desktop

81bp DNA에서 SpCas9·Cas12a 사이트를 찾고 예측과 XAI를 보여주는 Streamlit 앱이다.
이 폴더에 #179 패키징, #178 데스크톱 호출 경계, #151 코어 분리,
#180 취소 콜백을 통합했다. 코어·화면·모델·데이터가 모두 이 폴더 안에 있으며,
형제 `xai_dashboard` 체크아웃이나 별도 워크트리가 필요하지 않다.

- `desktop/`: 데스크톱 화면, 코어 예측 Renderer와 가중치 없는 미리보기. [경계 문서](docs/desktop-boundary.md).
- `predict.py`, `pam_scanner.py`, `xai/`: 공용 코어와 기존 Streamlit 화면.
- `models/`, `data/`: 로컬 가중치 두 개와 사전계산 데이터.
- `packaging/`: PyInstaller 오프라인 실험. [빌드 절차](packaging/README.md).

통합 출처와 검증 결과는 [통합 기록](docs/integration-179-178-151-180.md)에 있다.
데스크톱은 Best Candidates·변종 비교와 #187의 실제 XAI 보기를 구현했다. Best Candidates와 변종 비교는 실제 코어 예측을
Streamlit과 같은 열·문구·정렬·색 규칙으로 표시한다. 서로 다른 `score_kind`는
표를 나누며, Cas12a는 Best Candidates에 포함하고 SpCas9 변종 비교에서 제외한다.
전체 상세 보기의 필터·저장 대화상자는 #185에서 연결한다.

## 실행과 테스트

Windows에서 데스크톱 화면을 실행한다 (WebView2와 아래 코어 의존성 필요):

```powershell
python -m pip install pywebview==6.2.1
python -m desktop
# 모델 가중치 없이 화면과 가짜 XAI를 확인할 때:
python -m desktop --preview
```

창의 "모델 준비 중" 표시가 "모델 준비 완료"로 바뀌면 분석할 수 있다.
서열 입력 후 네 보기를 전환해도 예측을 다시 하지 않는다. 기본 실행은 실제
모델 점수와 XAI를 사용한다. XAI 보기에서 사이트·헤드와 Attention Rollout·IG·SHAP을
선택하고 계산 설정을 바꿀 수 있다. 설정 범위와 기본값은 Streamlit과 같다.
크기 슬라이더와 탭 전환은 계산 요청을 보내지 않으며, 같은 설명을 다시 선택하면
공용 XAI 메모리 캐시를 사용한다. 계산 중 다른 사이트·헤드·기법·설정을 고르면
이전 요청이 취소된다. IG·SHAP은 진행률을 표시하고, 취소 버튼도 제공한다.
Rollout은 순전파가 끝난 후 지난 결과를 버린다. Model B(SaCas9)는 표시하지 않는다.
`--preview`에서는 가짜 점수와 가짜 XAI를 표시한다. 미리보기의 점수와
XAI를 실제 분석에 사용하면 안 된다. 미리보기에는 torch나 모델 가중치가 필요 없다.
HTML·CSS·JS는 문자열로 로드하고 그림은 data URI를 사용한다.
Windows 세션의 이름 있는 mutex를 창 생성 전에 잡으므로, 다른 작업 폴더에서
다시 실행해도 두 번째 창이나 모델 작업자가 생기지 않는다. 종료·강제 종료 시
Windows가 핸들을 정리하므로 잠금 파일을 지울 필요가 없다.

기존 Streamlit 화면 실행과 전체 테스트:

```powershell
python -m pip install streamlit numpy pandas matplotlib logomaker shap torch transformers safetensors gdown pytest
python -m streamlit run app.py
python -X utf8 -m pytest test tests -q
```

가중치는 `models/SpCas9.safetensors`, `models/Cas12a.safetensors`에 둔다.
가중치가 없으면 `model_source.py`에 등록된 Drive 주소로 다운로드를 시도한다.
토크나이저는 최초 실행 시 Hugging Face에서 받는다. 부스에서는 모델과 토크나이저를
미리 준비하고, 실제 사용할 서열로 각 기법을 한 번씩 실행해 둔다.
테스트는 가짜 모델을 사용하므로 가중치·토크나이저 다운로드 없이 실행된다.
`tests/test_desktop_shell.py`는 Windows에서 `pywebview`, `psutil`이 설치되어 있으면
실제 WebView2 창으로 시작 상태·입력·보기 전환·중복 실행·HTTP 서버 미생성을 검증한다.
데스크톱이 이미 실행 중이면 실제 창 테스트는 건너뛰며, 창을 닫고 다시 실행하면 된다.
추가 의존성은 `python -m pip install pytest psutil pywebview==6.2.1`로 설치한다.
타입 검사는 `python -m pip install pyright` 후
`python -m pyright desktop/app.py desktop/instance.py desktop/screen.py desktop/__main__.py tests/test_desktop_shell.py tests/desktop_shell_probe.py`로 실행한다.
`test/`는 #151의 코어 테스트이며 `numpy`, `pytest`, `transformers`만으로 실행한다.
`tests/`에는 SHAP·진행률·Streamlit 통합 테스트가 있다.
`tests/test_nt_cancellation.py`는 설치된 PyTorch의 소형 메모리 텐서로 실제 설명 핸들을
검증하며, PyTorch가 없으면 건너뛴다. 사전학습 가중치는 사용하지 않는다.

## #150 통합 범위

[#150의 최신 정리](https://github.com/shin584/project/issues/150#issuecomment-6794974525)에
따라 #153의 진행률·IG 청킹과 #152의 선택 워밍업·SHAP 디스크 캐시를 통합했다.
#151에서 `predict.py`의 UI 의존성을 제거했다. #154의 facade는 별도 작업이다.
`load_predictors`와 `load_nt_tokenizer`는 프로세스 수준 `functools.cache`로
재사용하므로 같은 인자의 재실행마다 모델과 워밍업을 반복하지 않는다.
모델 파일이나 설정을 바꾸면 앱을 재시작한다. 코드에서 캐시를 초기화할 때는
`load_predictors.cache_clear()`와 `load_nt_tokenizer.cache_clear()`를 사용한다.
사용하지 않던 거리 보정 점수(`final_score`)는 제거했으며 표시 점수와 정렬 기준은 유지한다.
`app.py`는 변경하지 않았고 화면 변경은 #153의 `xai/view.py`에 한정한다.

- 워밍업은 `WARMUP_KEYS = (("A", "Cas12a"),)`만 대상으로 한다.
  SpCas9은 실제 요청까지 지연 로딩되며, `load_predictors(warmup=False)`로 워밍업을 끌 수 있다.
  워밍업 실패는 로그로 남기고 첫 실제 요청에서 다시 로딩한다.
- 디스크 캐시는 `(model_key, head, window_seq, nsamples, n_background)`를 모두 구분한다.
  `seed=0`일 때만 조회하며, 설정이 달라지면 실제 계산으로 넘어간다.
- 캐시 적중 시 추론과 `on_batch`를 호출하지 않는다. 미적중 시 실제 추론 배치마다
  진행률을 전달하고, 계산 완료·캐시 적중·예외 시 화면 래퍼가 진행률 영역을 비운다.
  콜백의 `done/total` 단위는 #153 구현의 **누적 서열 수**이고, 배치 번호가 아니다.
- IG는 최대 8개 적분 단계씩 역전파한다. #180에서 청크 완료 콜백을 추가했으며,
  화면이 선택적으로 사용한다.

## 계산 중 취소 (#180)

ADR 0001의 선택 콜백 `on_batch(done, total)`을 사용한다. 코어는 화면의
`Cancelled` 타입을 import하지 않는다. 화면이 정의한 예외를 콜백에서 던지면
IG·SHAP·예측 모두 호출자까지 그대로 전파하고, 같은 예측기를 다음 계산에 다시 쓸 수 있다.
콜백을 생략한 기존 호출은 그대로 동작한다.

- `integrated_gradients(handle, window_seq, steps=64, chunk_size=8, on_batch=None)`:
  각 청크를 마친 뒤 **완료한 적분 단계 수 / steps**를 보고한다. 마지막 청크가 짧아도
  정확히 `(steps, steps)`에 도달한다. 이후 완결성 확인용 출력 평가는 진행률에 포함하지 않는다.
- `predict_batch(sites, on_batch=None)`와 `run_prediction(scan, predictors, on_batch=None)`:
  각 추론 배치를 마친 뒤 **완료 서열 수 / 해당 예측기의 전체 서열 수**를 보고한다.
  `run_prediction`은 콜백을 그대로 전달하므로 모델이 바뀌면 서열 수가 다시 시작한다.
  빈 입력과 미연동 예측기는 콜백을 호출하지 않는다.
- `kernel_shap(..., on_batch=None)`의 기존 배치 콜백도 취소 예외를 전파한다.
  사전계산 캐시 적중 시에는 계산과 콜백 없이 반환한다.

**모델 로딩과 Attention Rollout은 중간 취소하지 않는다.** 로딩은 끝까지 수행하고,
Rollout도 한 번의 순전파를 완료한다. 화면은 이 구간 이후 요청 ID를 확인해 지난 요청의
결과를 버려야 한다. 코어의 취소 콜백이 이 두 구간을 끊지는 않는다.

계산은 작업자 하나에서 직렬로 실행한다. 특히 IG는 예측기에 마스크·토큰 구간·길이를
보관하므로, 취소 예외가 돌아와 이전 작업이 정리된 다음 새 작업을 시작해야 한다.

실제 가중치 읽기는 `predict` 로거의 INFO 메시지 `Loading model weights:`로
관찰할 수 있다. 화면 파일을 수정하지 않고 로그를 켜려면 다음처럼 실행한다.

```powershell
python -c "import logging, sys; logging.basicConfig(level=logging.INFO); from streamlit.web import cli; sys.argv = ['streamlit', 'run', 'app.py']; cli.main()"
```

기본 워밍업 대상인 Cas12a의 로드 메시지는 입력칸을 5번 변경해도 1개다.
SpCas9 사이트를 실제로 예측하면 그 모델의 로드 메시지가 별도로 1개 생긴다.
2026-11-16에는 실제 로컬 가중치와 오프라인 토크나이저로 변경하지 않은 `app.py`를
AppTest에서 실행해 확인했다. 입력칸 5회 변경과 분석 버튼 1회 실행 뒤에도
Cas12a 가중치 로드 로그는 1개였고, SpCas9은 로드되지 않았다.
[검증 기록](docs/benchmarks/core-decouple-2026-11-16.json)에 환경과 관측값을 남겼다.

## 사전계산과 시연 범위

```powershell
python make_precomputed_shap.py <81bp-DNA> [<추가-81bp-DNA> ...]
```

현재 저장소에는 시연 입력 **1개에서 얻은 Cas12a 사이트 2개**의 사전계산 파일이 있다.
파일의 키는 기본 설정 `nsamples=64`, `n_background=4`다.
#150에서 제안한 입력 3~5개의 커버리지는 아직 확보하지 않았다.
위 명령으로 선정한 시연 입력을 추가할 수 있다. 모델 파일을 교체했다면 앱을 재시작하고
기존 사전계산 파일을 별도로 보관한 뒤 다시 생성한다. 캐시 키에는 가중치 해시가 없다.

## 실행 시간과 검증

아래는 기존 보고값이며 이 통합 환경에서 새로 측정한 값과 구분한다.

| 항목 | 기존 CPU 측정·보고 | 출처 |
|---|---:|---|
| 모델 로드 | 3.7초 | #150, torch 2.12.0+cpu |
| 워밍업 없는 첫 추론 | 9.1초 | #150 |
| 워밍업 이후 추론 1개 | 0.44초 | #150 |
| 워밍업 이후 배치 8개 | 0.83초 | #150 |
| 워밍업 적용 후 첫 예측 | 약 2초 | PR #168 작성자 보고 |

출처: [#150](https://github.com/shin584/project/issues/150),
[PR #168](https://github.com/shin584/project/pull/168).
워밍업은 비용을 앱 시작 시점으로 옮기는 것으로, 프로세스 시작 자체를 즉시 끝내지는 않는다.
순방향 워밍업과 IG의 첫 역전파는 서로 다른 경로다.

2026-11-14 통합 브랜치에서 기존 로컬 Cas12a 가중치를 읽어 다음을 측정했다.
Windows x86-64, Intel Family 6 Model 142, CPU 스레드 4개,
Python 3.12.16, torch 2.15.0, transformers 5.23.0,
numpy 2.5.4, shap 0.54.1, streamlit 1.67.0 환경이다.
각 값은 1회 관측으로, 위 기존 환경과 직접적인 성능 비교를 뜻하지 않는다.

| 항목 | 통합 환경 실측 |
|---|---:|
| 최초 시작 + Cas12a 워밍업 (토크나이저 최초 다운로드 포함) | 89.763초 |
| 새 프로세스 시작 + 워밍업 (토크나이저 준비 후, 오프라인 재확인) | 18.692초 |
| 워밍업 후 첫 예측 1개 | 0.964초 |
| 다음 예측 1개 | 0.419초 |
| 워밍업 후 배치 8개 | 2.234초 |
| `load_predictors()` 캐시 적중 | 0.004초 |
| IG 첫 호출 (32단계, 청크 8개) | 18.577초 |
| SHAP 디스크 캐시 적중 (64/4) | 0.076초 |

SpCas9 가중치는 이 과정 전체에서 로드되지 않았다. SHAP 적중 시 진행률 콜백은 0회였다.
가중치 없는 테스트는 기존 39개와 통합 경계 6개를 합쳐 **45개 통과**했다.
통합 테스트는 설정 변경 시 재계산·진행률, 디스크 캐시와 화면 캐시의 연속 적중,
재실행 5회 후 워밍업 1회 유지, `warmup=False`를 검증한다.

실제 가중치로 청크 8개와 전체 32개를 한 배치로 계산한 IG를 비교했을 때
위치별 기여도의 최대 절대 차이는 `6.68e-9`였다. 저장된 SHAP의 예측값과 실제
모델 원출력 차이는 `1.19e-7`이었다.
단, IG 완결성 오차는 양쪽 모두 약 `0.27009`로 남았다.
청킹 동등성 확인이 IG의 기준 입력·투영 방식까지 검증했다는 뜻은 아니다.
측정 원문과 버전은 [실측 JSON](docs/benchmarks/dashboard-integration-2026-11-14.json)에 있다.

기본 64샘플에서는 KernelSHAP 회귀가 특이해져 큰 기여도나 수치 오차가 생길 수 있다.
이는 [PR #176](https://github.com/shin584/project/pull/176)에 보고된 기존 문제이며
통합 테스트에서도 재현됐다. 진행률·캐시 통합은 이 계산 설정을 변경하지 않는다.
샘플 수를 올리면 기존 64/4 사전계산 파일은 적중하지 않으므로 해당 설정으로 다시 준비해야 한다.
