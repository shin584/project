# XAI dashboard

81bp DNA에서 SpCas9·Cas12a 사이트를 찾고 예측과 XAI를 보여주는 Streamlit 앱이다.
이 브랜치는 대시보드 파일이 저장소 루트에 있는 `dashboard_core`를 기준으로 한다.

## 실행과 테스트

```powershell
python -m pip install streamlit numpy pandas matplotlib logomaker shap torch transformers safetensors gdown pytest
python -m streamlit run app.py
python -m pytest tests -q
```

가중치는 `models/SpCas9.safetensors`, `models/Cas12a.safetensors`에 둔다.
가중치가 없으면 `model_source.py`에 등록된 Drive 주소로 다운로드를 시도한다.
토크나이저는 최초 실행 시 Hugging Face에서 받는다. 부스에서는 모델과 토크나이저를
미리 준비하고, 실제 사용할 서열로 각 기법을 한 번씩 실행해 둔다.
테스트는 가짜 모델을 사용하므로 가중치·PyTorch·토크나이저 다운로드 없이 실행된다.

## #150 통합 범위

[#150의 최신 정리](https://github.com/shin584/project/issues/150#issuecomment-6794974525)에
따라 #153의 진행률·IG 청킹과 #152의 선택 워밍업·SHAP 디스크 캐시를 통합했다.
#151의 Streamlit 결합 제거와 #154의 facade는 보류 상태다.
`load_predictors`의 `st.cache_resource`를 유지하므로 재실행마다 모델과 워밍업을 반복하지 않는다.
`app.py`는 변경하지 않았고 화면 변경은 #153의 `xai/view.py`에 한정한다.

- 워밍업은 `WARMUP_KEYS = (("A", "Cas12a"),)`만 대상으로 한다.
  SpCas9은 실제 요청까지 지연 로딩되며, `load_predictors(warmup=False)`로 워밍업을 끌 수 있다.
  워밍업 실패는 로그로 남기고 첫 실제 요청에서 다시 로딩한다.
- 디스크 캐시는 `(model_key, head, window_seq, nsamples, n_background)`를 모두 구분한다.
  `seed=0`일 때만 조회하며, 설정이 달라지면 실제 계산으로 넘어간다.
- 캐시 적중 시 추론과 `on_batch`를 호출하지 않는다. 미적중 시 실제 추론 배치마다
  진행률을 전달하고, 계산 완료·캐시 적중·예외 시 화면 래퍼가 진행률 영역을 비운다.
  콜백의 `done/total` 단위는 #153 구현의 **누적 서열 수**이고, 배치 번호가 아니다.
- IG는 최대 8개 적분 단계씩 역전파한다. IG 진행률 표시는 추가하지 않았다.

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
