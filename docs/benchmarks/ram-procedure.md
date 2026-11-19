# #203 전체 프로세스 트리 RAM 측정 절차

이 절차는 소스 실행과 배포 실행을 별도 실측한다. 모델 파일 크기는 실행 자산의
식별에만 사용한다. 절약 여부는 실제 동시 프로세스 메모리 합으로 판단하며 절대
상한이나 목표 감소율을 사후에 만들지 않는다.

## 실행 전 등록

기존 NT 소스는 `xai_dashboard_desktop` / `dashboard_core_v2`에 보존한다.
v3는 `dashboard_v3`에서 실행한다. 실행본을 순서대로 종료하고 다음 것을 실행한다.
같은 PC·Python·라이브러리·입력·설정을 사용한다. 개발 PC에 설치된 보조 런타임도
실제로 import되면 비용에서 제외하지 않는다.

```powershell
py -3.13 -X utf8 docs/benchmarks/measure_ram.py register docs/benchmarks/issue203-ram-20261119-run02 --legacy C:/project/MC_project/xai_dashboard_desktop
py -3.13 -X utf8 docs/benchmarks/measure_ram.py run docs/benchmarks/issue203-ram-20261119-run02 legacy
py -3.13 -X utf8 docs/benchmarks/measure_ram.py run docs/benchmarks/issue203-ram-20261119-run02 v3
py -3.13 -X utf8 docs/benchmarks/measure_ram.py summarize docs/benchmarks/issue203-ram-20261119-run02
```

`register`는 앱 실행 전에 `contract.json`을 기록한다. 소스 commit 및 Python 파일
해시, 실제 checkpoint·tokenizer 파일 해시, 측정 도구 해시, PC, Python·라이브러리
버전, 환경 변수, 고정 입력·헤드·설정, 단계 순서·시간을 기록한다. 이후 소스·자산·
도구가 달라지면 `run`은 거부한다. 출력 폴더는 새로 만들어 기존 결과를 보존한다.

고정 입력은 #181의 81bp 서열이다. 설명은 `SpCas9|+|32` / `SpCas9-NG`,
`Cas12a|+|16` / `Cas12a`를 사용한다. 두 Cas에 대해 IG 32단계, SHAP 64샘플·배경
4개, 그래프 80%, CPU thread 환경 4로 실행한다. NT는 Rollout도 실행한다.
non-NT는 Rollout 미지원이다. Cas12a의 계열별 모델 윈도우 배치는 각 제품의
실제 규칙을 따른다. 기존 프로그램의 단위·디스크 캐시 정책과 v3의 교정 단위·
프로세스 캐시는 보존하며 동일한 수치 프로토콜이라고 주장하지 않는다.

## 관찰 범위와 메모리 지표

외부 sampler는 앱 Python 프로세스, 계산 자식, WebView2 browser/renderer/GPU/
utility 및 발견한 모든 자손을 추적한다. PID와 생성 시각을 함께 보존하며 부모가
사라진 뒤에도 발견한 프로세스는 계속 관찰한다. 프로세스별 경로·명령행·부모 PID와
시간별 working set, private commit, 누적 page fault 및 시스템 가용 RAM을 남긴다.
측정 도구 자체의 프로세스는 앱 합계 밖이다.

- 주 메모리 지표는 **같은 시각의 working set 합**이다. 공유 페이지가 중복될 수
  있으므로 중복 없는 물리 RAM이라고 부르지 않는다.
- **PrivateUsage 합**은 private commit이다. paging된 메모리를 포함하므로 resident
  RAM과 구별한다. 메모리가 작은 PC에서는 working set만으로 회수를 판단하지 않는다.
- 100ms는 목표 간격이다. 실제 timestamp·질의 시간·최대 간격을 보존한다. 더 짧은
  피크나 sampler 사이에 생겼다 사라진 자식은 관측하지 못할 수 있다.
- 단계 피크는 **동시 합의 최대 관측값**이다. 개별 프로세스의 서로 다른 시점
  lifetime peak를 더하지 않는다. 안정 상태는 마지막 2초의 중앙값·최소·최대다.
- 실제 앱의 JSON 직렬화·pipe 복사·PNG/base64/HTML·브리지·DOM 이미지 decode와
  렌더링·예측/XAI 캐시·gradient 런타임 비용이 합계에 포함된다. observer는 호출을
  대체하지 않고 작은 실행 메타데이터만 기록한다.

`ram_window.py`는 제품의 `desktop.app.main`을 실행하고 기존 DOM 폼으로 동작을
요청한다. 자동으로 발생하는 준비/탐색/예측/설명 호출도 `phases.jsonl`에 기록한다.
예측 후 제품이 기본 설명을 자동 요청하므로 예측 호출의 끝에서 설명 단계로
이어진다. 기본 설명은 기존 NT의 Rollout, v3 non-NT의 IG다. 이 자동 동작까지
포함한 전체 흐름과 명시적으로 선택한 고정 사례별 기법 측정을 구별한다.
이미지 decode 완료 후 각 단계에서 5초 대기하여 렌더링·캐시 보유 비용을 포함한다.

계산 프로세스 observer는 실제 예측기의 클래스·경로와 `_model`, `_tokenizer`,
`_runtime`의 적재 여부 및 런타임 모듈·thread 수를 기록한다. 새 모델을 로드하거나
forward/gradient를 대체하지 않는다. v3 UI에서 모델 라이브러리가 import되지
않는지도 각 단계에서 확인한다. non-NT 기본 상태와 복귀 상태는 NT 모델·tokenizer가
없는 선택 예측기와 runtime snapshot으로 확인한다.
기법별 실제 계산 함수 진입 횟수와 디스크 SHAP 조회·적중 횟수도 기록하므로,
자동 설명 이후 같은 기법을 다시 고른 캐시 적중을 새 계산으로 보고하지 않는다.

## 실행 단계와 종료

기존 NT: interpreter/import → 준비 → 고정 입력 예측과 자동 설명 → 고정 두 Cas의
IG/SHAP/Rollout → 앱 종료. v3: 기본 non-NT 준비·예측·IG/SHAP → NT 전환 후
재분석·IG/SHAP/Rollout → non-NT 복귀 후 자동 재분석 → 안정 구간 10초 → 종료.

각 전환은 이전 계산 PID와 자손의 종료, 입력 유지, 새 계산 PID를 확인한다.
복귀 후 남은 NT 자식이 없어야 한다. 프로세스 종료는 그 프로세스의 Python 참조·
가중치·캐시 주소 공간을 회수하므로 단순한 GC 호출과 구별한다. OS 파일 캐시는
앱의 잔여 모델 적재라고 부르지 않는다. 최종 앱 종료 후에도 3초 동안 발견한
프로세스가 남는지 확인한다. 접근 거부·timeout·잔여 자식은 실패로 기록한다.

`samples.jsonl.gz`는 원시 측정 자료, `sampler.json`은 프로세스 목록·간격·종료 상태,
`phases.jsonl`은 호출 경계, `runtime-*.jsonl`은 적재 상태, `window.json`은 창 검증,
`window.log`는 실제 실행 로그다. `summary.json`은 원시 자료에서 다시 계산한다.
단일 순차 실행의 결과이므로 반복 실험의 신뢰 구간이나 모든 입력의 보장이 아니다.

기존 프로그램의 SHAP이 디스크 캐시에 적중했으면 별도 조건을 먼저 등록하여 새
계산의 비용도 측정한다. `--bypass-legacy-disk-shap`은 파일을 바꾸지 않고 lookup을
미적중으로 돌려 기존 SHAP 계산을 실행한다. 기존 기본 실행과 이 추가 실험은
별도 폴더로 보존한다. v3는 원래 계산 프로세스에서 디스크 캐시를 사용하지 않는다.

```powershell
py -3.13 -X utf8 docs/benchmarks/measure_ram.py register docs/benchmarks/issue203-ram-20261119-cold01 --legacy C:/project/MC_project/xai_dashboard_desktop --bypass-legacy-disk-shap
py -3.13 -X utf8 docs/benchmarks/measure_ram.py run docs/benchmarks/issue203-ram-20261119-cold01 legacy
py -3.13 -X utf8 docs/benchmarks/measure_ram.py summarize docs/benchmarks/issue203-ram-20261119-cold01
py -3.13 -X utf8 docs/benchmarks/analyze_ram.py docs/benchmarks/issue203-ram-20261119-run02 --cold docs/benchmarks/issue203-ram-20261119-cold01
```

기본 실행 원본 코드는 `run02/observer/`에 보존한다. 추가 실험용 flag와 단일 모드
요약 기능을 넣은 뒤 새 계약을 등록했으며, 앞선 실행의 코드 해시를 바꾸지 않는다.
분석기는 원시 sample과 runtime·창·sampler 기록으로 프로세스 수명과 미적재 조건을
검사하고 `comparison.json`에 측정값과 검사 수를 기록한다. 감소율에는 합격 기준을
추가하지 않는다. 전환 호출 시작에는 아직 이전 계열이 살아 있으므로, 전환 피크를
새 계열의 정상 상태 피크에 섞지 않고 전환 전 구간과 별도 안정 구간을 비교한다.

## 최종 배포본에서 재사용

최종 패키지는 #204에서 별도 측정한다. 위 소스 결과를 배포 검증으로 사용하지 않는다.
기존/신규 배포 폴더 전체를 보존하고 executable·배포 manifest·모델·WebView2·runtime
버전 및 해시, PC, 입력·설정·단계 순서·측정 지표를 **먼저** 새 기록에 등록한다.

외부 sampler는 frozen executable에도 같은 방식으로 사용할 수 있다. 실행 파일을
수정하거나 Python 모델 런타임을 sampler에 적재할 필요가 없다.

```powershell
py -3.13 -X utf8 docs/benchmarks/measure_ram.py watch C:/new-evidence/frozen-v3 -- C:/new-release/XaiDashboard/XaiDashboard.exe
```

실제 배포 창에서 위와 같은 순서로 입력·기법·계열을 조작하고 각 단계 시작/종료의
UTC Unix timestamp를 별도 원시 기록에 남긴다. 이 명령의 `launch.json`만으로 실행 전
명세를 대신하지 않는다. UI 모듈 import/모델 적재 내부 확인은 패키지의 진단 경로와
빌드 소스의 기존 import 경계 검사를 함께 사용하고, 확인하지 못한 항목은 미검증으로
적는다. 종료 후 `sampler.json`의 잔여 자식·오류·실제 간격을 확인하고 동일 시각 합으로
단계별 피크·복귀 안정 상태를 산출한다. 깨끗한 오프라인 Windows 검증은 별도 항목이다.
