# #204 · Windows 오프라인 배포 인계

복사·검증한 폴더는 `C:/project-backups/dashboard-v3/issue204/XaiDashboardV3`다.
`XaiDashboardV3.exe`를 실행한다. `START-HERE.txt`의 복구 절차와
[`packaging/v3-offline.md`](../packaging/v3-offline.md)의 빌드·검증 절차를 함께 사용한다.
빌드 소스는 `995fff0b592c662ca514e21f0c9414e0a8808e54`이고, release manifest SHA-256은
`1335e70286c5dc2d2c64fb84d5718efff1dcd6a2963283fc774f3e4a3a910d57`이다.

## 자산과 디스크

- 로컬 vault에서 non-NT checkpoint와 출처를 복원하고, #192 보존 목록의
  NT 모델·토크나이저·Fixed Version WebView2 바이트를 검증해 스테이징했다.
  Keras 3.14.1은 PyTorch 2.12 CPU backend를 사용한다. PyTorch가 NT와 gradient
  연산도 담당한다. 별도 보조 런타임 없이 빌드했으며 TensorFlow는 제외했다.
- PyInstaller 6.22.3을 전용 Python 3.13.7 환경에서 사용했다. wheel 확보 후
  `--no-index --only-binary=:all: --require-hashes`로 설치했고 `pip check`는 통과했다.
  Python DNS/소켓 감사 훅을 설치·스테이징·빌드 환경에 적용했다. DNS 차단 확인,
  wheel 설치 보고서, 스테이징 목록, 빌드 로그는 `.packaging/issue204-inputs`와
  `.packaging/issue204-build01`에 있다.
- `robocopy /E /COPY:DAT /R:1 /W:1 /MT:8`로 개발 worktree 밖에 복사했다.
  6,852개를 복사했고 오류는 0건이다. `packaging/verify-v3.ps1 -Bundle <복사본>`로
  manifest에 등록된 **6,851개 파일**의 SHA와 크기를 확인했다.

| 배포 폴더 | 파일 바이트 합, manifest 포함 |
| --- | ---: |
| 보존 NT `XaiDashboard`, `54b77ee` | 5,475,822,581 |
| v3 `XaiDashboardV3`, `995fff0` | 5,416,094,488 |
| 증감 | **−59,728,093 (−1.09%)** |

새 산출물의 파일 바이트 합은 wheelhouse 315,294,671, venv 1,363,190,909,
스테이징 4,569,189,513, PyInstaller 작업물 261,243,926, 배포 폴더 두 벌
10,832,188,976으로 **17,341,107,995**다. 운영체제·로그·캐시를 포함한 C: 여유
공간은 입력 획득 후 36,085,133,312에서 복사 후 확인 시점 18,217,037,824바이트로
변했다. 검증 중 확인한 최저 여유 공간은 17,585,270,784바이트였다.
[빌드 전 디스크 예산](preservation/disk-addendum-204.md)

## 개발 PC 실제 실행

복사본을 빈 작업 디렉터리에서 실행했다. WebView2는 새 프로필, 도달할 수 없는
로컬 프록시, DNS 재정의 및 CDP offline 모드를 사용했다. Python 모델 경로는 빈
Hugging Face 캐시와 offline 환경 변수를 사용했다. 호스트 네트워크 어댑터는
활성 상태였다. 계측에서 사용한 브라우저 실행 경로는 복사본의
`_internal/webview2/msedgewebview2.exe`로 확인했다.

- v3 실제 창 검사 **46 통과 / 0 실패 / 0 건너뜀**. 첫 진입의 non-NT 기본값,
  두 계열·두 Cas의 예측, non-NT IG/SHAP과 NT IG/SHAP/Rollout, 네 보기,
  입력 유지·재분석, 취소와 후속 요청, 이전 이벤트 폐기, 자식 종료 뒤 오류 재시도,
  그래프 크기 조절과 CSV 모델명·버전을 확인했다.
- NT 상태에서 닫은 뒤 두 번째 실행은 **2 통과 / 0 실패 / 0 건너뜀**.
  non-NT 기본 시작과 Rollout 비활성을 다시 확인했다. 두 실행 모두 종료 코드 0,
  접근 오류·시간 초과·남은 자식 0건이었다.
- 소스 실행은 **25 통과 / 0 실패 / 0 건너뜀**이다. 소스와 배포에서 계열별
  38개 예측 행의 순서·헤드·표시값이 일치했다. 원출력과 제한 점수는
  `rtol=1e-5, atol=1e-8` 안에서 일치했다. non-NT 4건과 NT 6건의 그래프 설명
  문구·헤드·표시 단위가 일치했고, NT CSV의 기존 열은 보존 표와 같았다.
  토크나이저는 소스 HF 스냅샷 및 배포 export의 각각의 SHA를 보존했다.
  두 vocab의 4,107개 토큰 순서가 일치했다.
  [비교 결과](benchmarks/issue204-win-release01/parity.json) ·
  [소스 실행본과 tokenizer](benchmarks/issue204-win-release01/source/launch.json)
- 고정 입력과 사이트·헤드, IG 32, SHAP 64/4, 그래프 80%, 4스레드는
  [실행 전 계약](benchmarks/issue204-win-release01/contract.json)에 있다.
  [실제 창 결과](benchmarks/issue204-win-release01/v3/window.json) ·
  [non-NT CSV](benchmarks/issue204-win-release01/v3/non_nt.csv) ·
  [NT CSV](benchmarks/issue204-win-release01/v3/nt.csv)

계측 도구를 실행한 명령은 아래와 같다. 배포 창의 Python과 분리된 외부 관찰 환경에서
Playwright와 pywinauto를 사용했다. `launch.json`에 각 실행의 전체 경로와 인자가 있다.

```powershell
$legacy = 'C:/project-backups/dashboard-v3/issue192-20261118/workfolder/dist/XaiDashboard'
$handoff = 'C:/project-backups/dashboard-v3/issue204/XaiDashboardV3'
$observer = '.packaging/issue204-inputs/ui-automation'
$evidence = '.packaging/issue204-evidence01'
python -X utf8 docs/benchmarks/measure_package.py register $evidence --legacy $legacy --v3 $handoff
foreach ($mode in @('v3', 'v3-relaunch', 'legacy', 'source')) {
    python -u -X utf8 docs/benchmarks/measure_package.py run $evidence $mode --automation-site $observer
}
python -X utf8 docs/benchmarks/measure_package.py compare $evidence
$repeat = '.packaging/issue204-evidence02'
python -X utf8 docs/benchmarks/measure_package.py register $repeat --legacy $legacy --legacy-only
python -u -X utf8 docs/benchmarks/measure_package.py run $repeat legacy --automation-site $observer
```

## 실제 배포본의 전체 트리 RAM

#203의 100ms 목표 간격과 단계 구분으로 UI·계산·WebView2 전체 트리를 관찰했다.
표는 **동시 working set 피크 / private commit 피크**이며 단위는 MiB다.
Working set 합에는 공유 페이지가 중복될 수 있고, private commit에는 paging된
메모리도 들어간다. 비교 구간은 v3의 취소·오류 유도 등 추가 수용 검사 직전까지다.
같은 기법의 자동 호출과 명시 호출 중 최대치를 기록했다. 기존 배포본은
동봉된 SHAP 디스크 캐시 정책을 사용한다. [#203 계측 절차](benchmarks/ram-procedure.md)

| 단계 | 보존 NT 배포 | v3 기본 non-NT | v3 NT 전환 후 |
| --- | ---: | ---: | ---: |
| 시작·준비 | 2,203.7 / 2,603.1 | 488.7 / 354.8 | 1,749.7 / 2,638.4 |
| 예측 | 2,933.5 / 4,437.2 | 852.6 / 818.7 | 2,707.9 / 4,458.4 |
| SpCas9 IG | 2,462.6 / 4,695.3 | 942.6 / 982.5 | 2,143.2 / 4,724.2 |
| SpCas9 SHAP | 1,957.4 / 4,624.5 | 949.3 / 1,007.9 | 2,664.3 / 4,653.5 |
| SpCas9 Rollout | 2,767.6 / 4,607.6 | 미지원 | 2,046.7 / 4,663.4 |
| Cas12a IG | 2,335.7 / 4,866.2 | 946.5 / 1,008.9 | 2,546.2 / 4,888.5 |
| Cas12a SHAP | 1,977.6 / 4,655.5 | 967.2 / 1,035.1 | 1,986.4 / 4,728.4 |
| Cas12a Rollout | 2,217.1 / 4,656.5 | 미지원 | 2,010.2 / 4,732.3 |
| 흐름 전체 피크 | **2,933.5 / 4,866.2** | **967.2 / 1,035.1** | **2,707.9 / 4,888.5** |

- 기본 non-NT의 전체 트리 피크는 보존 NT 대비 working set **67.03%**,
  private commit **78.73%** 낮았다. 이 고정 사례에서 관찰한 값이다.
- NT에서 복귀하기 직전 마지막 2초의 중앙값은 **2,006.6 / 4,731.7**이었다.
  non-NT로 복귀한 뒤 10초 안정 구간의 마지막 2초 중앙값은 **614.5 / 953.7**이었다.
  이전 NT 계산 PID의 종료를 확인했다. 추가 수용 검사까지 포함한 v3 전체 피크는
  **2,910.8 / 4,888.5**였다.
- #203의 v3 소스 기본 흐름은 **1,123.2 / 1,193.1**이었다. 이번 배포 기본 흐름은
  그보다 **156.0 / 158.0 MiB** 낮다. 소스에는 실제로 import된 TensorFlow가
  포함됐고 두 실행의 시스템 메모리 상태도 다르므로 차이는 관찰값으로 남긴다.
  [#203 소스 실측](issue203-implementation.md)
- 보존 NT 첫 실행은 창 검사 8건을 통과했지만 초기 계측에서 단계 경계를 누락했다.
  브리지 준비를 기다리는 `b5195b0`으로 다시 등록·실행해 창·계측 검사
  **9 통과 / 0 실패 / 0 건너뜀**을 확인했다. 표는 그 재측정의 2,077개와
  v3의 4,452개 샘플을 사용한다. 두 실행의 최대 간격은 0.352초와 0.405초였고
  접근 오류·시간 초과·종료 후 남은 자식은 모두 0개였다.
  [바이트 단위 비교](benchmarks/issue204-win-release01/ram-comparison.json) ·
  [보존 NT 재측정](benchmarks/issue204-win-legacy-repeat/summary.json)

## 깨끗한 Windows 오프라인 게이트

**미실행.** 배포 폴더와 결과 경로만 연결하는
`C:/project-backups/dashboard-v3/issue204/sandbox/dashboard-v3.wsb`를 생성했다.
XML과 게스트 스크립트가 파싱되고 `Networking=Disable`임을 확인했다.
개발 PC에서 `.wsb` 실행 시 필요한 Windows Sandbox 실행 항목을 찾을 수 없었다.
`Get-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM`은
관리자 권한을 요구했고 `Get-VM`은 접근이 거부됐다. 깨끗한 Windows 게스트의
리소스 자립성과 사용자 흐름은 미완료로 남긴다.

## 모델 성능과 설명 품질

- 채택 non-NT checkpoint의 [#202 test 성능](issue202-implementation.md)을 인계한다.
  SpCas9은 5,186행·9헤드, Cas12a는 별도 1,500행 test를 평가했다. Cas12a의
  Pearson은 0.7612, MAE는 약 15.45%p다. test 입력의 원본→배포 최대 절대 출력
  차이는 각각 `2.38e-7`, `1.49e-7`이었다.
- XAI 수치 품질에는 미달이 남아 있다. non-NT SpCas9은 671 통과·269 실패,
  non-NT Cas12a는 85 통과·32 실패, NT는 371 통과·47 실패를 기록했다.
  각 수치 기준, 실패 사례, 단위 계약은 이전 실측에 보존되어 있다.
  [#199 SpCas9](issue199-implementation.md) · [#200 Cas12a](issue200-implementation.md) ·
  [#201 NT](issue201-implementation.md)
- 이번 배포 검증은 고정 사례의 예측·헤드·단위 및 실제 그래프 표시를 확인한다.
  설명 품질의 수치 판정은 위 검증 결과를 사용한다.

## 선택 회귀와 정적 검사

동적 predictor 선언·로컬 리소스 경계·복구·자식 수명·CSV 보기와 단위에 직접
연결된 기존 테스트를 실행했다.

```powershell
python -X utf8 -m pytest test/test_predict.py::test_name_resolved_predictors_are_declared_to_the_packaging_spec tests/test_integrated_layout.py test/test_offline_recovery.py tests/test_computation_process.py tests/test_family_switch.py tests/test_prediction_views.py tests/test_nt_xai_view.py -q -p no:cacheprovider
pyright packaging/build_desktop.py packaging/offline_build/sitecustomize.py docs/benchmarks/measure_package.py
python -X utf8 -m ruff check packaging/build_desktop.py packaging/desktop.spec packaging/offline_build/sitecustomize.py docs/benchmarks/measure_package.py
```

최종 회귀 **47 통과 / 0 실패 / 0 건너뜀**, 59.24초. Pyright 오류·경고 0,
Ruff 통과. 초기 동적 import 검사의 따옴표 의존성을 AST 검사로 수정했다.
빌드와 겹친 회귀에서 기존 자식 종료 검사 1건이 실패했고, 해당 검사의 단독
재실행과 최종 회귀에서 통과했다.

## 독립 검토

코드 기준 범위는 `git diff 01d6a01...ae587ff`다. 실제 #204 본문으로 명세를
검토했다. Standards에서 문서 규칙 위반은 0건이었다. 관찰 타이밍과 비교 실패
기록, 오래된 중복 선언, 보존 브랜치 빌드 문맥 관련 지적을 수정했다.
Spec의 취소한 요청 식별·이벤트 검사와 소스·토크나이저 출처 기록 지적도 반영했다.
수정 커밋은 `d95be0b`다.
