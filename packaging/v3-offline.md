# Dashboard v3 Windows 오프라인 배포 · #204

Windows x64 로컬 디스크에 `XaiDashboardV3` 폴더 전체를 복사한다. 폴더의
`verify-v3.ps1`로 체크섬을 검사하고 `XaiDashboardV3.exe`를 실행한다.
복구 시에도 검증된 원본 폴더 전체를 새 위치에 복사한다. 폴더 안의
`START-HERE.txt`에 전달용 절차가 있다. Windows의 .NET Framework 4.8을 사용한다.

## 로컬 자산으로 다시 빌드

- 소스를 `release-manifest.json`의 `source_commit`으로 체크아웃한다.
- `requirements-v3.lock`에서 고정한 Windows x64 CPython 3.13 wheelhouse를 준비한다.
  실제 wheel 파일명·크기·SHA-256은 release manifest에서 확인한다. wheel 확보는
  사전 준비이며 아래 설치와 빌드는 로컬 wheelhouse만 사용한다.
- #192 보존본의 두 NT 모델, tokenizer, Fixed Version WebView2 및 non-NT vault
  ZIP을 준비한다. [보존 및 복구 절차](../docs/preservation/README.md)
- [#204 디스크 예산](../docs/preservation/disk-addendum-204.md)을 확인하고
  새 출력 경로를 사용한다.

```powershell
$env:PYTHONPATH = (Resolve-Path packaging/offline_build).Path
$env:PIP_CONFIG_FILE = 'NUL'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
py -3.13 -m venv .packaging/v3-build-venv
$buildPython = (Resolve-Path .packaging/v3-build-venv/Scripts/python.exe).Path
& $buildPython -m pip install --no-index --only-binary=:all: --find-links $wheelhouse --require-hashes -r packaging/requirements-v3.lock
& $buildPython -m pip check
& $buildPython -X utf8 packaging/build_desktop.py .packaging/v3-build `
    --wheelhouse $wheelhouse --non-nt-archive $archive --assets $offlineAssets --models $ntModels
```

`$offlineAssets`는 `tokenizer`, `webview2`를 포함한 폴더이고 `$ntModels`는
두 `.safetensors` 파일을 포함한 폴더다. 기본값은 현재 worktree의
`.packaging/assets` 및 `models`이다. 빌더가 검증된 복사본을 새 경로에
스테이징하고 PyInstaller 작업 폴더와 캐시를 같은 출력 루트에 둔다.
최종 폴더는 `.packaging/v3-build/dist/XaiDashboardV3`이다.
Python socket/DNS 감사 훅은 설치와 PyInstaller 하위 Python에 적용한다.

non-NT는 Keras 3에서 PyTorch CPU를 backend로 사용한다. 같은 PyTorch가
예측·gradient와 NT 실행에 쓰인다. Cas12a는 원 `.keras`, SpCas9은 #98의
채택 export를 싣는다. 별도 gradient 런타임과 TensorFlow는 동봉 범위에서
제외했다. 모델 및 export 출처, 런타임 의존성, 파일 SHA가 manifest에 있다.

## 깨끗한 Windows에서 확인

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File packaging/new-v3-sandbox.ps1 -Bundle $release -Output $newEvidence
```

생성된 `dashboard-v3.wsb`를 실행한다. Sandbox는 네트워크를 끄고 배포 폴더와
빈 결과 폴더만 연결한다. Sandbox 안의 로컬 디스크로 복사하고 검증 결과와
OS·활성 어댑터·Python 명령 정보를 `results`에 남긴다. Sandbox를 쓸 수 없으면
동일 조건의 새 Windows VM에서 이 절차와 사용자 흐름을 수행한다. 게스트 준비
스크립트는 `manual_workload_pending`을 기록한다. 실제 흐름 결과를 별도 저장한다.

1. 전시용 기본 상태에서 non-NT가 선택되었는지 확인한다. 게스트 PowerShell에서
   `capture.json`의 `input_seq`를 읽어 게스트 클립보드에 복사할 수 있다.
   #203의 고정 입력을 붙여 넣고 SpCas9과 Cas12a의 후보 표를 확인한다. 네 보기에서 입력과 표가
   이어지는지 확인한다. [고정 입력과 설정](../docs/benchmarks/ram-procedure.md)
2. 해당 고정 두 사이트·헤드에서 non-NT IG/SHAP 그래프가 보이고 Rollout이
   비활성인지 확인한다. NT로 전환해 입력이 유지되는지와 세 XAI 그래프를 확인한다.
   다시 non-NT로 돌아와 자동 분석이 완료되는지 확인한다.
3. 연속 요청과 정밀 설명 중 취소 후 새 설명을 실행한다. 이전 계열이나 취소한
   요청의 늦은 결과가 최종 화면에 반영되는지 확인한다. 오류 재시도와 자식 종료는
   별도로 기록한다.
4. 두 계열에서 상세 보기의 검색 결과를 CSV로 저장한다. 마지막 두 열의 모델명과
   버전을 확인한다. NT를 선택한 상태에서 앱을 닫고 다시 열어 non-NT 기본 상태를
   확인한다.

게스트 결과에는 시각, manifest SHA, 각 흐름의 통과·실패·미실행과 근거를 남긴다.
수치 품질은 [#199/#200/#201의 기록](../docs/issue203-implementation.md)에서 추적한다.
