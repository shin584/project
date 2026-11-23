# Dashboard v3 최종 EXE 배포본 (dashboard-v4.0)

실행 파일 묶음은 git에 들어 있지 않고 GitHub Release `dashboard-v4.0`에 zip 3개로 첨부돼 있다.
이 폴더에는 그 묶음을 식별·검증하는 파일만 있다.

| 항목 | 값 |
|---|---|
| 빌드 | #221 rebuild 2 (`.packaging/issue221-build02/dist/XaiDashboardV3`) |
| 소스 커밋 | `63afa1168bfdf282136a82ee181268e80a0a046c` |
| 빌드 시각 (UTC) | 2026-11-23T04:15:13Z |
| `XaiDashboardV3.exe` SHA-256 | `f3b7f1028140f5762e09a939a3aa8d8e8524aaa0f69c2de836532f8a224e4e0e` |
| `release-manifest.json` SHA-256 | `3e0e2d3343849d76d30a014d002ab9d7cfc5bf6edfc910dd3daa71f0d9237353` |
| 파일 수 / 총 바이트 (manifest 제외) | 6,851 / 5,414,867,417 |

## Release 첨부 파일

| 파일 | 내용 | 바이트 | SHA-256 |
|---|---|---|---|
| `XaiDashboardV3-part1-app.zip` | 모델 가중치 2개를 뺀 전부 | 623,951,914 | `d41fed162f8bf88b4f8ea3987732e6690457df9989a735475a2431569bf9b67c` |
| `XaiDashboardV3-part2-SpCas9.zip` | `_internal/models/SpCas9.safetensors` | 1,921,845,498 | `c366af9027babbaffc60fa0c199474b9242bf8ff951c7a12d92c865810b0fb5e` |
| `XaiDashboardV3-part3-Cas12a.zip` | `_internal/models/Cas12a.safetensors` | 1,921,804,506 | `15f5b069a0176feeedd380d2937294e6f2cde8cf9f1614770ced9b310936eb0e` |

세 zip 모두 최상위에 `XaiDashboardV3/` 폴더를 담고 있어, 같은 위치에 풀면 하나의 폴더로 합쳐진다.

## 받고 실행하는 법

1. Release `dashboard-v4.0`에서 zip 3개를 받는다. 로컬 드라이브에 여유 공간이 약 10GB 필요하다(zip 4.5GB + 압축 해제 5.4GB).
2. (선택) zip 해시를 위 표와 비교한다.
   `Get-FileHash .\XaiDashboardV3-part*.zip -Algorithm SHA256`
3. **압축을 풀기 전에 zip 3개의 차단을 해제한다.** 각 zip을 마우스 오른쪽 버튼 → 속성 → 아래쪽 "차단 해제" 체크 → 확인.
   PowerShell로 한 번에: `Get-ChildItem .\XaiDashboardV3-part*.zip | Unblock-File`
4. 세 zip을 **같은 폴더에** 모두 푼다(Windows 기본 "압축 풀기" 가능). 결과는 `XaiDashboardV3\` 하나.
5. 전체 파일을 검증한다.
   `powershell -NoProfile -ExecutionPolicy Bypass -File .\XaiDashboardV3\verify-v3.ps1`
   출력의 `manifest_sha256`이 위 표의 값과 같아야 한다.
6. `XaiDashboardV3\XaiDashboardV3.exe`를 실행한다. 자세한 안내는 `START-HERE.txt`.

### 실행 시 `Failed to resolve Python.Runtime.Loader.Initialize` 에러가 날 때

3단계(차단 해제)를 건너뛰고 압축을 푼 경우다. 인터넷에서 받은 zip을 Windows 탐색기로 풀면
"인터넷에서 받은 파일" 표시(`Zone.Identifier`)가 풀린 파일 전부에 복사되고, 창을 띄우는
.NET Framework가 이 표시가 붙은 `Python.Runtime.dll`을 불러오지 않는다.
압축을 다시 풀 필요 없이, 풀린 폴더에서 표시를 지우고 다시 실행하면 된다.

`Get-ChildItem .\XaiDashboardV3 -Recurse -File | Unblock-File`

표시만 지우므로 파일 내용과 `verify-v3.ps1` 결과는 바뀌지 않는다.

## 이 폴더의 파일

- `build-identity.json` — 빌드 식별 정보 (`docs/benchmarks/issue221-recheck/rebuild02/`의 사본)
- `release-manifest.json`, `START-HERE.txt`, `verify-v3.ps1` — 번들에 들어 있는 파일의 사본
