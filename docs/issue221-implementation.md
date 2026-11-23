# #221 · #215 EXE 재빌드와 재현 시나리오 재검사

#215의 수정(#217~#220)을 담은 전시용 EXE를 다시 빌드하고, 보류됐던
`.packaging/exhibition-audit-20261121/`의 재현 시나리오를 소스와 EXE에서 같은
검사 스크립트로 반복했다. 두 실행 모두 **13 통과 / 0 실패 / 0 건너뜀**이다.
결과는 [`benchmarks/issue221-recheck/`](benchmarks/issue221-recheck/)에 있다.

## 빌드 식별

| 항목 | 값 |
| --- | --- |
| 빌드 소스 커밋 | `06b838124e4aca7b68b2730cbd269d90ed65e21e` (#220 `1e41a43` + 검사 스크립트만 추가, 제품 코드 동일) |
| release manifest SHA-256 | `ba4c85cd2ad1837448c17658b7916ef979668c6e9c57f77992d1e7da5425d8d2` |
| `XaiDashboardV3.exe` SHA-256 | `53db57775ae5350431751295ba0366e095934bd03b96977fe4d69608bb545cf5` |
| 파일 / 바이트 | 6,851개 / 5,414,867,378 (`verify-v3.ps1` 검증 통과) |
| 빌드 시각 · 도구 | 2026-11-22T11:52:15Z · Python 3.13.7 · PyInstaller 6.22.3 |
| 배포 폴더 | `.packaging/issue221-build01/dist/XaiDashboardV3` (git 미추적) |

[`packaging/v3-offline.md`](../packaging/v3-offline.md)의 절차와 #204의 로컬
wheelhouse·venv·non-NT vault ZIP(`4c0206…bfcd21c`)을 그대로 썼다. release manifest는
빌더가 고정한 `issue: 204` 스키마를 유지한다(`verify-v3.ps1`이 이 값을 검사한다).
요약은 [`build-identity.json`](benchmarks/issue221-recheck/build-identity.json).

```powershell
$env:PYTHONPATH = (Resolve-Path packaging/offline_build).Path
$env:PIP_CONFIG_FILE = 'NUL'
$buildPython = (Resolve-Path .packaging/issue204-inputs/venv/Scripts/python.exe).Path
& $buildPython -X utf8 packaging/build_desktop.py .packaging/issue221-build01 `
    --wheelhouse .packaging/issue204-inputs/wheels `
    --non-nt-archive C:\project-backups\dashboard-v3\issue192-20261118\non-nt-models-4c0206782d9f7c1aacfeaa763376bf3624cf31f86338b659e0973fdc5bfcd21c.zip
```

## 검사 방법

[`benchmarks/recheck_exhibition.py`](benchmarks/recheck_exhibition.py)의 `run`이 실제
모델·오프라인(새 WebView2 프로필, 죽은 프록시, DNS 재정의, CDP offline, HF offline)으로
창 하나를 띄우고 WebView2 화면, 실제 저장 대화상자(pywinauto), 계산 프로세스의
OS 상태(일시 정지·강제 종료)만으로 조작한다. 소스는 `.packaging/issue204-inputs/venv`의
Python으로 `desktop_entry.py`를, EXE는 빈 HF 캐시와 `PYTHONPATH` 없이 실행했다.
non-NT 기본 계열과 `capture.json`의 고정 입력(후보 38행)을 쓴다.

```powershell
python -u -X utf8 docs/benchmarks/recheck_exhibition.py run .packaging/issue221-recheck/source source
python -u -X utf8 docs/benchmarks/recheck_exhibition.py run .packaging/issue221-recheck/exe exe `
    --exe .packaging/issue221-build01/dist/XaiDashboardV3/XaiDashboardV3.exe
python -X utf8 docs/benchmarks/recheck_exhibition.py history docs/benchmarks/issue221-recheck/history-source.json
& $buildPython -X utf8 docs/benchmarks/recheck_exhibition.py history docs/benchmarks/issue221-recheck/history-exe.json `
    --exe .packaging/issue221-build01/dist/XaiDashboardV3/XaiDashboardV3.exe
```

소스 실행은 커밋 `6bf4c2d`(추적 파일 변경 없음)에서 했다.

## 결과

| 시나리오 (#215 재현 항목) | 소스 | EXE |
| --- | --- | --- |
| 적재 한도 초과: 첫 모델 적재 중 계산 프로세스 정지 → 오류 "모델 적재가 적재 한도(165초)를…"와 "모델 준비 다시 시도", 분석 잠금, 재시도로 같은 계열 복구 | 165.5초, 통과 | 165.5초, 통과 |
| CSV 대화상자 닫음: 오류 없이 끝나고 파일 없음 | 통과 | 통과 |
| 대기 중 종료 후 CSV 저장: 대화상자 없이 "계산 프로세스가 다시 시작되어 저장하지 못했습니다…" 오류, "CSV를 저장하는 중..." 미잔존, 파일 없음 → 재분석 후 저장 38행 성공 | 통과 | 통과 |
| 취소 없는 멈춤: 분석 중 정지 → 무진행 한도 뒤 "계산이 응답하지 않아…" 오류, "계산 프로세스를 다시 시작하는 중…"·분석 잠금, 같은 계열 새 PID, 자동 재실행 0건 | 105.1초, 통과 | 105.2초, 통과 |
| 멈춘 계산 취소 후 새 분석: 정지된 설명 계산 취소("계산을 취소했습니다.") → 분석 → 취소 유예 뒤 `cancel_grace` 강제 재시작, 분석 잠금, 계열 전환 없이 새 분석 38행 완료 | 53.1초, 통과 | 53.2초, 통과 |
| 요청 10만 건: 브리지로 10만 건 2회, 모두 요청 ID 응답 후 분석 정상 | 통과 | 통과 |
| 강제 재시작(적재 한도·무진행·취소 유예) 후 창 닫기(WM_CLOSE): 종료 코드 0, 남은 자식 0개 | 0 / 0개 | 0 / 0개 |

두 실행 모두 JS 오류 0건, 외부 샘플러의 접근 오류·시간 초과 0건이다.

### 요청 기록 상한

창 밖에서는 `Dispatcher._outcomes`를 읽을 수 없어 `history`로 따로 확인했다.
EXE 쪽은 EXE의 PYZ에 동결된 `desktop.api` 바이트코드를 꺼내 실행했다. 두 쪽 모두
`OUTCOME_LIMIT = 256`이고 1천·1만·10만 건 뒤 기록 **256건**, tracemalloc 추적
메모리 26,224 → 26,376바이트였다.
[소스](benchmarks/issue221-recheck/history-source.json) ·
[EXE](benchmarks/issue221-recheck/history-exe.json)

창 실행의 UI 프로세스 private 메모리(MiB)는 참고값으로 남긴다.

| 실행 | 1회차 10만 건 | 2회차 10만 건 |
| --- | ---: | ---: |
| 소스 | 78.0 → 125.9 (+48.0) | 125.9 → 146.9 (+21.0) |
| EXE | 80.1 → 140.4 (+60.3) | 140.4 → 156.2 (+15.8) |

요청 기록은 위와 같이 상한에 묶여 있지만, 2회차에도 UI 프로세스 메모리가 늘었다
(예비 실행에서는 +6.8, +7.5). 이 증가는 요청 기록이 아니라 pywebview 브리지·
WebView2·할당기 쪽으로 보이며 원인은 확인하지 않았다. 10만 건씩 반복할 때
수렴하는지는 9시간 연속 운전 티켓에서 확인할 항목으로 넘긴다.

## 관찰과 판단

- **분석 중 취소 버튼의 시점.** 분석의 취소 버튼(작업 표시줄)은 PAM 탐색 응답 뒤
  예측 단계에서 나타난다. PAM 탐색 중에 계산 프로세스가 멈추면 누를 취소가 없고,
  무진행 한도가 복구한다(위 "취소 없는 멈춤"). 그래서 "취소 후 새 분석"은 즉시
  취소 버튼이 보이는 설명 계산을 멈춰 재현했다.
- **"취소했습니다" 문구.** 새 분석을 누르면 화면이 "PAM 탐색 중..."으로 바꾸므로,
  재시작 시점의 안내는 그 문구였다. 취소 직후 문구와 재시작 중 모델 상태
  "계산 프로세스를 다시 시작하는 중..."은 확인했다.
- **대화상자 닫음 뒤 안내 문구.** 위 실행의 빌드에서는 대화상자를 닫으면 작업은
  끝나지만 안내에 "CSV를 저장하는 중..."이 남았다(#215 이전 `a73d11b`과 같은 동작).
  이후 화면이 대화상자를 닫으면 이 안내를 지우도록 고쳤고
  (`tests/test_desktop_shell.py`의 `dialog_closed_ui == {"notice": "", ...}`),
  위 EXE는 이 수정 전 빌드다. 수정을 담은 EXE 재빌드·재실행은 하지 않았다.
- **예비 실행.** 첫 정식 실행(`.packaging/issue221-recheck-attempt01`)에서 EXE는
  12/12 통과했으나 소스는 창 연결이 늦어 non-NT 적재가 정지 전에 끝나 적재 한도
  시나리오가 성립하지 않았다. 검사 스크립트가 계산 프로세스를 실행 직후 멈추도록
  고치고(`6bf4c2d`, 화면이 아직 "모델 준비 중"인지 확인 추가) 두 실행을 모두 다시 했다.

## 범위 밖과 남은 미검증

- 9시간 연속 운전, 전시장 PC 오프라인 실행 → 후속 티켓.
- NT 계열에서의 재현 시나리오는 이번 검사에 넣지 않았다(개발 PC 가용 메모리 약
  2.7 GiB). NT의 정상 협력 취소·한도 측정은 #218/#220의 실제 모델 probe 기록을 따른다.
- 새 빌드는 개발 worktree 안에만 있다. #204처럼 worktree 밖 실행용 복사본은 만들지 않았다.
