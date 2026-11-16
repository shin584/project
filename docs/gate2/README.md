# #188 게이트 2 실행 절차

기준은 `docs/equivalence-baseline/`의 #181 산출물이다. 고정 입력과 IG 32단계,
SHAP 64샘플·배경 4개를 유지한다. 이 검증은 기존 기준을 다시 생성하지 않는다.

현재 판정과 실행 증거는 [게이트 2 결과](result.md), 화면 차이는
[항목별 대조](CHECKLIST.md)에 기록한다.

## 최종 데스크톱 빌드

Windows x64, Python 3.13에서 `packaging/requirements.txt`를 설치하고
`packaging/README.md`의 tokenizer·Fixed WebView2 staging을 준비한다.

```powershell
python -X utf8 packaging/build_desktop.py
```

최종 산출물은 **`dist/XaiDashboard/` 전체 폴더**다. `XaiDashboard.exe`는
실제 `desktop.app`을 실행한다. #179의 `XaiGate1.exe`는 시험 창이므로 최종
시연용 exe로 전달하지 않는다. 모델 두 개, tokenizer, Fixed WebView2,
사전계산 SHAP, 비교 기준이 함께 들어간다. Streamlit은 최종 exe에서 제외한다.
빌드용 의존성에 Streamlit이 남아 있는 것은 게이트 1 재현 환경을 유지하기 위해서다.

`release-manifest.json`에는 빌드 호스트·버전·소스 SHA-256과 배포 파일 전체의
SHA-256을 기록한다. `source_commit`은 빌드 시점 HEAD이며, 미커밋 구현도
`source_sha256`으로 식별한다. exe·가중치·WebView2는 Git에 커밋하지 않는다.

## 실제 창에서 수치·동작 확인

실행 중인 데스크톱 창을 닫은 후, 다른 PC로 폴더 전체를 복사한다.
새 결과 폴더를 지정하며, 기존 결과 폴더에 덮어쓰지 않는다.

```powershell
powershell -NoProfile -File packaging/validate-desktop.ps1 `
  -Bundle D:/XaiDashboard -Output D:/gate2-check
```

검증 스크립트도 함께 전달한다. 다른 PC에는 Python이 필요하지 않다.
.NET Framework 4.8이 설치된 Windows 10/11 x64가 필요하며 WebView2는 동봉한다.
스크립트는 배포 파일 전체 해시를 확인하고 빌드 PC와 실행 PC 이름을 기록한다.
동일한 이름을 쓰는 PC는 별도로 물리적으로 다른 장비인지 확인해야 한다.

직접 실행할 수도 있다:

```powershell
XaiDashboard.exe --validate D:/gate2-artifacts
# 개발 소스 검증 (개발 PC HF 캐시 사용, 배포 판정 근거로 대체하지 않는다):
python -X utf8 -m desktop --validate .packaging/gate2-source
```

검증 경로는 실제 창의 입력·네 보기·설정·사이트·기법 선택과 CSV 저장을 조작한다.
OS 저장 대화상자만 지정된 경로로 응답한다. 실제 Renderer·모델·계산·차트는 그대로
실행한다. 공개 함수의 반환값을 관찰하며 대체하지 않는다.

- `candidates.json`: 실제 DOM의 표 열·행·순서·점수와 계산 결과의 반올림 전 값.
- `crispr_prediction_results.csv`: 실제 CSV 저장 경로의 파일. 기준과 바이트 diff.
- `xai-*.npz`: 화면 차트에 사용한 두 모델 × 세 기법의 실제 배열.
  `np.allclose(rtol=1e-5, atol=1e-8)`, SHAP은 추가로 완전 일치 검사.
- `01-*.html`~`06-*.html`: 선택된 탭과 live 입력 상태를 보존한 DOM.
  두 번째 모델의 XAI 화면에는 `-Cas12a` 접미사를 붙인다.
- `validation.json`: 계산 설정 범위/기본값, 크기 변경, 보기 전환 시 요청·계산
  미발생, 같은 기법 복귀 시 계산 캐시 적중, 서버/네트워크 관측과 오류.

실행 파일은 새 빈 HF 캐시와 오프라인 로더를 사용한다. Python 소켓 접근을 차단한다.
네이티브 WebView2까지의 네트워크 차단을 입증하는 수단은 아니므로 #179의
깨끗한 오프라인 환경 검증도 별도로 완료해야 한다. TCP 리스너는 종료 전의
스냅샷이며 전체 실행 기간의 샘플링이 아니다.

종료 코드 0은 **수치·동작 스모크 통과**다. 게이트 통과를 뜻하지 않는다.
`gate_status`는 `not_passed`로, 시각 대조와 다른 PC 확인은 `pending`으로 남긴다.
실행 중 창을 닫거나 다른 데스크톱이 단일 인스턴스를 점유하면 실패한다.

## 시각 대조와 최종 판정

```powershell
python packaging/render_desktop_evidence.py D:/gate2-check/desktop
```

리뷰 PC에만 Playwright와 Edge가 필요하다. 출력 PNG는 저장된 실제 DOM을 Edge에서
다시 렌더한 그림이다. 네이티브 창 캡처와 구분한다. 외부 요청·스크립트는 끈다.
기준 `docs/equivalence-baseline/screenshots/CHECKLIST.md` 전 항목을 대조하고
실제 다른 PC에서도 창·예측·세 XAI 기법을 확인한다.

숫자가 일치해도 문구·표 열·색·차트·필터 중 미확인/불일치가 있거나 다른 PC 실행
증거가 없으면 **Streamlit 시연**으로 판정한다. #177에 근거와 미완료 항목을 남기며,
게이트를 통과했다고 표시하거나 #188을 닫지 않는다.

기준에 기록된 IG pred/완결성 문제는 이 작업에서 계산식을 바꾸지 않는다.
동등성은 기존 결과와의 일치이며 과학적 타당성 검증을 뜻하지 않는다.
