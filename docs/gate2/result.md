# #188 판정: Streamlit 시연

게이트 2는 **미통과**다. 수치 일치와 개발 PC 실행 성공만으로 전항목 통과를
판정하지 않는다. 화면의 일부 문구·컨트롤 형태/배치 차이가 남고, 최종 exe를
다른 PC에서 실행한 증거가 없다. #188은 열린 상태로 유지한다.

## 수용 기준 대조

| 기준 | 결과 | 증거 |
|---|---|---|
| 네 보기의 문구·열·색·차트·필터 전항목 동등 | 부분 충족 | [항목별 대조](CHECKLIST.md): 표·차트는 일치, 설명 요약과 배치 차이 |
| 고정 입력 후보 목록·정렬·점수 일치 | 통과 | [candidates.json](evidence/candidates.json), 사이트 13 / 상세 38행 |
| XAI 3기법 `np.allclose` | 통과 | 두 모델의 NPZ, IG/SHAP/Rollout 프로필·행렬 최대 차이 0 |
| CSV diff 일치 | 통과 | 38행, 2,716 bytes, UTF-8 BOM과 CRLF까지 동일 |
| 계산 설정·그래프 크기 동등 | 통과 | IG 32, SHAP 64/4; 크기 40~100/5/80 |
| 크기 변경·보기 전환 재계산 없음 | 통과 | 실제 요청/계산 카운터 유지, 기법 복귀 시 캐시 적중 |
| 다른 PC의 최종 exe 창·예측·세 XAI | 미확인 | environment.json의 different_pc=false |
| #177에 판정 코멘트 | 기록 | 하단 코멘트 링크 |

## 빌드와 실행 증거

최종 배포 대상은 `dist/XaiDashboard/` **전체 폴더**다.
`packaging/desktop.spec`가 실제 데스크톱 진입점을 묶으며, #179 시험 창과 구분한다.
모델 두 개·오프라인 tokenizer·Fixed WebView2·사전계산 SHAP·비교 기준을 포함한다.

[build-summary.json](evidence/build-summary.json)에 파일 수·크기·빌드 버전·소스
SHA-256·exe SHA-256·전체 release-manifest.json SHA-256을 남겼다. manifest 자체는
자기 해시 대상에서 제외하므로 파일 수/크기는 페이로드 기준이다.
빌드 시점 HEAD는 작업 시작 커밋이며, 최종 소스는 `source_sha256`으로 식별한다.
[archive-source-match.json](evidence/archive-source-match.json)은 exe 내부의 실제
데스크톱·코어 19개 모듈을 현재 소스를 컴파일한 코드 객체와 비교한 결과다.

전달용 `packaging/validate-desktop.ps1`로 전체 파일 해시를 확인한 후,
저장소 리소스를 찾지 않는 별도 작업 폴더에서 최종 exe를 실행했다.
[environment.json](evidence/environment.json)과
[validation.json](evidence/validation.json)에 실행 위치·호스트·결과를 기록했다.
exe는 매번 빈 HF 캐시를 만들고 동봉 tokenizer와 모델만 사용한다.
실제 창에서 두 모델 예측과 세 XAI 기법, CSV 저장, 크기 변경·탭 전환·기법 복귀를
완료했다. 검증 종료 코드 0은 개발 PC 수치·동작 스모크 통과이며 게이트 통과는 아니다.
검증 본체는 112.631초였으며, 페이로드는 6,862개 파일 / 5,474,935,798 bytes다.

CSV와 후보 JSON은 기준과 일치했고, 두 모델의 XAI 배열 최대 절대 차이는 모두 0이다.
SHAP은 `np.allclose`에 더해 완전 일치를 확인했다. Rollout 토큰 라벨도 일치한다.
Python 네트워크 시도 0건, 종료 전 TCP 리스너 0개, HTTP 서버 미생성을 확인했다.
네이티브 WebView2 전체 네트워크 차단이나 #179의 깨끗한 환경 통과 증거를 대신하지 않는다.

실제 DOM을 Edge에서 다시 렌더한 PNG 9장은 `evidence/`에 있다.
기준 6장과 두 번째 Cas12a 모델의 XAI 3장을 포함한다.
[체크리스트](CHECKLIST.md)에서 원본과 달라진 부분을 숨기지 않고 기록했다.

## 회귀 검사와 코드 리뷰

`python -X utf8 -m pytest test tests -q` — **206 passed, 0 skipped**.
실제 WebView2 테스트 3개를 포함하며 기존 SHAP 특이행렬 경고 1건이 있다.
[JUnit 원문](evidence/tests.xml)을 보관한다.

변경 Python 파일의 Pyright 검사는 오류 0건이다. Ruff는 기존
`detail_view.py`의 TRY004(ValueError를 사용하는 필터 계약)를 제외한 변경 파일에서
통과했다. 코어·XAI 계산식과 #181의 기준 수치·이미지는 변경하지 않았다.
비교 도구에 SHAP 완전 일치와 Rollout 토큰 라벨 확인을 보강했다.

[코드 리뷰](review.md): Standards 0건 / Spec 0건. 이는 게이트의 모든 실행·시각 기준이
통과했다는 뜻이 아니다. 전체 테스트에서 발견한 미리보기의 모델 import 회귀를 수정하고,
실제 창 테스트와 전체 검사를 다시 통과시켰다.

다른 PC 검증을 마칠 때는 [실행 절차](README.md)를 사용하고, 화면 차이까지 해소한
뒤 게이트 판정과 #177 코멘트를 갱신한다.

[#177 판정 코멘트](https://github.com/shin584/project/issues/177#issuecomment-6917648269)
