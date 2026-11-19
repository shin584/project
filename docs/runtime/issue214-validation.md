# #214 이중 보관·보관본 보고 재생성·후속 인계

공식 Run ID는 `issue212-20261119-required01`, 동결 통합 커밋은
`0ef34155d365660f99b4c05a4a45ead90c549ce4`다. #212의 필수 측정과 #213의
계획된 조사는 완료됐으나 **필수 수치 검사 605 통과 / 43 실패 / 0 건너뜀,
Q17 최종 `fail`, 조립기 종료코드 1**을 유지한다. #214는 보관·보고·인계를
검증했으며 새 모델 측정, 후보 탐색, 부분 재개, 배포 검증을 수행하지 않았다.

## 보관 위치와 무결성

다음 두 합의된 run ID 디렉터리에 새 `issue212-20261119-required01-final-evidence.zip`을
실제로 복사하고 양쪽 전체 파일 checksum과 크기를 확인했다.

```text
C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01/
D:/MC_project-backups/dashboard-v3/issue205/issue212-20261119-required01/
```

두 최종 ZIP은 **657,774,005 bytes**, SHA-256은 모두
`601bee0c02af0fd35e7e8baec4349eda92b6caa7037ecbee755ad5fdfd783ff1`이다.
새 이름으로 복사한 #213의 정확한 불변 바이트이며 새 실행/계약 식별을 부여한
것이 아니다. 226개 ZIP 항목의 이름·크기·내용을 버전 관리된
[#213 실행·checksum 인덱스](evidence/issue213-20261119-investigations.json)의
225개 payload와 ZIP 내부 `archive-index-213.json`에 대조했다. 모든 payload의
SHA-256을 확인했다. 내부 인덱스 자체는 전체 ZIP의 pinned checksum으로 보호된다.

내용은 원시 배열, 동결 입력/계약/프로토콜, 관련 코드 snapshot, 모델과 manifest,
7개 완료 영수증, 환경, 실제 argv/종료코드/로그, 후보·정밀 진단, RSS 관측,
필수/최종 보고서와 사전 준비한 39개 dependency wheel이다. 설치된 의존성 트리는
wheel로 재구성하며 Python/stdlib/pip/Windows는 외부 전제다.

기존 두 세대도 양쪽 전체 checksum을 다시 확인했다. 덮어쓰지 않았다.

| 보존한 파일 접미사 | Bytes | SHA-256 |
| --- | ---: | --- |
| `-required-evidence.zip` (#212) | 412,698,030 | `64523d30c54892cca632636c90e0a12581dc5754b37b529819b8007f3d72915b` |
| `-investigation-evidence.zip` (#213) | 657,774,005 | `601bee0c02af0fd35e7e8baec4349eda92b6caa7037ecbee755ad5fdfd783ff1` |

모델 bytes는 등록된 `bundle/snapshot/extra/1/model-vault.json`의 기존 pinned vault와
대조했다. C:/D: 양쪽 `issue192-20261118/non-nt-models-<sha256>.zip`의 SHA-256은
`4c0206782d9f7c1aacfeaa763376bf3624cf31f86338b659e0973fdc5bfcd21c`이며,
그 안의 SpCas9/Cas12a checkpoint가 동결 계약과 각각 일치한다.
실제 전체 경로와 모델별 checksum은 아래 재생성 기록에 보존했다.

**C:/D:는 두 로컬 볼륨이다. 외부 호스트 또는 물리적으로 독립된 백업이라는
증거는 없으며 그렇게 주장하지 않는다.**

## 원 측정 폴더 없는 재생성

버전 관리된 [재생성 기록](evidence/issue214-20261119-replay.json)은 실행 manifest,
실행한 명령, 모든 복원 파일 checksum, 모델 vault 연결, 보고서 checksum, 차단
탐침과 실제 모듈 경로를 남긴다. 보관본만 읽었으며 원 측정 폴더를 checksum
기준으로 사용하지 않았다. 작업은 `dashboard_v3` 브랜치의 워크트리에서 수행했다.

실제 명령 (Windows x64 Python 3.13.7, NumPy 2.3.2):

```powershell
python -X utf8 docs/runtime/evidence/replay_archive_214.py --index docs/runtime/evidence/issue213-20261119-investigations.json --output C:/project-backups/dashboard-v3/issue205/issue214-20261119-replay01 --source 1 --finalize
$LASTEXITCODE
```

이 명령의 종료코드 **0은 보관·재생성 검증 성공**이다. 내부 수치 조립기는
별도로 **1**을 반환했다. 기록은 두 종료 의미를 합치지 않는다. 실제 재생성은
2026-11-19 04:06:06–04:07:07 UTC에 수행했다 (초기 archive checksum 시간 제외).

1. C:/D:의 #212/#213 보관본을 pinned checksum으로 확인했다.
2. D:의 완전한 #213 ZIP을 새 `replay01/restored/`에 복원했다.
3. 복원된 `bundle/snapshot/code/docs/runtime/assemble_verification.py`를
   `-I -B` 자식에서 실행했다. 현재 checkout의 조립기를 사용하지 않았다.
4. 원 checkout과 원 측정 base에 Python audit 경로 차단을 걸었다. 32개 원
   source/input와 원 manifest 탐침 **33/33 차단**, 실제 재생성 중 예기치 않은
   차단 위반 **0**이다. 네트워크·자식 실행·모델 계산 라이브러리 import도 거부한다.
5. 새 `summary-regenerated.json`이 보관된 `summary-final.json`과 **byte-identical**이다.
   판정뿐 아니라 영수증·checksum·원시 배열 식별·근거 연결도 동일하다. 전체
   보고서 SHA-256은 `374fe1f2eece0781c0a80cf5bcbde240f365435d9edb57d8c8b9941d375f130f`다.
6. 재생성 전후 복원 파일 **226/226 unchanged**, 새 측정 **0**을 확인하고
   C:/D: 최종 보관본을 새 파일명으로 복사·검증했다.

이 차단은 Python 프로세스 수준이며 OS 격리가 아니다. 보고서 재생성은 로컬
Python/NumPy를 사용했다. 과거 독립 복구 측정의 TF-free 환경을 새로 실행했다거나
이 재생성 환경이 깨끗한 배포 환경이라고 주장하지 않는다. 보고 관련 5개 모듈의
실제 경로도 모두 복원된 snapshot을 가리킨다.

미래 검토자는 동일 명령에서 **새 존재하지 않는 output**을 지정하고 `--finalize`를
생략한다. 이미 최종 복사본이 있으면 덮어쓰기를 거부한다. `--source 0`은 C:,
`--source 1`은 D:를 복원 원본으로 선택하며 항상 양쪽 checksum을 검사한다.
`--finalize` 없이도 모든 내용/보고 검증을 수행한다. 원래 임시 측정 폴더는 필요 없다.

직접 공개 보고 CLI만 실행하려면 복원 후 다음 명령을 사용한다. 새 보고서는
bundle 바깥에 두며, 이 명령은 예상대로 **1**을 반환한다.

```powershell
python -I -B -X utf8 C:/project-backups/dashboard-v3/issue205/issue214-20261119-replay01/restored/bundle/snapshot/code/docs/runtime/assemble_verification.py --work C:/project-backups/dashboard-v3/issue205/issue214-20261119-replay01/restored/bundle --output C:/project-backups/dashboard-v3/issue205/issue214-20261119-replay01/summary-direct.json
$LASTEXITCODE
```

이 직접 명령은 audit 차단을 추가하지 않는다. 차단·무변경·일치까지 검증하는
권장 명령은 위 `replay_archive_214.py`다. 원 측정의 모든 확장 argv/환경/종료코드는
ZIP의 `*.command.json`과 영수증에 있다. 원 측정 명령과 단계별 결과는
[#212 기록](issue212-validation.md), 조사 명령은 [#213 기록](issue213-validation.md)에
보존했다. 완료된 묶음의 측정 명령은 재실행하지 않는다. 새 측정이 필요하면
[required-212.md](required-212.md)의 새 run ID/빈 base 절차를 따른다.

## 수치 결과·정정·한계

| 범위 | 완료/수치 결과 | 해석 |
| --- | --- | --- |
| 필수 core | 583/583 검사, 543 통과 / 40 실패 | 수집 완료, 수치 실패 |
| 원래 진단 | 65/65 검사, 62 통과 / 3 실패 | 방향 비교 원래 실패 유지 |
| 전방/층/가중치/함수 일치 | 등록된 검사 통과 | 전방 SpCas9 9헤드; gradient/XAI는 NG만 |
| IG 32 기본값 | **5/8 실패** | 별도 256단계 **8/8 통과**와 5/16으로 합치지 않음 |
| IG 해상도 | 기본값 SpCas9 S2 scaled 최대차 실패 | 높은 상관만으로 통과라 하지 않음 |
| SHAP 기본값 | cross-seed 16/16 실패, default-vs-finer 8/8 실패 | additivity/동일 seed 재현 통과가 안정성 통과는 아님 |
| float32 고정 위치 | 72/72 관측, 2개 passing control 포함 | 원인 미해결 |
| IG 후보 비교 | 32회 완료 | SpCas9 128 / Cas12a 256은 제한된 완결성 개선 후보 |
| SHAP 후보 비교 | 184회 완료, 두 모델 적합 후보 없음 | 조건부 seed 3×3/fresh 재호출은 `not_applicable`, 0회 |
| 후보 비용 | 적용 대상 216/216회, 1395.821초 / 5400초 | 미시작 0, SHAP 최대 240회 이내; 추가 후보 탐색 없음 |
| 조건부 float64 | SpCas9 64/64 + Cas12a 16/16 관측, 두 자식 exit 0 | 둘 다 `unresolved`; 원래 간격 좌표 7곳은 여전히 실패 |
| float64 비용 | 총 635.892초 / 1800초 | 모델별 적격성 5분·진단 10분 준수; unavailable/time_limited 없음 |
| 독립 복구 (#212) | exit 0, 원 경로/네트워크 차단·TF 불가 확인 | 최소 forward/gradient 실행; 배포/OS 검증 아님 |

최종 보고의 `evidence_errors` 43개는 기존 실패 검사 항목이며 추가 무결성·누락·혼합
오류는 없다. 추가 조사는 완료됐고 원인 미해결/적합 후보 없음 상태다. 미검증,
unavailable, time_limited 조사 행은 없다. 조건부 SHAP 확인의 일반 카운터는 모델당
0 완료/28 incomplete를 보이지만, 행 상태가 `not_applicable`이므로 그 56회를
필수 누락이나 통과한 확인으로 세지 않는다. 전체 헤드/다른 경로·NT·배포와 아래
RAM 범위는 여전히 미검증이다.

Q17 판정은 필수 실패 **fail/1**이 우선한다. 필수 통과여도 조사 미검증·고정밀
불가/시간 제한은 incomplete/1이다. 필수 통과 및 조사 완료 후 원인 미해결/후보
없음은 completed_with_findings/0이며 전 기준 충족 때만 pass/0이다. 이번 실행은
첫 번째 경우다. 실행 완료와 수치 품질 성공을 구분한다.

[과거 #193 README](README.md)의 5/16, gradient 원인 단정, 전방/XAI 검증 범위,
최소 RSS의 배포 의미, 경고/legacy cache 담당 누락을 정정했다. 과거 `contract.json`,
`verification.json`과 원시 자료는 수정하지 않았고 새 식별을 소급 부여하지 않았다.
341.3 MB는 과거 최소 forward/gradient staging 관측이다. #212의 독립 복구 peak
413,839,360 bytes도 두 모델을 보유한 최소 forward/gradient 과정이다. 개발 환경의
단일 모델 IG peak는 SpCas9 441,520,128 / Cas12a 601,616,384 bytes이며 Cas12a는
TF가 실제 import됐다. #213 RSS는 프로세스별 표본이며 일부 초기 관측이 없다.
어느 값도 전체 UI/IG/SHAP process-tree peak나 지속 보유 한계를 입증하지 않는다.

## 후속 담당별 검토 가능한 인계

| 티켓 | 넘기는 근거와 필요한 작업 | 아직 승인하지 않은 범위 |
| --- | --- | --- |
| #195 | `snapshot/contract.json`, canonical `inputs.npz`, `core-raw.npz`의 Cas12a sequence+354 파생 특징 이중 입력과 각 입력 gradient/완결성. 시퀀스 입력과 특징 입력 기여를 함께 검토 | 다른 합성/composite 경로에 현재 결과 전용 불가; 선택 경로와 완결성 계약 재검증 필요 |
| #199 | non-NT SpCas9 NG의 기본 IG/SHAP·gradient 실패. 예측 표 유지, 해당 설명에 경고·실패 진단값·명시적 정밀 재시도 제공. 진행률/취소 요구 포함 | NG 외 헤드 XAI 품질 미승인. IG 128은 완결성 개선 후보일 뿐 기본값 아님; SHAP 확인 후보 없음 |
| #200 | non-NT Cas12a 현재 이중 입력 경로의 기본 실패. 경고·진단·정밀 재시도와 진행률/취소. 특징 기여와 sequence 표시 단위/경로를 함께 검토 | IG 256은 완결성 개선 후보이며 해상도 상위 비교 없음. 다른 경로 품질/SHAP seed 안정성 미승인 |
| #201 | #208이 식별/프로토콜 없는 legacy NT SHAP cache를 거부하고 fresh 계산하는 계약. 기존 4개 파일의 단위 교정·설명 프로토콜·품질 승인·재생성을 함께 조정 | non-NT 후보를 NT에 적용 불가; 64/4로 무조건 재생성 불가. NT 시간/RSS/품질 별도 측정 |
| #203 | `ram.json`, `staged.json`, `process-rss-213.jsonl`과 stage/보유 모델·TF import 여부. 계열 전환·프로세스 교체 후 메모리 회수와 유지 수명 검증 | 표본 RSS를 전체 트리 peak로 합산하지 않음; UI/WebView2/IG/SHAP 전체 RAM·취소/교체 미검증 |
| #204 | `prepared/`의 39개 wheel·고정 코드/모델과 #212 독립 복구 증거. 외부 Python/stdlib/pip/Windows 전제, Python audit 수준 경로/네트워크 차단 한계 | 최종 패키지 빌드·새 오프라인 Python provisioning·깨끗한 Windows·OS 격리·전체 RAM·물리 독립 백업 미검증 |

정밀 재시도 비용은 [#213 설정별 최대 wall time/오차/변동 표](issue213-validation.md)에
있다. 관측 IG 후보 최대는 SpCas9 128 **1.584초**, Cas12a 256 **4.911초**;
SHAP 512/32 최대는 **25.439초 / 36.220초**지만 두 모델 모두 후보 승인에 실패했다.
이는 고정 네 사례·NG/현재 이중 입력·CPU에서의 fresh 호출 비용이며 import/load,
여러 예측 행, 전체 화면 대기, NT 비용 보장이 아니다. 예측 행별 약 1분은 목표이며
새 60초 수치 gate가 아니다. 기본값 IG 32 / SHAP 64/4는 변경하지 않았다.

수치별 근거는 `summary-final.json`의 `claims`, `open_failures`, `execution`,
`investigations` → 같은 ZIP의 영수증과 `*-raw.npz`, `diagnostic-plan.json`,
`candidate-analysis-213.json`, `precision-analysis-213.json`, `precision-timing-213.json`으로
연결된다. 모든 경고·재시도·단위·RAM·배포 구현은 위 소유 범위에 남겼다.

## 선택 회귀와 리뷰

변경한 것은 보관본 재생성 기록 도구와 보고 문서다. #205에서 합의한 공개
검증 CLI/보고/종료코드 경계의 기존 `test/test_runtime_verification.py`를 선택했다.
변조·누락·다른 run 혼합 거부, 불변 보고 재조립, Q17 전 상태와 필수 실패 우선순위를
검증한다. 모델 수치/화면/캐시 구현은 변경하지 않아 해당 전체 서브트리나 저장소
전체 스위트를 실행하지 않았다. 보조 기록 도구는 위 실제 보관본 실행으로 검증했다.

| 실제 명령 | 통과 | 실패 | 건너뜀 | 결과 |
| --- | ---: | ---: | ---: | --- |
| `python -m pytest test/test_runtime_verification.py -q -rs` | 106 | 0 | 0 | 192.20초 |
| 위 `replay_archive_214.py ... --source 1 --finalize` | 보관/재생성 검증 완료 | 0 | 0 | 도구 exit 0, 내부 수치 조립기 exit 1 |
| `python -m py_compile docs/runtime/evidence/replay_archive_214.py docs/runtime/evidence/guarded_replay_214.py` | 구문 확인 | 0 | 0 | exit 0 |
| `python -X utf8 docs/runtime/evidence/replay_archive_214.py --help` | CLI 확인 | 0 | 0 | exit 0 |
| `git diff --check` | diff 확인 | 0 | 0 | exit 0 |

106개는 도구의 회귀 테스트이며 모델 품질 검사 605/43과 합치지 않는다. 변조/누락/
혼합 입력은 공개 CLI에서 `fail`/exit 1로 거부됨을 확인했다. 새 수치 측정 테스트로
대체하거나 이미 통과한 범위를 불필요하게 반복하지 않았다.

독립 Standards/Spec 리뷰의 기준점은 작업 시작 HEAD `7aaaf24`이며 변경 파일만
검토했다. Standards: 문서화된 규칙 위반 **0건**, smell 지적 **0건**. Spec:
수용 기준을 막는 지적 **0건**. Spec의 추가 정정 의견에 따라 기존 README의
NT legacy 파일 설명도 provenance 부족/미시드 생성 근거로 한정했다. non-NT
불안정성을 NT 품질 판정에 전용하지 않는다. 문서만 정정했으므로 링크와 diff를
다시 확인했으며 수치/CLI 테스트를 반복하지 않았다.

독립 리뷰에서도 두 볼륨의 기존/최종 6개 ZIP checksum과 보고 byte equality,
226개 복원 파일·33개 차단 탐침·기록 도구 checksum·동결 모듈 경로·과거 자료
보존을 확인했다. 최종 문서 정정은 실행했던 기록 도구 바이트를 바꾸지 않는다.

#205/#193/#191 및 기존 이슈의 수정·종료, 이슈 댓글·외부 메시지 발송은 수행하지
않았다. 이 문서가 후속 담당에게 전달할 검토 가능한 결과물이다.
