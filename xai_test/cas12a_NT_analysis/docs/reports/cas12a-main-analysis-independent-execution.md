# Cas12a NT 본 분석의 독립 실행 변경

2026-11-09. 기준 커밋: `9f8727ba7f35e0336faa4f479cc303fcbdc6ddb2`.
사용자는 이전 검증 기록을 보존하면서 그 실행시간 초과 차단을 이번 Test 본 분석의
시작 조건에서 분리하는 변경을 명시적으로 승인했습니다. 결정은
[ADR 0011](../../docs/adr/0011-cas12a-main-analysis-runs-independently-of-validation-time-blocks.md)에 기록했습니다.
이 기록은 [이전 구현 보고](cas12a-main-analysis-implementation.md)의 실행 정책 미확정 상태 이후의 변경입니다.

## 실행 결과

본 분석 함수에서 과거 `Study` 객체와 `reserved_seconds` 인자를 제거했습니다.
새 노트북도 `open_study`, `STUDY_ROOT`, `EVIDENCE_ROOT`, `RESERVED_SECONDS`를 요구하지
않습니다. 기존 ledger·receipt·charge·종료 시각은 읽거나 변경하지 않으며 새 세션의
시작·종료·경과 시간·환경·오류만 본 분석 결과 폴더에 기록합니다. 새로운 자동 시간
제한을 임의로 추가하지 않았습니다.

입력·모델·코드·환경 identity, 기존 Test membership, FP32/eager, 실제 token/mask/span,
finite 출력, IG endpoint/completeness/수렴, rollout parity, source별 완료·실패·미계산
coverage와 shard 재개는 유지했습니다. 과거 ADR·평가 artifact·학습 완료 checkpoint와
데이터 분할은 수정하지 않았습니다.

## 검증

- 노트북에 reservation 차단이 남아 있는 실패를 먼저 재현하고 제거했습니다.
- 변경한 공개 경계의 테스트 10개가 모두 통과했습니다. 생성된 노트북의 분석 셀 자체를
  작은 실제 NT 모델과 합성 Test로 실행했습니다. 외부 런타임 선택만 CUDA에서 CPU로 바꾸고
  노트북의 호출·기본 분석 범위·저장 경로는 그대로 사용했습니다.
- 실제 overrun 상태를 가진 합성 과거 study와 외부 receipt의 모든 bytes가 본 분석 전후
  동일한 상태에서 새 분석이 완료되는지 확인했습니다. 새 세션의 경과 시간도 기록됩니다.
- 중단 후 재개, 호환되지 않는 입력·환경 거부, ISM/IG 수치 실패 표시, 모델 없는 보고,
  정상 zero norm·undefined 보존과 사례 전용 ID의 통계 분모 분리를 함께 확인했습니다.
- 변경 Python 파일의 Ruff 검사·서식 검사와 staged diff 공백 검사를 통과했습니다.
  이전 수치 kernel과 과거 validation 경로의 전체 검사를 반복하지 않았습니다.

## Standards

독립 규칙 리뷰: 하드 위반 0건, 추가 개선 판단 0건. 승인된 실행 경계 분리에 한정되고
기존 ADR 보존과 신규 세션 기록이 문서·테스트에 일치합니다.

## Spec

독립 명세 리뷰: 지적 0건. reservation·기존 study 재개 의존성을 제거했고 입력·수치
검증과 저장·재개 경로는 유지합니다. 실제 Colab GPU 실행 증거와 코드 구현은 구분합니다.

## 새 전달물

- 노트북: `Cas12a_NT_Test1500_main_a9e50fa7f668_colab.ipynb`
- 소스 ZIP: `cas12a-nt-test1500-main-source-a9e50fa7f668.zip`
- ZIP SHA256: `a9e50fa7f668130635d44c125131d9f2a49f6cf0a9e10176040689e8fc552999`
- 전달 폴더: `.scratch/cas12a-test1500-main-independent-20261109/`

커밋 예정 소스만 별도 폴더에 복사해 전달물을 생성했습니다. 노트북 셀 문법과 ZIP의
모든 member bytes·manifest를 확인했습니다. Colab에서 GPU 선택 → 1번 설치 → 2번 같은
빌드 ZIP 업로드 → 3번 기존 자산 경로 확인 → 4번 분석 → 5번 보고 → 6번 다운로드
순서로 실행합니다. 기존 학습 모델과 데이터는 다시 업로드하지 않습니다.

이번 변경은 과거 검증용 시간 차단을 제거한 실행 가능한 코드 전달입니다. 실제
Test 1,500행 GPU 계산 완료나 Colab 런타임 가용성·완료 시간의 보장은 아닙니다.
