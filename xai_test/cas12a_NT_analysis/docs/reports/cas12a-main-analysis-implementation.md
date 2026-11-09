# Cas12a NT 본 분석 구현

2026-11-09. 기준 커밋: `240a6a02a43f9c10a3a06861e02726d46d992247`.
근거: [수정 명세](../planning/cas12a-main-analysis-revision-spec.md)와
[기존 인계](../planning/cas12a-main-analysis-handoff.md). 이 문서는 그 인계 이후의
**코드 구현 상태**이며 실제 Test 1,500행 GPU 완료 보고가 아닙니다.

## 구현 결과

`verify_test_inputs`가 원래 Test membership을 별도 identity로 검증합니다.
`prepare_main`은 새 예측 관측 전 IG 통계 ID·수치 기준·코드/입력 identity를 고정합니다.
`run_main`은 모델 한 번 로드 → 전체 예측·예측만으로 사례 선정 → 전수 ISM/rollout →
고정 IG와 모든 사례의 source ID 합집합을 계산합니다. 기존 계산 커널과 사례·일치도
로직을 재사용하며 correctness/warmup/cost/reload 단계를 자동 반복하지 않습니다.

소스별 JSON 완료 확인과 checksum이 있는 NPZ를 저장합니다. 재개 시 완료 shard를
재사용하고 orphan/pending 파일을 제외합니다. 수치 실패는 원인과 source ID를 보존하며
성공·실패·미계산 coverage를 분리합니다. Python/핵심 패키지 버전·FP32/eager 설정을
한 번 고정하고 생성 세션에 연결합니다. 호환되지 않는 환경의 계산은 합치지 않습니다.

`report_main`은 모델 없이 저장 자료를 읽어 CSV, 방법별 NPZ/sidecar, 성능·coverage,
주/보조 IG–ISM 통계와 group 진단, 위치·토큰 profile, rollout 진단, 사례 설명,
정적 SVG와 한국어 보고서를 생성합니다. random subset과 case-only 분모를 섞지 않으며
실패 IG를 0으로 채우지 않습니다. 정상 zero norm·undefined는 이유와 함께 보존합니다.

설명 및 실행 명령은 [본 분석 안내](../colab-main-analysis.md)에 있습니다.
기존 자산·분할·checkpoint·평가 자료·study·receipt·charge·ADR·이슈는 교체하거나
재작성하지 않았습니다. 기존 미커밋 구현 중 직접 필요한 IG parameter-gradient 최적화와
native predictor mutation-version 확인만 선행 변경으로 포함하며 다른 변경은 보존합니다.

## Standards

독립 규칙 리뷰: 문서화된 하드 위반 0건, 판단상 개선 1건.

“가능한 Duplicated Code”: 저장·검산·보고에 NPZ shape/dtype 스키마 생성이 반복됐습니다.
공용 `array_schema`로 모아 수정했습니다.

## Spec

독립 명세 리뷰: P2 지적 2건, 모두 수정했습니다.

1. “수치 실패는 실패로 남긴다”: ISM의 nonfinite/parity 예외를 미계산으로 분류하던
   문제를 수정했습니다. 실제 nonfinite mutant를 가진 합성 checkpoint로 실패를 재현하고,
   source별 failed 기록과 부분 보고를 확인했습니다. 원래 예외도 호출자에게 전달합니다.
2. “호환되는 완료 결과만 재개에 사용”: 입력·코드 외에 최초 계산 환경도 고정하고
   재개 시 한 번 비교합니다. shard는 생성 세션·환경 identity를 참조합니다. 다른 Python
   버전으로 재개할 때 새 계산을 거부하는 회귀 테스트를 통과했습니다.

추가 독립 리뷰를 반복하지 않고 지적된 경계의 회귀 테스트로 수정을 확인했습니다.

## 검증

- 본 분석 집중 테스트 9개 통과: 실제 작은 native NT를 사용한 합성 Test end-to-end,
  모델 import를 금지한 별도 프로세스의 보고, 중단·재개, identity 거부, 정상 zero norm과
  raw/clipped 성능의 알려진 값, IG 수치 실패, 과거 미정산 study 보존, 환경 변경 거부,
  ISM 수치 실패 분류, 400-row 고정 표본과 case-only 분모, Colab 코드·ZIP identity.
- 커밋 예정 파일만 별도 폴더에 복사한 상태에서 end-to-end와 Colab 묶음 테스트 2개 통과.
  따라서 다른 미커밋 작업이 있어야만 실행되는 형태가 아님을 확인했습니다.
- 변경 Python 파일의 Ruff 검사와 staged diff 공백 검사 통과.
- Cas12a NT 하위 테스트 실행: 229개 통과, 재개 테스트 1개 실패. 이 실행과 병행해
  본 분석 소스를 수정하여 코드 identity guard가 변경을 감지한 것입니다. 이후 소스를
  고정한 집중 테스트 9개 실행에는 해당 재개 테스트도 포함되며 모두 통과했습니다.
  변경하지 않은 수치 kernel·기존 validation 경로의 테스트는 통과했습니다.

저장소 전체 테스트나 실제 1,500행 CPU 설명 계산은 실행하지 않았습니다.

## 전달물과 남은 실제 실행

커밋 훅의 함수 선언 서식 변경까지 반영한 최종 빌드 `e3cebb1d0d7e`:

- `Cas12a_NT_Test1500_main_e3cebb1d0d7e_colab.ipynb`
- `cas12a-nt-test1500-main-source-e3cebb1d0d7e.zip`
- source ZIP SHA256: `e3cebb1d0d7eae60338e992def877d8479d7006a213cc3859db2b22eee41250b`
- 로컬 전달 폴더: `.scratch/cas12a-test1500-main-delivery-20261109-final/`

기존 overrun·미정산 시간과 간소화된 실행 정책의 관계는 미확정입니다. 추가 예산이나
400-row fallback을 임의 승인하지 않았습니다. 노트북의 reservation은 `None`이며 실제
GPU 실행은 기존 cumulative `Study`와 결정된 reservation을 요구합니다. 기존 차단 상태는
그대로 거부합니다. 코드 준비·모델 없는 보고는 이 결정 없이도 사용할 수 있습니다.

정책이 정해진 뒤 같은 빌드의 노트북·ZIP으로 Colab GPU에서 실제 Test를 계산하고,
반환 산출물의 source coverage·수치 품질·성능·설명·사례를 확인해야 합니다. CPU 합성
테스트와 현재 전달물은 실제 T4 처리 속도나 Test 1,500행 완료 증거를 대체하지 않습니다.
