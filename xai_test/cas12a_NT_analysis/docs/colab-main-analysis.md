# Cas12a NT Test 1,500행 본 분석

본 분석 입구는 `nt_main.prepare_main` → `nt_main.run_main`입니다. validation 시연용
`nt_direct`와 구분합니다. 기존 source, partition, selected_export, durable_export.json을
읽으며 재분할·재학습·checkpoint 교체를 하지 않습니다.

저장소 루트에서 전달물을 생성합니다.

```powershell
python -m xai_test.cas12a_NT_analysis.nt_main_colab build --output <새 전달 폴더>
```

`Cas12a_NT_Test1500_main_<build>_colab.ipynb`와
`cas12a-nt-test1500-main-source-<build>.zip`은 같은 빌드입니다. ZIP 하나를 업로드하면
bytes와 모든 member를 자동 검증합니다. 추출 경로는 `/content/c12main-<build>`입니다.
이전 validation/direct ZIP과 혼용하지 않습니다. 결과 ZIP은 GPU 추론 없이 내려받습니다.

## 실행 설정과 현재 정책 경계

경로·IG 범위 설정은 노트북 3셀에 모여 있습니다. 기존 Drive 자산 경로를
유지합니다. 전수 Test는 production에서 1,500행을 요구하며 검증용 fixture만 작은 Test
membership을 허용합니다. validation ID 필드에 Test ID를 넣지 않습니다.

기본 IG membership은 결과 관측 전 전체 Test ID입니다. 전수 우선 준비이며 전수 IG의
실측 affordability를 뜻하지 않습니다. 표본을 사용할 때는 기존 `nt_selection`으로
**사전 고정한** coverage manifest와 비용 결정 근거를 전달해야 합니다. 표본 크기를
자동으로 400행으로 정하거나 실행 도중 결과를 보고 바꾸지 않습니다. 모든 선정 사례와
통계 membership의 합집합을 source ID당 한 번 계산합니다. 사례 전용 ID는 통계에 넣지 않습니다.

2026-11-09 사용자 승인에 따라 이번 Test 본 분석은 과거 검증의 시간 초과 차단과
분리했습니다. 기존 study·ledger·외부 receipt를 읽거나 수정하지 않으며 reservation 입력도
요구하지 않습니다. 세션별 시작·종료·실제 경과 시간과 오류만 새 결과 폴더에 기록합니다.
기존 미정산 시간을 지우거나 종료 시각을 추정하지 않으며, 자동 실행시간 제한을 새로
추가하지 않습니다. 이 결정은 [ADR 0011](../../docs/adr/0011-cas12a-main-analysis-runs-independently-of-validation-time-blocks.md)에 기록합니다.

Colab 실행 순서(노트북 코드 셀의 주석 번호 기준):

1. 본 분석 노트북을 열고 GPU 런타임을 선택합니다.
2. 1번 셀로 의존성을 설치하고 2번 셀에서 **같은 빌드의 소스 ZIP 하나**를 업로드합니다.
   기존 모델·데이터는 Drive에 있는 파일을 사용합니다.
3. 3번 셀에서 `SOURCE`, `PARTITION`, `MODEL_RUN` 경로를 확인하고 실행합니다.
4. 4번 셀로 바로 GPU 본 분석을 시작합니다. 별도 승인·예산 hash 붙여넣기는 없습니다.
5. 5번 셀로 저장 자료의 보고서를 확인하고 6번 셀로 원시 shard와 보고서 ZIP을 내려받습니다.

기본 결과 경로는 `/content/drive/MyDrive/cas12a-xai/main-test1500/<build>`입니다.
같은 경로를 동시에 두 런타임에서 실행하지 않습니다. Colab의 자체 런타임 제한이나
연결 중단은 발생할 수 있으며 저장 완료 shard는 같은 빌드·경로로 재개할 때 재사용합니다.

## 계산·저장·재개

모델은 실행 세션에서 한 번 로드합니다. correctness/warmup/비용 측정과 fresh reload를
자동 반복하지 않습니다. 계산은 전체 native FP32 raw/clipped 예측 → 예측·라벨 기반 사례
선정 → 전수 ISM → 전수 rollout → 고정 IG와 모든 사례 순서입니다.
`stages=("ism", "rollout", "ig")`를 명시해 방법별 계산을 나눌 수 있습니다.

필수 확인은 입력·모델·코드 identity, 원래 Test membership, 실제 34bp token/mask/span,
FP32/eager·finite 출력, IG endpoint/completeness/수렴, rollout parity입니다.
IG는 기존 zero reference와 200→400→800→1600 midpoint 기준을 유지하며 parameter
gradient만 잠시 끄고 원래 flag를 복구합니다. 가벼운 mutation version 확인은 소스마다,
전체 model state 확인은 방법 경계에서 수행합니다. 본 분석은 과거 ledger를 읽지 않습니다.

`plan.json`은 통계 ID·설정·identity를 고정합니다. `shards/<method>/<source>.json`이
마지막 저장 완료 확인이며 NPZ가 먼저 저장됩니다. 저장이 중단된 orphan/pending 파일은
coverage에 포함하지 않습니다. 재접속 시 입력을 다시 검증하고 같은 output으로
`prepare_main`과 `run_main`을 실행합니다. 호환되지 않는 코드·입력·설정은 거부하며
정상 완료 shard와 수치 실패 기록은 덮어쓰지 않습니다. 실패 기록을 지우고 성공으로
재해석하지 않습니다. 강제 종료되어 종료 기록이 없는 세션은 종료 시각을 추정하지 않습니다.

최초 계산 시 `execution.json`에 Python 및 NumPy/SciPy/PyTorch/Transformers/Safetensors
버전과 FP32/eager·배치 설정을 고정합니다. 재개 시 한 번 비교하며 다른 계산 환경의
결과를 합치지 않습니다. 각 shard는 생성 세션과 환경 identity를 참조합니다.
ISM 계산 중 발생한 수치 오류도 source ID와 원인이 있는 `failed` 기록으로 남깁니다.
예외는 원래대로 전달하고 부분 보고는 해당 실패를 미계산과 구분합니다.

진행 callback은 방법·완료 소스 수·요구 수·저장 위치를 전달합니다. `sessions/`는 실제
시작·종료·환경·오류를 기록합니다. 종료 파일 없는 세션의 종료 시각을 추정하지 않습니다.
GPU 계산이 중단돼도 저장된 결과로 부분 보고서를 만들 수 있습니다.

## 모델 없는 보고와 산출물

```powershell
python -m xai_test.cas12a_NT_analysis.nt_main_colab report --results <결과 폴더> --output <새 보고 폴더>
```

또는 `nt_main_report.report_main(results)`를 사용합니다. 이 경로는 torch/transformers
import와 모델 로드를 요구하지 않습니다. 완료된 shard를 한 번 읽고 저장 bytes, source
binding, 작은 수치 요약을 검산합니다. 새 보고 snapshot을 만들어 기존 증거를 보존합니다.

- `predictions.csv`, sidecar: source ID·서열·라벨·raw/clipped fraction·오차·PAM.
- `ism.npz`, sidecar: source × 34 positions × A/C/G/T. WT/non-WT mask, 102개 실제 변이,
  raw/clipped/relative delta, undefined mask와 PAM-breaking 태그. 각 소스의 분모는 위치당 3입니다.
  수치 출력이 없는 ISM 실패 소스는 NPZ 행으로 만들지 않고 sidecar의 실패 coverage에 기록합니다.
- `ig.npz`, sidecar: source × 12 token positions. CLS·9 DNA·2 padding signed mass와 품질·retry
  기록. 실패값은 NaN이며 원래 시도별 mass는 원시 shard에 보존합니다. 정상 zero norm은 성공일 수 있습니다.
- `rollout.npz`, sidecar, `rollout_diagnostics.json`: CLS 포함 10×10, 재정규화하지 않은
  DNA 9×9, parity, 제거 CLS 질량·query 유사성. encoder attention flow로만 해석합니다.
- `metrics.json`, `summary.json`: 전체/TTTV/TTTT raw·clipped Spearman, Pearson, MAE, MSE,
  집단 수와 undefined 사유. 방법별 요구·성공·실패·미계산 ID와 전체 완료 여부.
- `agreement.json`: 같은 길이 5개 6-mer의 주 통계, 9-token/PAM 제외/mass sensitivity,
  평균 profile, 길이군·순수 flank군 global rank·within/between 기여. 사례 전용 결과는 별도입니다.
- `cases.csv`, `cases_annotated.json`: 원래 선정 근거와 방법별 실제 설명·실패·미계산 상태.
- `position_profiles.json`, `token_profiles.json`, 정적 SVG 3개와 `report.ko.md`.

전체/TTTV/TTTT 집계는 실제 저장된 모집단과 성공 수를 분모로 명시합니다. 실패 IG를 0으로
채우지 않고 정상 undefined·낮은 상관을 보존합니다. validation 자료, Test 전수,
random subset, case-only를 섞지 않습니다. source ID로 join하며 서로 다른 방법의 행 순서를
join 키로 사용하지 않습니다. 각 NPZ의 sidecar에는 축·단위·layout·품질·입력/코드 identity와
선정 역할이 있습니다. SpCas9의 30bp·multi-head 축은 복제하지 않습니다.

구현 검증은 합성 CPU Test와 작은 native NT를 이용합니다. 실제 Test 1,500행 GPU 실행·
소요 시간·최종 성공 coverage는 Colab에서 반환한 산출물을 따로 확인해야 합니다.
