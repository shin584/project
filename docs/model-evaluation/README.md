# 배포 non-NT 모델의 test 성능과 출처 (#202)

2026-11-19, `dashboard_v3`에서 배포 checkpoint와 test 성능을 연결했다.
SpCas9의 기존 평가 인계와 Cas12a의 새 held-out test 평가를 구분한다.
이는 예측 성능·전방 런타임 parity의 근거이며 XAI 품질이나 배포 준비 완료의
판정은 아니다. 재학습·checkpoint 재선택·분할 변경·NT 비교는 수행하지 않았다.

## 실행 전 고정한 기준

[contract.json](contract.json)은 추론 전에 커밋 `7d1fc7d`로 고정했다.
표본, checkpoint SHA256, 전처리, 결측 처리, 출력 단위, 지표, clip 전후 기준,
parity 허용 오차가 들어 있다. float32 추론, `training=False`, batch 64,
CPU 4 threads를 사용하고 지표는 float64로 누적한다. 지표의 성능 하한을
새로 정하거나 test 결과에 따라 기준을 조정하지 않았다.

- 주 결과는 **clip 전 원출력**이다. target과 원출력은 published percentage/100,
  MAE는 fraction, MSE는 fraction², 상관은 무차원이다. MAE를 %p로 읽으려면
  100을 곱한다. 원출력의 [0,1] clip 후 지표는 별도 보조 결과로 보존한다.
- SpCas9의 미측정 target은 헤드별로 제외하고 n·누락 수를 기록한다.
  예측은 모든 5,186행·9헤드에 대해 생성하며 비유한 예측은 실패한다.
- Cas12a target은 학습 시와 같이 원본 %/100을 [0,1]로 제한한다. 원본의
  빈 행·주석 행 2개는 기존 loader 규칙으로 제외되어 15,000행이 남는다.
  test 1,500행에는 결측이 없고 추가로 제외한 test 행도 없다.
- 원출력은 모든 행·헤드에서 `abs(error) <= 1e-6 + 1e-5 * abs(reference)`를
  만족해야 한다. 원출력·clip 후 지표 모두 Pearson/MAE/MSE 절대 차이 `1e-5`,
  Spearman `1e-4` 이내여야 한다. 입력은 원 학습 인코더와 배포 인코더의
  float32 배열이 모든 행·입력에서 정확히 같아야 한다.

## SpCas9: 기존 평가 인계와 전체 test 재실행

배포 export SHA256은 `ddbf86aa…ce65`, 원 학습 checkpoint SHA256은
`24e31cc6…d2b8`이다. 보존된 `export-manifest.json.source.sha256`,
`test_eval/results.json.predictions.checkpoint_sha256`, `training_log.json`의
checkpoint SHA가 모두 같은 원본을 가리키는지 검사했다.
전체 식별값·헤드 이름 매핑·학습 로그·split checksum은
[report.json의 handoff](evidence/report.json)에 있다.

원본은 validation loss로 epoch 41을 선택했고, 학습 29,134 / validation 3,676 /
published test 5,186행이다. 기존 test 평가는 같은 원 checkpoint에 대한
`manual_v2_post_training_test` 실행이었다. 기존 기록의 dirty source revision과
snapshot digest는 그대로 인계한다. 아래 수치는 보존된 예측 CSV에서 재계산해
기존 JSON과 `1e-12` 이내로 일치한 **기존 평가**이며 새 독립 연구가 아니다.

| 기존 head 이름 | 측정 n | 미측정 | Pearson | Spearman | MAE | MSE |
|---|---:|---:|---:|---:|---:|---:|
| SpCas9 | 5021 | 165 | 0.941453 | 0.921874 | 0.051150 | 0.005822 |
| SpCas9-NG | 3982 | 1204 | 0.867852 | 0.840964 | 0.064207 | 0.006835 |
| VRQR variant | 3979 | 1207 | 0.871195 | 0.880612 | 0.058379 | 0.006752 |
| xCas | 5170 | 16 | 0.869820 | 0.880965 | 0.060273 | 0.007191 |
| Sniper-Cas9 | 5170 | 16 | 0.940219 | 0.922102 | 0.049677 | 0.005690 |
| SpCas9-HF.1 | 5177 | 9 | 0.781161 | 0.789504 | 0.070906 | 0.014547 |
| eSpCas9(1.1) | 5176 | 10 | 0.861187 | 0.847474 | 0.062733 | 0.010586 |
| HypaCas9 | 5177 | 9 | 0.838629 | 0.833665 | 0.059923 | 0.010095 |
| evoCas9 | 5161 | 25 | 0.670067 | 0.633846 | 0.044789 | 0.008374 |

이번에는 전체 5,186행에서 원 Keras/TensorFlow, export Keras/TensorFlow,
배포 Keras/torch를 새로 추론했다. 기존 소수 fixture parity를 대체 근거로 쓰지 않았다.

| 새 전체 test 비교 | 최대 절대 출력 차이 | 판정 |
|---|---:|---|
| 원본 TensorFlow → export TensorFlow | 0 | 9헤드 통과 |
| 원본 TensorFlow → 배포 torch | 2.384185791e-7 | 9헤드 통과 |
| 보존된 예측 CSV → 새 원본 TensorFlow | 1.639127731e-7 | 9헤드 통과 |

원본/배포 원출력 지표의 최대 차이는 Pearson `1.69e-8`, Spearman `1.82e-6`,
MAE `1.28e-9`, MSE `4.01e-10`이었다. 일부 출력이 [0,1] 밖에 있으므로 SpCas9의
clip 후 지표는 원출력 지표와 다르다. 각 헤드의 새 raw/clipped 결과와 두 런타임의
차이는 보고서 `new_full_test_comparisons.spcas9`에 모두 보존했다.

## Cas12a: 선택된 checkpoint의 별도 test 평가

checkpoint SHA256 `94ac9778…b28f`를 그대로 사용한다. 원 학습은 validation loss로
epoch 5를 선택했다. 기존 `validation_predictions.csv` 1,500행과 validation MSE
`0.0445623666`은 **선택용 validation**이며 test 결과가 아니다.

선택 이전부터 보존된 partition manifest의 학습 12,000 / validation 1,500 /
test 1,500 membership을 그대로 사용했다. `configuration.json`에 보존된 세 membership과
전체 manifest digest `e45c8b11…dc9c`, partition digest `680fc75f…2f6d`, 원본 파일·표·행별
서열·학습 partition digest를 원 학습 소스의 검증기로 대조했다. 재분할하지 않았으며
test source-row/서열/표적 및 두 입력 배열을 보존했다. 34bp 입력과 354개 파생 특징은
원 학습 인코더와 배포 인코더에서 전체 test에 대해 정확히 일치한다.

| 새 held-out test, n=1500 | Pearson | Spearman | MAE | MSE |
|---|---:|---:|---:|---:|
| 원 Keras/TensorFlow, raw | 0.761203862 | 0.748909696 | 0.154544342 | 0.043484108 |
| 배포 Keras/torch, raw | 0.761203859 | 0.748909696 | 0.154544342 | 0.043484108 |
| 배포 Keras/torch, clip [0,1] | 0.761203859 | 0.748909696 | 0.154544342 | 0.043484108 |

이 test에서 출력은 이미 [0,1] 안에 있어 clip 전후 결과가 같다. 원본/배포 최대
절대 출력 차이는 `1.490116119e-7`, 지표 차이는 Pearson `2.17e-9`, Spearman `0`,
MAE `5.09e-10`, MSE `3.10e-10`으로 동결 기준을 통과했다.

## 실제 보관·재현 가능 여부와 한계

[index.json](evidence/index.json)이 **11,087,416-byte**
[issue202-run01.zip](evidence/issue202-run01.zip)의 SHA256을 고정한다.
archive에는 다음의 실제 바이트가 있으며, 외부 원본 폴더 없이 복원할 수 있다.

- 두 배포 checkpoint와 SpCas9 원 학습 checkpoint, 기존 학습·export·평가 기록,
  SpCas9 train/val/test CSV와 기존 예측 CSV 등 #192 manifest의 22개 자산 전체.
- Cas12a 원본 xlsx와 별도 partition.json, 기존 선택용 validation 예측 CSV 및
  학습 source_snapshot.zip. 원본 데이터 파일이 없다는 가정으로 평가를 생략하지 않았다.
- 이번 실행의 평가 도구·배포 모듈 소스, contract, 환경 버전, 단계별 명령·종료 코드·로그,
  전체 test의 source-row·서열·target·전처리 입력·원본/export/배포 예측 NPZ.

[registration.json](evidence/registration.json)은 측정 전 입력과 소스의 상대 경로·SHA256을,
[execution.json](evidence/execution.json)은 실제 명령·종료 코드·시간을 보존한다.
측정은 archive 안의 복사된 소스로 수행했고 단계마다 동결 파일의 해시를 확인했다.
`audit`는 전체 raw 배열에서 모든 지표와 판정을 재계산해 보고서 전체와 비교한다.
해당 명령은 재추론과 구분한다.

SpCas9 상위 원본 xlsx도 이 개발 PC에는 남아 있지만 이번 archive에는 기존 published
CSV를 보존했다. 이는 기존 test의 재평가에 충분하며, 원 논문의 데이터 취득이나
분할 생성 절차 전체를 재현했다는 뜻은 아니다. 기존 SpCas9 dirty 평가 소스의
동시점 snapshot 전체는 채택 vault에 없다. 기존 JSON의 파일별 digest는 보존하지만
그 역사적 실행 바이트 전체의 복원은 주장하지 않는다.

Cas12a 학습 소스 snapshot은 #117의 **사후 복원**이다. 일치하는 Git revision과
validation replay 근거를 보존하나, 학습 당시 실행 소스·과거 fit 부재를 소급해서
증명하지 않는다. 이번 새 test 실행의 소스와 입력은 별도로 동결했다.

여기서는 Keras 3.14.1, TensorFlow 2.21.0, torch 2.12.0+cpu, Python 3.13.7을 사용했다.
torch가 채택 배포 backend이고 TensorFlow는 원 모델의 참조 실행용이다. 개발 PC의
Keras가 TensorFlow 설치를 탐지하므로 이 평가의 torch 프로세스에도 TF import 로그가
있다. TF 없는 배포본·오프라인 PC·패키징·RAM은 이번 성능 평가로 검증하지 않는다.

## 재실행

새 작업 폴더만 사용한다. 기존 결과를 덮어쓰지 않는다. 작업 폴더의 `runtime/` 아래
소스와 모델은 archive에 포함되며 Python 패키지는 위 버전으로 별도 준비해야 한다.

```powershell
# 최초 실제 실행 (worktree 루트)
py -3.13 -X utf8 docs/model-evaluation/evaluate.py run --work .packaging/issue202-run01 --cas12a-source ../cas12a/dataset/kim_2018_cas12a_dataset.xlsx --cas12a-partition ../cas12a/models/NT_cas12a/result/20261101-nt500m-full/partition.json
# prepare/reference/deployed/report/audit 모두 exit 0; 28 pass / 0 fail / 0 skip

# archive SHA256을 index.json과 비교한 뒤 새 디렉터리에 복원
$archive = 'docs/model-evaluation/evidence/issue202-run01.zip'
$index = Get-Content -Raw docs/model-evaluation/evidence/index.json | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $index.sha256) { throw 'Archive checksum mismatch' }
py -3.13 -m zipfile -e $archive .packaging/issue202-recovery

# 보존된 배열에서 지표/판정 재계산 (모델 재추론 아님)
py -3.13 -X utf8 .packaging/issue202-recovery/issue202-run01/runtime/docs/model-evaluation/evaluate.py audit --work .packaging/issue202-recovery/issue202-run01
# 28 pass / 0 fail / 0 skip, exit 0

# 복원된 소스·checkpoint·데이터만으로 전체 test 재추론
py -3.13 -X utf8 .packaging/issue202-recovery/issue202-run01/runtime/docs/model-evaluation/evaluate.py run --work .packaging/issue202-reinference --cas12a-source .packaging/issue202-recovery/issue202-run01/inputs/cas12a.xlsx --cas12a-partition .packaging/issue202-recovery/issue202-run01/inputs/partition.json
# 실제 재추론: 28 pass / 0 fail / 0 skip, 모든 단계 exit 0
```

이 별도 재추론의 [명령·결과](evidence/reinference.json),
[등록](evidence/reinference-registration.json), [보고서](evidence/reinference-report.json)도
보존했다. raw 파일의 모든 SHA256이 첫 실행의 archive와 같으므로 raw 배열을 중복
저장하지 않았다. 두 보고서의 전체 test 비교·지표도 정확히 같다.

실행 완료와 남은 한계, 선택한 회귀 범위는 [#202 구현 기록](../issue202-implementation.md)에 있다.
