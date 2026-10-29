# SpCas9 9-Hydra v2 훈련·평가 구조

`models/SpCas9/` 안의 코드와 기존 published split으로 9개 head의 모델을 새로 훈련하고 평가한다.
원본 Excel이나 다른 프로젝트 모듈은 훈련·평가 실행에 필요하지 않다.

## 파일 구조

```text
models/                              # 명령 실행 위치
└─ SpCas9/
   ├─ __init__.py
   ├─ spcas9_train.py                # 모델 구성·훈련·저장·자동 Test 평가
   ├─ spcas9_eval.py                 # 예측·평가 보고서 생성
   ├─ spcas9_data.py                 # 서열 인코딩·head별 label 처리
   ├─ spcas9_metrics.py              # 공통 지표 계산
   ├─ spcas9_execution.py            # 실행 이력·출력 경로 관리
   ├─ spcas9_split.py                # split 관련 상수·체크섬·파일 처리
   ├─ spcas9_study_inputs.py         # 독립 평가의 고정 split 검증
   ├─ .gitattributes                # 입력 파일의 줄바꿈 변환 방지
   ├─ structure.md
   │
   ├─ published_split/
   │  ├─ train.csv                  # 별도 업로드: 29,134행
   │  ├─ val.csv                    # 별도 업로드: 3,676행
   │  ├─ test.csv                   # 별도 업로드: 5,186행
   │  └─ split_checksum.json        # 포함: 기존 split 검증용 메타데이터
   │
   ├─ v2_manual_<실행시각>/          # 전체 훈련 후 생성되는 예시 경로
   │  ├─ 9hydra_v2.keras
   │  ├─ training_log.json
   │  └─ test_eval/
   │     ├─ results.json
   │     ├─ predictions.csv
   │     └─ predictions_checksum.json
   │
   └─ v2_manual_<실행시각>_val_eval/  # 별도 validation 평가 시 생성
      ├─ results.json
      ├─ predictions.csv
      └─ predictions_checksum.json
```

CSV 데이터셋, 모델 가중치, 기존 실행 결과는 이 코드 업로드에 포함하지 않는다.
훈련 전에 기존 `train.csv`, `val.csv`, `test.csv`를 `published_split/`에 별도로 배치한다.
이 파일들은 `split_checksum.json`에 기록된 원본 바이트와 일치해야 한다.
`.gitattributes`는 Git이 입력 파일의 줄바꿈을 바꿔 체크섬이 달라지는 것을 방지한다.
기존 모델 가중치 없이 새로 훈련할 수 있으며, 독립 평가에는 평가할 `.keras` 파일을 지정한다.

## Python 환경

Python 3.12 이상과 TensorFlow, Keras 3, NumPy, pandas, SciPy가 필요하다.
기존 검증 환경은 Python 3.13, TensorFlow 2.21.0, Keras 3.14.1, NumPy 2.3.2,
pandas 2.3.3, SciPy 1.16.1이다.

## 전체 훈련

저장소 루트에서 PowerShell을 열고 실행한다. 패키지 이름은 대소문자를 포함해 `SpCas9`이다.
`SpCas9/` 안에서 `python spcas9_train.py`로 직접 실행하지 않고 상위 `models/`에서 모듈로 실행한다.
`python`은 필요한 패키지가 설치된 환경의 실행 파일을 사용한다.

```powershell
Set-Location models
$runDir = Join-Path 'SpCas9' ('v2_manual_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $runDir) { throw "이미 존재하는 폴더: $runDir" }
python -m SpCas9.spcas9_train --seed 42 --output-dir $runDir
if ($LASTEXITCODE -ne 0) { throw '훈련이 실패했습니다.' }
```

Adam(0.001), batch size 32, 최대 50 epoch, early stopping patience 5를 사용한다.
각 head에서 실제 측정된 label만 loss와 지표에 반영하며, 최소 `val_loss`의 가중치를 저장한다.
전체 훈련이 끝나면 저장된 최적 모델을 다시 읽어 Test 전체를 평가하고 `test_eval/`에 기록한다.
Test 지표는 checkpoint 선택에 사용하지 않는다.
9개 head별 MAE·MSE·Spearman·Pearson을 계산·출력하며 NGG/non-NGG 분리 결과는 생성하지 않는다.
출력 폴더는 실행마다 새 경로를 지정한다.

선택적으로 `--smoke`를 추가하면 작은 표본으로 1 epoch를 실제 훈련하고 저장·reload를 확인한다.
Smoke 실행에서는 자동 Test 성능 평가와 `test_eval/` 생성을 생략한다.

## 별도 validation 평가

전체 훈련이 성공한 뒤 같은 PowerShell 창에서 실행한다.

```powershell
$evalDir = $runDir + '_val_eval'
python -m SpCas9.spcas9_eval --split val --checkpoint (Join-Path $runDir '9hydra_v2.keras') --output-dir $evalDir
if ($LASTEXITCODE -ne 0) { throw '평가가 실패했습니다.' }
```

새로 훈련한 모델의 독립 평가는 `--split val`을 명시한다.
독립 평가의 기본 `--split test`는 코드에 고정된 기존 v2 checkpoint만 허용한다.
새 모델의 Test 결과는 전체 훈련 직후 자동 생성되는 `test_eval/`에서 확인한다.
독립 평가는 공개 split의 고정 체크섬·행 수·메타데이터·분할 간 중복을 검증한다.

## 결과와 실행 이력

- `9hydra_v2.keras`: 최소 validation loss에서 선택된 모델.
- `training_log.json`: seed, 하이퍼파라미터, split 체크섬, 환경, epoch별 기록과 선택 epoch.
- `results.json`: head별 측정 행 수와 지표, PAM-library·tRNA 층별 지표, 실행 이력.
- `predictions.csv`: split 메타데이터와 head별 정답·예측. 미측정 label은 결측값으로 유지.
- `predictions_checksum.json`: 예측 CSV·checkpoint·split의 체크섬.

효율 label은 원래 실험 percentage를 100으로 나눈 값이다.
일반 평가 보고서는 schema 3을 사용하며, schema 1/2 분기는 과거 연구 결과 재검증 호환용으로 남아 있다.
실행 이력의 소스 해시는 `SpCas9/*.py`, Git 변경 상태는 `SpCas9/`로 범위를 한정한다.
Git 실행 파일이나 저장소가 없어도 소스 해시를 기록하며 revision·dirty는 `null`로 남긴다.
