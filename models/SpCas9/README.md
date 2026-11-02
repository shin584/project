# SpCas9 9-Hydra v2

9개 단백질 head를 갖는 v2의 훈련·평가와 e_system 호환 추론 모델 export를 제공한다.
모든 명령은 저장소의 `models/`에서 실행한다. 훈련·평가와 필요한 published split은
[structure.md](structure.md)를 따른다.

## 추론 모델 export

TensorFlow, Keras 3, NumPy가 설치된 환경에서 기존 v2 checkpoint를 명시한다.
Export에는 훈련 데이터, published split, 원본 Excel, e_system 설치가 필요하지 않다.
아래 `v2_manual_RUN`을 실제 훈련 결과 폴더로 바꾸고 매번 새 destination을 사용한다.
Destination의 상위 폴더는 미리 존재해야 한다.

```powershell
Set-Location models
python -m SpCas9.spcas9_export --source SpCas9/v2_manual_RUN/9hydra_v2.keras --destination SpCas9/inference_RUN
if ($LASTEXITCODE -ne 0) { throw 'Export가 실패했습니다.' }
```

Python에서도 같은 공개 함수를 사용할 수 있다.

```python
from SpCas9.spcas9_export import export_checkpoint

manifest = export_checkpoint(
    "SpCas9/v2_manual_RUN/9hydra_v2.keras", "SpCas9/inference_RUN"
)
```

새 폴더의 `model.keras`가 추론 모델이며 `manifest.json`에 원본·출력 SHA-256,
head 매핑, 검증 입력, 오차, 라이브러리 버전과 exporter 소스 해시가 기록된다.
`validation.status == "passed"`이고 모델 SHA-256이 manifest와 일치해야 성공이다.
저장 후 재로딩과 원본 예측 비교를 마친 뒤 마지막에 manifest를 게시한다.
기존 destination과 원본을 덮어쓰는 경로는 거부한다. 중단·실패한 폴더도 재사용하지 않는다.

## 입력과 출력

- 입력: `In_Seq`, float32 `(N, 30, 4)`, A/C/G/T 채널 순서.
- 30bp 구성: 앞쪽 flank 4 + protospacer 20 + PAM 3 + 뒤쪽 flank 3.
- 출력: float32 `(N, 1)` 배열 정확히 9개의 Python 리스트.

| 인덱스 | 단백질 | 원본 출력 레이어 |
| --- | --- | --- |
| 0 | SpCas9(WT) | `Output_SpCas9` |
| 1 | SpCas9-NG | `Output_SpCas9_NG` |
| 2 | VRQR | `Output_VRQR_variant` |
| 3 | xCas9 | `Output_xCas` |
| 4 | Sniper | `Output_Sniper_Cas9` |
| 5 | SpCas9-HF1 | `Output_SpCas9_HF_1` |
| 6 | eSpCas9(1.1) | `Output_eSpCas9_1_1_` |
| 7 | HypaCas9 | `Output_HypaCas9` |
| 8 | evoCas9 | `Output_evoCas9` |

Exporter는 딕셔너리 순서가 아닌 레이어 이름으로 head를 선택한다.
학습 모델의 딕셔너리 출력과 head별 sample weight는 그대로 유지된다.
가중치·linear 활성화·실험 percentage/100 단위를 보존하며 유한한 음수, 0, 1 초과 값을
변환하지 않는다. 거리 Gaussian 가중치와 표시용 100배 계산은 소비 코드가 담당한다.
서열 추출·PAM 탐색·미지 염기 처리는 호출자가 담당한다.

```python
from pathlib import Path
import hashlib
import json
import keras

directory = Path("SpCas9/inference_RUN")
manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
artifact = directory / "model.keras"
assert manifest["validation"]["status"] == "passed"
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["artifact"]["sha256"]
model = keras.models.load_model(artifact, compile=False)
# predictions = model.predict(encoded_windows)  # 9개 배열의 리스트
```

Custom objects 없이 로드한다. 모델·manifest를 함께 보관한다.

## 테스트와 기존 검증 근거

`pytest`를 추가로 설치한 환경에서 합성 모델 기반 계약 테스트를 실행한다.
이 명령은 훈련이나 optimizer update를 수행하지 않는다.

```powershell
python -m pytest SpCas9/test/test_spcas9_export.py -q -k "not real_selected_checkpoint"
```

실제 checkpoint를 포함한 전체 export 테스트는 합의된 수동 실행 원본을 별도로 준비한다.
원본 SHA-256은 `24e31cc6b9df8520e2bf761d33f7bb475c4658a9a479242b67db882946d0d2b8`이다.
모델 가중치와 CSV 데이터셋은 이 코드 패키지에 포함하지 않는다.

```powershell
$env:SPCAS9_ACCEPTANCE_CHECKPOINT = 'C:/path/to/test_run/9hydra_v2.keras'
python -m pytest SpCas9/test/test_spcas9_export.py -q
```

실제 checkpoint 테스트는 원본 누락·체크섬 불일치 시 실패한다.
임의의 향후 checkpoint도 exporter CLI로 변환·재로딩 비교할 수 있지만,
위 역사적 acceptance 테스트의 원본은 고정돼 있다.

2026-10-30 이식 후 `models/`에서 실제 checkpoint를 지정한 27개 테스트가 모두 통과했다.
Exporter와 테스트의 Ruff 검사·포맷 검사도 통과했다.
검증 환경은 Python 3.13.7, TensorFlow 2.21.0, Keras 3.14.1, NumPy 2.3.2였다.

코드 이식의 기준은 `model_analysis`의 `fc33886`이며 exporter는 #96 구현을 그대로 가져왔다.
[보존된 #98 report](docs/evidence/issue98-report.json)와
[manifest](docs/evidence/issue98-manifest.json)는 이식 전의 실제 checkpoint·e_system 검증 기록이다.
그 안의 `xai_test/`, `.scratch/` 경로와 해시는 당시 실행을 가리키며 이 패키지의 실행 결과가 아니다.
당시 N=1·N=3 저장/재로딩 최대 오차는 모든 head에서 0.0,
실제 소비 코드의 27개 결과는 원본과 최대 약 `5.96e-8` 차이로 일치했다.
재현 절차와 소비 코드 검증 도구는
[원본 인계 문서](https://github.com/shin584/project/blob/fc33886/xai_test/docs/spcas9-export-handoff.md)에 있다.

## e_system 적용 범위

이 패키지는 별도 추론 파일을 생성한다. e_system의 기존 하드코딩된
`models/Multi-Cas_1IN_9Hydra_Divide_Testset.keras` 경로와 Streamlit 캐시를 변경하지 않는다.
9개 최종 점수가 모두 -1 이하일 때의 선택 오류·이전 후보 점수 재사용,
비유한값 처리, 상세 검색의 0~100 필터 제약은 기존 소비 코드에 남아 있다.
소비 코드 수정·명시적 모델 선택·캐시 갱신·v1 복귀 준비는
[#95](https://github.com/shin584/project/issues/95)에 별도 추적돼 있다.
[#94](https://github.com/shin584/project/issues/94)의 출력 호환성 검증은
생물학적 성능 개선이나 운영 적용을 의미하지 않는다.
