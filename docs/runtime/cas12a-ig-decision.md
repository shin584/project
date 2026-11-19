# #195 Cas12a CNN IG 경로 결정과 결과 계약

2026-11-19. 부모 #191, 선행 #193. **합성 경로를 채택한다.** 단일 DNA에서 자동
계산한 특징까지 포함하는 서열 설명이 화면 목적에 맞고, 실제 채택 checkpoint에서
특징 동등성·soft gradient·chain rule·함수 일치를 확인했다. **수치 품질 통과를
뜻하지 않는다.** 고정 수치 검사는 124 통과 / 8 실패 / 0 건너뜀, 최종 `fail`,
종료코드 1이다. 합성 경로 C2는 256단계에서도 완결성 기준에 실패했다.

## 결과 확인 전 동결

[cas12a-ig-contract.json](cas12a-ig-contract.json)에 두 경로의 정의, C1–C4,
54개 특징 parity 입력, 비정규화 soft fixture, gradient 확인법과 모든 허용 오차를
신규 측정 전에 고정했다. #193의 공통 허용 오차를 이어받았으며 과거 이중 입력
측정 결과를 합성 경로 결과로 사용하지 않았다. 코드 수준의 구현 가능성 조건과
수치 품질 조건도 측정 전에 구분했다. 기준이나 기본 계산량을 결과에 맞춰 바꾸지 않았다.

- 계약 SHA-256: `d79c10934ce9f42fe354b881448cbe48dd5b8a941b4672214abc6005028ffe54`.
- 실행 직전 등록: `2026-11-19T05:58:52.472797+00:00`.
- 소스 기준 HEAD: `93d1115338d70f4d4b09ff4ea5d2566d38cde0fc` + 등록된 변경 파일.
  미커밋 소스로 실행했으므로 HEAD만으로 측정 코드를 특정하지 않는다.
  [registration.json](evidence/issue195-comparison01/registration.json)의 파일별 SHA와
  [snapshot.zip](evidence/issue195-comparison01/snapshot.zip)이 실제 실행 소스를 특정한다.
- checkpoint: `models/non_nt/cas12a-cnn-20261102/cas12a_cnn_manifest.keras`,
  SHA-256 `94ac977865efa1cdad734ae05e033cbaf40b38e109013a6d2fe2b9361aa8b28f`.
- Python 3.13.7, Keras 3.14.1, torch 2.12.0+cpu, NumPy 2.3.2, h5py 3.14.0;
  torch backend / CPU 4 threads / inference 모드 / float32, 적분 누적 float64.

## 두 경로의 수학적 정의와 의미

입력 `x`는 ACGT 채널 순서의 `[34,4]` 배열, `p=x[8:31]`은 23개 protospacer
위치다. `D(x)`의 354개 특징은 다음 순서의 연속 다항식으로 정의한다.

1. `sum(p[:,C]+p[:,G])/23`.
2. `sum(p[-8:,C]+p[-8:,G])/8`. 학습 코드의 `gc_seed` 이름은 PAM에서 먼
   마지막 8개 위치를 가리킨다. 생물학적 seed 영역이라는 주장이 아니다.
3. `p[i,b]*p[i+1,c]`, `i=0..21`, `b,c=ACGT` 순서로 펼친 352개 값.

미분 구현은 [cas12a/continuous.py](../../cas12a/continuous.py), 원 특징의 기준은
보존된 `cas12a.features.extract_seq_features`다. ACGT one-hot 및 N의 영벡터
인코딩에서 같은 354개 값을 낸다. soft 입력을 문자열/argmax로 되돌리지 않는다.

| 항목 | 합성 경로 (채택) | 이중 입력 경로 (비교·기존 검증 보존) |
| --- | --- | --- |
| 함수와 독립 입력 | `F(x)=M(x,D(x))`, 서열 하나 | `G(x,z)=M(x,z)`, 서열과 특징 둘 |
| 기준 입력 | `x0=0`, `D(0)=0` | `x0=0`, `z0=0` |
| 중간 모델 입력 | `(alpha*x,D(alpha*x))` | `(alpha*x,alpha*D(x))` |
| GC / dinucleotide 스케일 | `alpha` / `alpha²` | `alpha` / `alpha` |
| gradient | `partial_x M + J_D(x)^T partial_z M` | `partial_x M`, `partial_z M` 각각 |
| IG | `x * integral(grad F(alpha*x))` | `x * integral(partial_x M)`, `z * integral(partial_z M)` |
| 서열 그래프의 의미 | 특징 경로까지 포함한 단일 서열 기여 | 서열 입력 부분의 기여만 표시 |
| 완결성의 대상 | 위치별 signed 합과 `M(x,D(x))-M(0,0)` | 두 입력 signed 합의 합계와 같은 끝점 차이 |
| 수치 검증 | 원 특징 parity, soft VJP, chain rule, 좌표 차분, 완결성·해상도 | 입력별 좌표 차분, 공동 완결성, 서열 부분 해상도 |
| 화면 계약 | 기존 34위치 그래프로 전체 서열 설명 | 부분 설명 라벨과 펼쳐보기의 입력별 signed 합·공동 완결성 필요 |

같은 끝점을 갖더라도 다른 중간 경로를 적분하므로 attribution 배분이 다르다.
두 경로의 위치별 일치는 합격 조건에 넣지 않았다. 이중 입력의 특징 기여를 염기에
임의 배분하거나 입력별 합계를 생물학적 중요도 비율로 변환하지 않는다.

합성 경로는 파생 특징이 사용자와 독립인 입력처럼 보이지 않게 하고 단일 DNA의
전체 함수에 대한 위치별 설명을 제공한다. 이중 입력도 정의 가능한 수학적 경로지만
서열 부분 그래프만으로 전체 예측을 설명할 수 없다. 합성 경로가 더 정확한 생물학적
모형이거나 더 적은 적분 단계로 수렴한다는 이유로 채택한 것은 아니다.

## 실제 모델 검증 결과

실행 명령은 워크트리 루트에서 다음과 같다. 출력 폴더는 새 경로여야 한다.

```powershell
python -X utf8 docs/runtime/verify_cas12a_ig.py --output .packaging/issue195/comparison01
```

검증기는 실행 전 계약·코드 snapshot과 checksum을 등록하고, 실행 중 소스 변경을
거부하며, 원시 배열과 모든 판정을 남긴다. [report.json](evidence/issue195-comparison01/report.json),
[arrays.npz](evidence/issue195-comparison01/arrays.npz), [manifest.json](evidence/issue195-comparison01/manifest.json)을
저장소에 보존했다. 배열/등록 파일의 checksum은 보고서에도 연결돼 있다.

- 54서열 × 354특징 parity: 최대 절대 오차 **0**. GC 선형 / pair 제곱 스케일도
  alpha 0, 0.25, 0.5, 1의 모든 사례에서 오차 **0**.
- 비정규화 float64 soft fixture의 가중 특징합 VJP: 독립 해석 미분 및 `h=1e-5`
  중앙차분과 비교. 최대 절대 오차 **2.10e-11** 이하, 기준 `1e-8` 통과.
- 실제 모델 합성 gradient와 `partial_x M + J_D^T partial_z M`: C1–C4의
  alpha 0.5, 1에서 최대 절대 오차 **1.73e-8** 이하, 기준 `1e-6` 및 cosine
  `>=0.9999` 통과. `J_D^T`는 tensor 구현과 별도의 해석 미분으로 계산했다.
- 두 경로의 gradient 호출 출력과 동일 중간 입력의 전방 출력, 합성 raw 함수의
  soft/끝점 출력: 최대 오차 **0**, 기준 `1e-6` 통과.
- 실제 모델 좌표 중앙차분 `h=1e-2`: 합성 8/8 통과, 최대 상대 L2 **0.019248**.
  이중 입력 16개 입력별 검사 중 **C3 / alpha=1 / Input_Seq 1개 실패**
  (상대 L2 **0.022229** > 0.02; cosine 0.999754). 두 입력을 합쳐 실패를 숨기지 않았다.
  이는 제한된 지점의 검사이며 #193의 과거 gradient 실패를 취소하지 않는다.
- 원시 전방·gradient·기여도·진단 배열은 모두 유한하다.

다음은 완결성 절대 오차의 **raw 확률 단위**다. 기준은 모든 칸에 `<=0.001`
(0.1 %p)이며 실패는 굵게 표시했다. 예측 기준값 `M(0,0)=0.7279022932`는 실제
DNA의 예측값이 아니다.

| 사례 | 합성 32 | 합성 256 | 이중 입력 32 | 이중 입력 256 |
| --- | ---: | ---: | ---: | ---: |
| C1 | 0.000590845 | 0.000069462 | 0.000107853 | 0.000016280 |
| C2 | **0.003990365** | **0.001227337** | **0.004434473** | 0.000515744 |
| C3 | **0.001888127** | 0.000099079 | **0.002765618** | 0.000128641 |
| C4 | **0.001309912** | 0.000020229 | **0.001210572** | 0.000616112 |

32→256단계 위치별 안정성은 두 경로 각각 4/4 통과했다. 합성 경로의 최저 Pearson은
0.999985, reference peak 대비 최대 차이는 0.007097, 유의 위치 부호 일치는 100%다.
이중 입력은 각각 0.999959, 0.016596, 100%이며 이 수치는 서열 입력 부분에 대한
안정성이다. 완결성은 이중 입력의 두 signed 합을 함께 검사했다.

총 **124 통과 / 8 실패 / 0 건너뜀**은 132개 수치 판정의 수다. 실패는 완결성
7개와 이중 입력 좌표 gradient 1개다. 합성 경로 구현 가능성은 확인했지만 기본 32단계는
완결성 1/4만 통과했고, 256단계도 C2에서 실패했으므로 보편적인 정밀 권장값으로
승격하지 않는다. 계산 가능한 결과를 돌려주는 기능 성공과 수치 품질 실패를 구별한다.

## 후속 구현이 사용할 공개 계약

`non_nt_runtime.cas12a_explain_handle(runtime)`가 채택 경로의 진입점이다.
`path="dual"`은 동일 조건 비교용이다. 기존 `explain_handle(..., derive=cas12a_derive)`는
계속 이중 입력이므로 #193의 동결 생산자와 과거 결과 의미를 바꾸지 않는다.
#200의 실제 non-NT predictor는 새 진입점을 사용해야 한다.

| 결과/식별 | 합성 경로 계약 |
| --- | --- |
| `handle.ig_inputs` / `ig_grad_fn` | `[34,4]` 입력 하나 / 동일 `F`의 전체 서열 gradient 하나 |
| `IGResult.attribution` | `[34]` signed 값, 입력 ACGT 채널 IG의 합. 파생 특징을 통한 기여 포함 |
| `IGResult.pred` / `baseline_pred` | 제한 전 `M(x,D(x))` / `M(0,0)`, raw 확률 |
| `IGResult.completeness_error` | `abs(sum(attribution)-(pred-baseline_pred))`, raw 확률 |
| `handle.input_totals` | 유일한 `Input_Seq` 값이 전체 signed 합. 독립 특징 기여의 비율이 아님 |
| `handle.ig_path` | `composite` |
| `handle.model_key` | 실제 적재 checkpoint의 SHA와 경로 프로토콜을 포함한 `ExplanationIdentity` |

화면에서는 예측값·기준값에 100을 곱해 %, 기여도·완결성 오차에 100을 곱해 %p로
변환한다. 내부 결과/캐시는 raw를 보존한다. 축/기술 참고는 **파생 특징 경로를 포함한
서열 기여도**라는 의미를 유지한다. 기존 그래프 구성을 쓰며 추가 품질 경고·진단 패널을
만들거나 실패를 이유로 그래프를 숨기지 않는다. 계산 예외/결과 없음/NaN·inf의 기존
오류 처리는 유지한다. 기본 IG는 32단계, SHAP은 64샘플·배경 4개다. 정밀 재계산은
#200에서 계산량을 알리는 선택 기능과 진행률·취소를 제공한다.

영벡터와 보간 중 soft 입력은 실제 DNA가 아니며, 독립 채널 섭동도 생물학적 변이를
뜻하지 않는다. PAM을 포함한 서열 채널도 보간된다. 기여도는 기준과 경로에 의존하고
생물학적 인과관계를 입증하지 않는다. 이 해석 한계는 기존 기술 참고에 반영할 계약이며
새 품질 경고 화면이 아니다. 이중 입력을 향후 별도 채택하면 위 비교표의 부분 라벨,
입력별 signed 합과 공동 완결성을 모두 제공해야 한다.

프로토콜은 다음처럼 분리한다.

- 채택: `cas12a-cnn-ig-composite-gc-dinuc-zero-midpoint-raw-v1`.
- 비교: `cas12a-cnn-ig-dual-zero-midpoint-raw-v1`.

계열 `non_nt`, checkpoint SHA, 헤드, 서열, 단계수/SHAP 설정·seed 외에 이 프로토콜과
runtime 버전·dtype·threads를 기존 메모리/디스크 캐시 키에 넣는다. 특징 정의·경로·기준·
단위가 바뀌면 프로토콜을 바꾼다. `handle.model_key`를 예측기의 계산 식별과 캐시 요청에도
사용해야 한다. 적재 출처가 입증되지 않은 수동 생성 runtime은 식별을 주장하지 않는다.
새 raw 함수는 SHAP의 혼합 one-hot에서도 특징을 함께 재계산하지만 이번 측정은 SHAP
품질 검증이 아니며 #193의 실패나 NT의 캐시 재생성을 대체하지 않는다.

## 검증 범위와 미검증 항목

회귀 범위는 새 특징/handle, 공통 runtime 적재, 기존 IG 결과 및 계산 식별 캐시 경계다.
실행 명령과 최종 회귀 수는 아래 구현 검증 기록에 남긴다. 저장소 전체 스위트는 실행하지 않는다.

이번 모델 실행은 개발 PC의 torch backend다. 설치된 TensorFlow가 Keras의 탐지 과정에서
import된 것을 기록했으며, TF-free 배포나 RAM 측정으로 주장하지 않는다. 프로덕션 화면
연결·선택 정밀 설정(#200), NT 단위(#201), 전체 트리 RAM(#203), 새 배포본과 깨끗한
오프라인 Windows(#204)는 이번 작업의 검증 범위 밖이다. 실제 모델의 held-out 성능과
생물학적 적합성도 이번 수치 실험으로 검증하지 않았다.

## 구현 검증 기록

기존 `test/test_non_nt_runtime.py` 기준선은 20 통과였다. 새 경계는 실패하는
테스트부터 특징 추출 및 채택 handle을 구현했고, 최종 관련 회귀는 다음과 같다.

```powershell
python -X utf8 -m pytest test/test_cas12a_ig.py test/test_non_nt_runtime.py test/test_explain.py test/test_explain_cache.py test/test_cache_identity.py -q -rs
```

**100 통과 / 0 실패 / 0 건너뜀**. 새 8개 테스트는 학습 특징 동등성, 해석 gradient,
실제 모델 IG 접합, soft 전방, 배치 gradient 독립성, 경로별 메모리/디스크 키 분리,
진행률·취소 및 잘못된 경로 거부를 확인한다. pytest 결과는 위 실제 모델 수치 검사
124/8과 별개이며 합산하지 않는다. `.gitattributes`로 동결 계약과 보존 자료의
줄바꿈 자동 변환을 막아 checkout 후에도 checksum을 유지한다.
