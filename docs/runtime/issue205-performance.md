# #205 정밀 설정의 탐색적 성능 측정

개발 PC에서 실제 non-NT 모델 2종의 대표 서열 한 개·선택 헤드 한 개를 측정했다.
수치 품질 게이트의 재검증이 아니며, 기존 실패나 동결 계약을 바꾸지 않는다.
실행 ID는 `20261118T092845Z`, 소스 HEAD는 `c7df95b`다.

## 측정 조건

- Windows 개발 환경, Python 3.13, Keras 3.14.1/torch 2.12.0+cpu,
  `torch_num_threads=4`, CPU 실행.
- SpCas9은 계약의 S1·헤드 index 1, Cas12a는 C1·헤드 index 0.
- 모델별 별도 프로세스, 모델 적재와 warmup 후 순차 계산. IG 청크는 8,
  SHAP은 seed 0·디스크 캐시 비활성. 라이브러리 import와 모델 적재 시간은 표에서 제외.
- 시간은 완료한 3회 실행의 중앙값이다. CPU 시간은 프로세스의 user+system
  CPU 시간을 합친 값으로 wall time과 다르다.
- RSS는 약 10ms 간격의 표본과 끝점으로 관측했다. 설정을 같은 프로세스에서
  순서대로 실행해 allocator·라이브러리 메모리가 남을 수 있다. 설정별 새 프로세스
  RAM 비교가 아니며 순간 피크를 놓칠 수 있다.
- 개발 환경의 TensorFlow probe가 Cas12a에서는 실제 TensorFlow를 import했다.
  이 메모리 수치를 TF-free staging이나 최종 배포본의 수치로 취급하지 않는다.

## 주요 결과

| 설정 | SpCas9 wall / CPU 초 | Cas12a wall / CPU 초 |
| --- | ---: | ---: |
| IG 32 | 0.467 / 1.688 | 0.933 / 3.109 |
| IG 256 | 2.764 / 9.844 | 4.960 / 18.000 |
| SHAP 64/4 | 0.524 / 1.906 | 1.141 / 3.453 |
| SHAP 512/4 | 2.870 / 9.609 | 4.658 / 15.219 |
| SHAP 512/32 | 19.563 / 69.594 | 40.057 / 119.938 |

IG 32→256의 wall time은 약 5.9배·5.3배, SHAP 64/4→512/32는 약 37.4배·35.1배였다.
중간 설정 등 모든 완료 측정은 raw JSON에 있다. SHAP 64/32는 SpCas9 3회 중앙값
2.697초, Cas12a는 완료한 1회 4.347초다. 후자를 3회 중앙값으로 표시하지 않는다.

| 관측된 단계별 RSS 피크 (MiB) | SpCas9 | Cas12a |
| --- | ---: | ---: |
| IG 32 | 469.3 | 619.8 |
| IG 256 | 493.6 | 620.8 |
| SHAP 64/4 | 505.6 | 640.6 |
| SHAP 512/32 | 530.7 | 769.3 |

이 표는 측정 순서에서 관측한 프로세스 전체 RSS다. 숫자의 차이를 해당 설정만의
추가 메모리라고 단정하지 않는다. UI·WebView2·NT가 포함된 전체 트리 RAM도 아니다.

IG는 청크를 유지하면 주요 gradient 배치 크기도 유지된다. SHAP은 모델 전방
배치를 64행으로 유지하지만 전체 합성 입력과 Cas12a 특징을 먼저 만들기 때문에
샘플수·배경수 증가에 따라 입력 버퍼가 커진다. 계산량의 8배·64배 증가를
프로세스 RAM의 같은 배수 증가로 해석하지 않는다.

## 실행과 완료 범위

실제 실행 명령:

```powershell
& 'C:/Users/cmino/AppData/Local/Programs/Python/Python313/python.exe' -X utf8 '.scratch/issue205-performance/20261118T092845Z/measure.py'
```

측정 도중 Cas12a 시작 전에 설정 순서를 주요 비교 우선으로 바꿨다.
SpCas9에서 사용한 원래 스크립트는 `measure-spcas9-original.py`로 보존했고,
각 JSON의 script SHA와 configuration order로 어느 버전을 썼는지 식별한다.

- 계획한 성능 호출 72회 중 52회 완료, 1회가 모델별 180초 제한으로 중단,
  19회는 미시작. 수치 품질 테스트의 통과·실패 수가 아니다.
- SpCas9: 12설정×3회 완료, 자식 종료코드 0. 시작 포함 총 131.8초.
- Cas12a: 위 주요 표의 5설정×3회와 SHAP 64/32 1회 완료.
  SHAP 64/32 두 번째 실행 중 180초 제한으로 종료, 자식 종료코드 1.
  추가 6설정과 해당 설정의 나머지 반복은 완료하지 않았다.
- 최상위 orchestration은 결과를 기록하고 종료코드 0을 반환한다. 개별 모델의
  성공 여부는 `orchestration.json`을 확인한다. Cas12a raw의 `status=running`은
  마지막 저장 시점 상태이며 최종 timeout 판정은 orchestration 쪽이 기준이다.
- 제품 코드·계약·모델 bytes는 변경하지 않았다. 새로운 gradient/IG/SHAP 품질
  판정, NT 성능, TF-free 복구, UI 전체 트리 성능은 검증하지 않았다.

## 원시 자료

[보존 디렉터리](../benchmarks/issue205-performance-20261118/)에 스크립트 2개,
모델별 JSON과 stdout/stderr, orchestration 결과 및 checksum manifest를 보관한다.
각 모델 JSON은 코드·계약·모델 hash, 환경, 모든 완료 반복의 시간·RSS를 담는다.

실제 실행은 위 `.scratch` 경로에서 했으며 보존 사본은 별도로 복사한 것이다.
재실행하면 JSON과 로그를 다시 쓰므로 보존 디렉터리를 직접 실행 위치로 쓰지 않는다.
이 소규모 측정의 차이만으로 설정을 품질 승인하거나 모든 서열의 예상시간을
확정하지 않는다.
