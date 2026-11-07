# #146 Google Colab 실행 안내

이 실행 묶음은 **validation만** 사용합니다. 사용자가 Colab GPU에서 실행한
결과를 가져오면 원시 결과와 ledger를 모델 없이 재검산합니다. 코드 준비나 CPU
테스트 통과만으로 #146의 실제 GPU 검증이 완료되지는 않습니다.

## 전달받을 파일

- `Cas12a_NT_validation_colab.ipynb`: Colab에 업로드할 실행 노트북.
- `cas12a-validation-source.zip`: 로컬에서 검토한 코드와 계약 문서. 노트북에
  이 ZIP의 SHA256이 고정돼 있으므로 같은 빌드에서 나온 두 파일을 사용합니다.
- `bundle.json`: 전달 파일의 SHA256 기록. 결과를 전달할 때도 보관합니다.

저장소 루트에서 아래 명령으로 새 디렉터리에 묶음을 만들 수 있습니다.

```powershell
python -m xai_test.cas12a_NT_analysis.nt_colab build --output .scratch/issue146-colab
```

ZIP에는 분석 코드와 읽기 전용 입력 검증 모듈만 들어갑니다. GitHub에 push되지
않은 현재 코드도 포함하므로 원격 브랜치의 과거 코드를 대신 실행하지 않습니다.
모델·데이터·다른 프로젝트의 변경은 ZIP에 넣지 않습니다.

## Colab에서 실행

1. 노트북을 Colab에 업로드하고 GPU 런타임을 선택합니다.
2. 설치 셀을 실행합니다. 설치 후 이미 로드된 라이브러리와 버전이 다르다는
   메시지가 나오면 런타임을 다시 시작한 뒤 셀을 순서대로 실행합니다.
3. 코드 ZIP을 업로드합니다. 노트북이 전달 SHA256과 내부 파일 manifest를
   검증한 뒤 새 로컬 폴더에 풉니다. Drive 접근은 본인 계정에서 승인합니다.
4. 기존 source.xlsx, partition.json, selected_export, durable_export.json의
   Drive 경로를 설정합니다. 기존 모델과 partition은 그대로 사용합니다.
5. 이 연구의 **최초 study 생성**인지 **기존 study 재개**인지 명시합니다.
   기본은 재개입니다. 기존 study·receipt가 사라졌다는 이유로 최초 생성을
   선택하거나 다른 study root로 예산을 초기화하면 안 됩니다.
6. protocol 준비 셀에서 실제 validation ID, 수치 기준, 배치와 예산 규칙을
   확인합니다. 표시된 hash를 다음 셀에 붙여 넣고, 단일 실행자와 새 FP32 Test
   미관측을 확인합니다. 이 시점까지 모델 추론은 시작하지 않습니다.
7. 정확성 → warmup → production 비용 3개 batch 셀을 순서대로 실행합니다.
   수치 실패나 예산 초과가 나면 후속 셀을 계속 실행하지 않습니다.
8. forecast/freeze 셀을 실행합니다. 전수 1,500행 IG의 예약 비용이 먼저
   평가됩니다. 불가능할 때만 무작위 400행 이상과 사례 추가 최대 15행을
   계산할 범위를 동결합니다. 400행도 불가능하면 Test 실행을 중단합니다.
9. 반환 ZIP 셀을 실행하고 결과 ZIP 및 표시된 SHA256을 전달합니다.

첫 실행에서 이미 사용한 GPU 시간이 있다면 새 study 생성으로 0초 처리하지
말고 기존 ledger와 독립 보관 receipt를 먼저 가져와야 합니다. 노트북에는
Test 분석·재학습 셀이 없습니다.

## 수치 기준과 시간 해석

정확성 8행, 별도 warmup 2행, 비용 측정 32행씩 3개 그룹의 ID는 결과 관측 전에
정해집니다. 비용 그룹에서는 32행 실제 배치 전체를 singleton·fresh reload와
비교합니다. FP32/eager, autocast·TF32 비활성화를 기록합니다.

IG는 raw fraction 단위 completeness `1e-4 + 0.05*abs(delta)` 및 연속 signed
vector L1 수렴 `1e-4 + 0.05*L1(current)`를 사용합니다. `abs(delta)<=1e-6`이면
completeness는 절대 기준만 적용합니다. 200→400→800→1600 midpoint 단계를
시도하며 두 해상도 이상이 필요합니다. 이 값은 실행 전에 선언한 공학적
허용 기준이며, 실제 모델 품질은 저장된 결과에서 별도로 확인합니다.
rollout의 query L1 0.01은 퇴화 진단 기준이며 퇴화 자체는 실패가 아닙니다.

예산 상한은 연구 전체 누적 21,600초입니다. 정확성·각 비용 단계는 1,800초,
warmup은 900초를 예약하며, 성공·실패 시에도 예약 시간을 환불하지 않습니다.
최초 loading부터 parity, 계산, 저장·재로드, receipt 저장을 포함합니다.
forecast는 세 비용 그룹의 최대 실측 rate, IG 최대 retry ladder, 여유 배수
1.5와 고정 600초, 향후 8회 세션의 loading 및 분 단위 예약 올림을 포함합니다.
실제 Test 단계의 예산도 같은 ledger에서 통제해야 합니다.
마지막 외부 receipt 쓰기 자체의 지연은 해당 receipt의 elapsed 값에 들어가지
않습니다. 예약 상한 검사는 이 쓰기까지 포함하며, forecast에서는 이 미측정
꼬리 비용을 고정 여유에 포함합니다. 다른 receipt 저장은 실측 비용에 포함됩니다.

## 중단 후 재개

Drive의 study 폴더와 별도의 evidence 폴더를 모두 보존합니다. evidence에는
매 ledger 변경 직후의 immutable receipt가 자동 저장됩니다. 정상 종료된
단계는 재실행 시 그대로 재사용합니다. 동시에 여러 런타임을 실행하지 마세요.
이 경로는 기존 sequential-writer 계약을 유지하며 분산 lock을 제공하지 않습니다.

중간에 런타임이 끊기면 이전 런타임/프로세스가 실제로 종료됐는지 확인하고,
노트북의 수동 정산 셀에서 종료 증거·종료 UTC·보수적 시간 상한을 제출합니다.
열린 작업은 자동 만료시키지 않습니다. 정산 뒤 미완료 단계의 `attempt`만
명시적으로 올려 새 전체 비용 측정을 수행합니다. 이전 결과와 비용은 남습니다.
수치 실패가 저장된 protocol은 자동 재시도하지 않으며 기준을 사후 수정해
같은 결과를 성공으로 바꾸지 않습니다.

receipt가 없거나 저장 오류·불확실한 시간이 있으면 자동 복구하지 않습니다.
반환 가능한 원시 진단 자료를 보존하고 추가 실행을 멈춥니다. filesystem
readback은 Google Drive 서버 복제 완료 자체의 증명은 아닙니다.

## 결과를 받은 뒤 로컬 검산

반환 ZIP은 study의 전체 ledger·방법별 NPZ/JSON·코드 snapshot, 독립 receipt,
동결 protocol, measurements, coverage 및 freeze receipt를 포함합니다.
분할·checkpoint를 변경하거나 모델을 다시 실행하지 않고 검산합니다.

```powershell
python -m xai_test.cas12a_NT_analysis.nt_colab audit --archive returned.zip --sha256 <전달받은-SHA256> --output .scratch/issue146-returned
```

실행 실패나 자원 부족도 결과로 가져옵니다. `feasible: false`, 수치 실패,
미완료 coverage 또는 검증 불가능한 history를 #146 성공으로 처리하지 않습니다.

실행 명세: [Colab 계약](planning/issue146-colab-contract.md).
독립 설계 검토: [리뷰 기록](reports/issue146-design-review.md).
환경 참고: [공식 Colab FAQ](https://research.google.com/colaboratory/faq.html),
[공식 PyTorch synchronize 문서](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html).
