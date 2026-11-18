# #210 IG·SHAP 정밀 후보 비교 — 실행 경로와 회귀

부모: #205. 이 티켓은 **실행 가능한 비교 경로와 회귀**를 완성한다. 공식 실측은
#213이 #212의 고정 코드·계약·검증 묶음으로 수행한다. 여기의 fixture 통과는 하네스
회귀이지 모델 측정이 아니다.

## 등록된 것 (측정 전 고정)

- `protocol-210.json`: `protocol-206.json`을 그대로 담고(sha256을 `extends`에 기록)
  다음만 추가한다. `contract.json`과 protocol-206은 바꾸지 않는다.
  - `investigation_command`: `["{python}", "{root}/docs/runtime/candidate_investigation.py", "{work}"]`.
    `{root}`는 워크트리 루트다. 생산자 스크립트는 `docs/runtime/*.py`로 init 때 함께
    고정된다.
  - `candidate_protocol`: 사례(S1–S4, C1–C4)·헤드(SpCas9 NG)·경로(Cas12a 현재 이중 입력)·
    설정·seed·실행 순서·warm-up·재사용 규칙·예산 90분(5400 s)·선정/확인 규칙.
  - `shap_confirmation_*`의 `planned_by_candidate`: 확인 단계 신규 호출 수는 기본 28회다
    (사례당 3×3 중 1차에서 측정한 3개를 뺀 6회와 fresh (0,0) 1회). 두 축이 겹치는
    `ns512/bg4`가 후보면 같은 묶음에서 동일한 측정 5개를 재사용하므로 20회다.
    SHAP 총합은 최대 184 + 28 + 28 = 240회다.
- `candidate_policy.py`: 계획(순서), 1차 적합성, 선정, 확인, 예산 검증 규칙. 생산자와
  조립기가 **같은 함수**를 쓴다.
- `candidate_investigation.py`: 생산자. 원시 배열과 호출 기록을 남긴다.

### 실행 순서

1. 모델마다 적재한 뒤 시간 재지 않는 warm-up(전방 1회, gradient 1회). 측정 호출이 아니다.
2. IG: SpCas9 → Cas12a, 사례 순, 단계 32/64/128/256.
3. SHAP 1차: SpCas9 → Cas12a, 사례 순. 샘플 축(배경 4·배경 seed 0, 샘플 64/128/256/512 ×
   부분집합 seed 0/1/2), 이어서 배경 축(샘플 512·부분집합 seed 0, 배경 4/8/16/32 ×
   배경 seed 0/1/2). 교집합 `ns512/bg4/background0/subset0`은 한 번만 잰다. 사례당 23회.
4. 두 모델의 1차가 끝난 뒤 선정, 확인은 SpCas9 → Cas12a. 사례 순으로
   (배경 seed, 부분집합 seed)를 행 우선으로 돌며 동일한 1차 측정은 건너뛰고(비트 동일하게
   복사), 마지막에 fresh (0,0)을 1회 부른다.

모든 측정은 `xai.explain`을 직접 부르는 fresh 호출이다. XAI 메모리 캐시를 거치지 않고,
SHAP은 `use_disk_cache=False`로 사전계산 SHAP 캐시도 조회하지 않는다.

### 예산

시계는 warm-up 뒤 첫 호출 직전에 시작한다. **완료된 호출 사이**에서만 확인한다. 실행 중이던
호출은 끝까지 마치고 기록한다. 경과가 5400 s 이상이면 새 호출을 시작하지 않고
`termination: planned_budget`으로 남긴다. 미완료 행은 `time_limited`와 완료·미완료 수를
가지며, `candidate_summary.unstarted_calls`에 시작하지 않은 호출이 남는다. 예산을 자동으로
늘리지 않는다. 필수 baseline·float32 진단·RAM·독립 복구·조건부 float64는 이 예산에 들지 않는다.

### 판정 (경계값은 `contract.json`에서 읽고, 계약이 문구로 적은 0.25·0.05·0.1 계수는 #193 `verify`와 같게 적용한다)

| 행 | 상태 |
| --- | --- |
| `ig_*` | 4사례 모두 유한·완결성 오차 ≤ SpCas9 5e-3 / Cas12a 1e-3인 단계 중 관측 최대 wall time이 가장 짧은 것(동률이면 단계수가 작은 것) → `pass` + `completeness_improvement_candidate`. 없으면 `no_eligible_candidate`. |
| `ig_*`.`resolution` | 32/64/128 대 256, 사례별 상관 ≥0.99·최대차 ≤0.05×max\|attr256\|·강한 위치 부호 일치. 별도 보고이며 상태를 바꾸지 않는다. 256 자기 비교는 없다. |
| `shap_screen_*` | 4사례 모두 모든 실행 유한·가법성 기준 충족, **측정한 축마다** 3쌍 상관 ≥0.9·최대차 ≤0.25×max\|phi(0,0)\|. 미측정 축은 `unverified_axes`. 최대 fresh wall time → sample×background → sample → background 순으로 하나를 고른다(`first_stage_candidate`). |
| `shap_confirmation_*` | 4사례 모두 9조합과 fresh 유한·가법성, 36쌍 전부 상관 ≥0.9·최대차 ≤0.25×max\|phi(0,0)\|, fresh 최대차 ≤1e-9 → `pass` + `seed_stability_confirmed_candidate`. 실패 시 `no_eligible_candidate`이고 다른 후보를 시험하지 않는다. 1차 후보가 없으면 `not_applicable`. |

상관을 정의할 수 없으면(상수·비유한) `pearson_r: null`이고 통과하지 않는다. 누락된
측정은 통과할 수 없다.

조립기(`assemble_verification.py`)는 protocol에 `candidate_protocol.selection`이 있으면
원시 배열과 호출 기록에서 후보 행을 **다시 유도**한다. 생산자 행·요약이 유도 결과와 다르거나,
호출 순서가 등록 계획과 다르거나(다른 후보 추가 탐색 포함), 예산 뒤에 시작한 호출이 있거나,
예산 소진 없이 멈췄거나, 재사용 배열이 1차 측정과 비트 동일하지 않으면 `fail`이다.

Q17 연계: 후보 행이 모두 완료돼도 #211 float32/float64 생산자가 등록되지 않은 묶음에서는
그 행이 `unverified`이므로 종합은 `incomplete`/1이다. 후보 없음·확인 실패는
`completed_with_findings`/0, 예산 소진은 `incomplete`/1, 필수 실패는 언제나 `fail`/1이다.
예기치 않은 예외는 생산자가 결과를 쓰지 않고 2로 끝나므로 단계가 중단되고 새 묶음과
전체 재측정이 필요하다.

## 이름과 범위

결과 이름은 **완결성 개선 후보**(IG)와 **seed 안정성 확인 후보**(SHAP)로만 쓴다. 고정 8사례·
SpCas9 NG 헤드·Cas12a 현재 이중 입력 경로에 한정한다. 기본값(IG 32, SHAP 64/4)은 그대로이고
자동 승격은 없다. IG 512를 추가하지 않으며 IG 256이나 SHAP 512/32를 정답으로 쓰지 않는다.
#193의 default-versus-finer와 gradient 실패는 그대로 남는다. 전체 헤드·경로 품질 승인은
#199/#200, NT는 #201 소관이다.

## 공식 실행 (#213)

```powershell
python docs/runtime/verify_contract.py init --work $bundle --run-id <id> --protocol docs/runtime/protocol-210.json
# ... #206 필수 단계 (reference / verify / diagnose / ram / offline) ...
python docs/runtime/verify_contract.py investigation --work $bundle
python docs/runtime/assemble_verification.py --work $bundle --output <new-summary.json>
```

#211이 float32/float64 생산자를 넣으면 같은 `investigation` 단계에서 두 생산자의 행을
함께 써야 한다. 등록 명령은 하나이므로 그때 합성 생산자를 새 추가 protocol에 등록한다.

## 실행 결과 (2026-11-18, 하네스 회귀)

선택 근거: 새 생산자·선정 규칙(`test_candidate_selection.py`), 바꾼 조립기·Q17 정책·
investigation 단계(`test_runtime_verification.py`), 같은 묶음 경계를 쓰는 독립 복구
(`test_offline_recovery.py`).

```
python -m pytest test/test_candidate_selection.py test/test_runtime_verification.py   test/test_offline_recovery.py -q -p no:cacheprovider -rs
→ 135 passed, 0 failed, 0 skipped (368.8 s)
```

포함: 전 사례·전 seed 적합성, 한 사례 실패의 차단, 미정의 상관, 동률 처리(IG 단계수,
SHAP sample×background), 확인 실패 후 미탐색(교차 seed·fresh), 예산 경계 뒤 미시작과
완료/미완료 수, 계획 종료(재실행 거부) 대 비정상 중단(단계 aborted·재개 거부), 원시
증거와 다른 행·순서·재사용 배열의 거부, 필수 실패 우선, Q17 pass/incomplete/
completed_with_findings, 그리고 실제 두 모델로 등록 명령을 1초 예산 사본으로 돌린 계획
종료 스모크.

변이 확인: 예산 검사를 끄면 예산 테스트가, 조립기 재유도를 끄면 증거 거부 테스트 6개가 실패한다.

미검증: 공식 전체 실측(#213). 실제 모델로는 1초 예산 스모크만 실행했고 후보 선정 결과는
아직 없다. #211 행은 이 생산자에서 `unverified`다.
