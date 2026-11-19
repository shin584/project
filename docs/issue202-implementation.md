# #202 — 배포 non-NT 모델의 test 성능과 출처 인계

`dashboard_v3`, 기준 커밋 `1a297cdbdff2cc1b9214811b4a9ac8b8d4739b81`에서 작업했다.
[평가 계약](model-evaluation/contract.json)을 `7d1fc7d`로 추론 전에 커밋했고,
평가 소스·모델·원본 데이터·분할을 별도 실행 폴더에 복사해 해시를 고정했다.
기존 `xai_dashboard_desktop` / `dashboard_core_v2`는 변경하지 않았다.

[상세 인계 문서](model-evaluation/README.md)에 헤드별 지표, 단위·결측·clip 정책,
선택과 test의 구분, checksum 연결, 실제 파일 보관 여부와 재실행 명령이 있다.
[보고서](model-evaluation/evidence/report.json),
[실행 기록](model-evaluation/evidence/execution.json),
[재현 archive와 SHA256](model-evaluation/evidence/index.json)을 함께 보존했다.

## 결과

- SpCas9: 동일 원 checkpoint SHA의 기존 5,186행 평가와 학습·분할·export를 연결했다.
  보존된 CSV에서 9헤드의 Pearson/Spearman/MAE/MSE를 재계산해 기존 JSON과 일치했다.
  새 전체 test 추론에서 원본→export 출력 차이는 0, 원본→배포 torch는 최대 `2.38e-7`이다.
- Cas12a: 선택용 validation 1,500행과 별도의 동결된 test 1,500행을 구분했다.
  새 test에서 배포 모델의 Pearson `0.761203859`, Spearman `0.748909696`,
  MAE `0.154544342`, MSE `0.043484108`이다. 단위는 fraction이며 MAE는 약 15.45%p다.
  출력 clip 전후 지표가 같고, 원본→배포 최대 절대 출력 차이는 `1.49e-7`이다.
- 원본 인코더/배포 인코더의 전체 test 입력 배열은 두 모델 모두 정확히 일치했다.
  동결 기준의 헤드×비교 28건이 통과했다. 각 건은 모든 행의 출력 parity와
  raw/clipped 각각 4개 지표 일관성을 함께 판정한다. **28 통과 / 0 실패 / 0 건너뜀**이다.

## 선택한 검증 범위와 실제 명령

평가 도구의 지표·결측·clip·판정·해시 경계를 검증하고, 실제로 평가한 두 배포 예측기의
기존 공개 경계(입력·헤드 순서·원출력 단위·checkpoint 식별)를 회귀 범위로 선택했다.
앱 동작 변경은 없다. 저장소 전체 스위트는 AGENTS.md의 #191 범위 합의에 따라 실행하지 않았다.

개발 중:

```powershell
py -3.13 -X utf8 -m pytest test/test_model_evaluation.py -q -p no:cacheprovider
# 10 passed / 0 failed / 0 skipped, 2.11s
py -3.13 -X utf8 -m pytest test/test_non_nt_spcas9.py test/test_non_nt_cas12a.py -q -p no:cacheprovider
# 29 passed / 0 failed / 0 skipped, 27.63s
```

실제 모델 실행:

```powershell
py -3.13 -X utf8 docs/model-evaluation/evaluate.py run --work .packaging/issue202-run01 --cas12a-source ../cas12a/dataset/kim_2018_cas12a_dataset.xlsx --cas12a-partition ../cas12a/models/NT_cas12a/result/20261101-nt500m-full/partition.json
# prepare/reference/deployed/report/audit: exit 0, 28 pass / 0 fail / 0 skip
# 각 단계 11.84 / 60.43 / 28.47 / 3.27 / 3.08s
```

archive를 SHA256 검증 후 별도 디렉터리에 풀고 복원된 소스로 `audit`를 실행했다.
전체 배열에서 보고서의 지표·판정 재계산은 **28 통과 / 0 실패 / 0 건너뜀**, exit 0이다.
복원된 checkpoint·데이터·소스로 재추론한 실행도 **28 통과 / 0 실패 / 0 건너뜀**,
모든 단계 exit 0이다. 해당 명령은 [상세 문서](model-evaluation/README.md)의 마지막
명령이며 기존 원본 폴더를 참조하지 않는다. [재추론 기록](model-evaluation/evidence/reinference.json)에
새 실행 명령·시간과 별도 등록·보고서의 경로를 보존했다. 전체 test 비교 결과와 원본
지표가 같고 원시 NPZ 및 메타데이터의 모든 SHA256도 첫 archive와 일치한다.

마무리 선택 회귀 검사와 정적 검사:

```powershell
py -3.13 -X utf8 -m pytest test/test_model_evaluation.py test/test_non_nt_spcas9.py test/test_non_nt_cas12a.py -q -p no:cacheprovider
# 39 passed / 0 failed / 0 skipped, 17.84s
py -3.13 -X utf8 -m pyright docs/model-evaluation/evaluate.py test/test_model_evaluation.py
# 0 errors / 0 warnings / 0 informations
py -3.13 -X utf8 -m ruff check docs/model-evaluation/evaluate.py test/test_model_evaluation.py
# All checks passed
```

초기 정적 검사에서 타입 추론·원본 snapshot 모듈 import 경고를 발견해 동적 import와
명시적 외부 모델 타입 경계를 정리했다. 실제 test 실행 전에 해결했으며 기준·모델은
변경하지 않았다. TensorFlow의 입력 구조·폐기 예정 API 알림은 로그에 보존했다.

## 남은 한계

Cas12a 학습 소스의 사후 복원 한계는 보존했다. SpCas9의 기존 dirty 평가 소스 snapshot
전체는 채택 vault에 없어 역사적 실행 소스 전체의 복원을 주장하지 않는다. 반면 이번
평가의 입력·소스·checkpoint·예측은 archive 안에 모두 실제 보관했다. test 결과로
재학습·재선택·분할 변경을 하지 않았다. 성능 기준을 사후에 바꾸지 않았다.

이 결과는 XAI 수치 품질의 실패를 해결하지 않으며, 깨끗한 Windows 오프라인 배포,
패키징 및 RAM 검증은 #203/#204 범위로 남는다.

## Standards

`code-review` 스킬의 두 독립 에이전트가 작업 시작점부터의 변경을 검토했다.
검토 명령은 `git diff 1a297cdbdff2cc1b9214811b4a9ac8b8d4739b81...c444835`이며,
커밋 목록은 `7d1fc7d`, `c444835`이다. tracker 설정 문서가 없어 #202의 실제
GitHub 본문을 직접 제공했다. 이 마지막 문서 보완은 리뷰 결과를 기록한다.

규칙 준수 지적은 **0건**이다. AGENTS.md의 브랜치·기존 화면 보존·선택 회귀 범위·
실제 모델 검증·명령과 결과 보고를 준수했다. 원출력/헤드/계열 용어와 역사적 평가·
새 test 실행의 구분도 적절하며 조치가 필요한 baseline smell은 발견하지 못했다.
리뷰어는 archive SHA256, 등록 파일 57개, archive 안의 평가 소스·계약과 최종
파일의 바이트 일치, 원 보고서 보존을 독립적으로 확인했다.

## Spec

요구사항 지적은 **0건**이다. 리뷰어는 동결 계약의 checkpoint·membership·전처리·
단위·결측·raw/clipped 지표·parity 기준, SpCas9 5,186행·9헤드와 별도 Cas12a
1,500행 평가, 기존 평가·선택 validation·새 추론의 구분을 확인했다.

archive SHA256과 등록 파일 57개, raw artifact 해시, 전체 예측 배열의 크기,
Cas12a test membership와 partition 비중복을 독립 검증했다. 보존된 배열에서
Cas12a의 네 지표를 별도로 재계산해 보고서와 일치함을 확인했다. SpCas9의
dirty source와 Cas12a 사후 학습 출처 한계가 명시되어 있으며 재학습·재선택·
재분할·NT 비교는 발견하지 못했다.

최종 리뷰 집계는 **Standards 0건 / Spec 0건**이다. 문서 경로, 공개한 archive와
JSON의 모든 digest, ZIP CRC, 측정 소스와 현재 소스 일치 및 `git diff --check`도
통과했다. 문서만 보완한 마지막 커밋에는 모델·테스트를 반복 실행하지 않았다.
