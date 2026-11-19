# NT XAI 단위와 검증 — #201

기존 NT 예측값과 표의 0~100% 제한은 유지한다. 설명은 제한 전 함수를 분해한다.
SpCas9 헤드마다 예측·기준값은 `(raw × std + mean) × 100`, 기여도·오차는
`raw_delta × std × 100`으로 표시한다. 평균은 기여도에 더하지 않는다.
Cas12a의 확인된 비율 출력은 ×100이다. 결과 객체의 원출력·기여도 배열은 그대로
보존하고 `units`가 화면의 변환을 제공한다. 통계가 없으면 원출력 단위로 표시하며
효율·백분위·%p 기여도라는 이름을 붙이지 않는다. Rollout은 방향 없는 attention
정보 의존도이며 효율 기여도가 아니다.

Cas12a는 `../cas12a/nt_data.py`의 백그라운드 차감 indel 백분율 ÷100 타깃과
`units.py`의 `PERCENT_PER_UNIT=100`을 확인했다. NT export가 기록한 checkpoint
SHA256 `eae04f4f…71969`가 현재 보존된 모델 및 측정 모델과 같은지 확인한다.
출력 근거의 파일 해시와 헤드 이름 대응은 [output-provenance.json](output-provenance.json)에 있다.

SpCas9 통계의 원본은 `../SpCas9/SpCas9_v3.ipynb`의 학습 셀과
`../xai_test/dataset/SpCas9_v3/train.csv`이다. 타깃을 float32로 읽어 100으로 나눈 뒤
훈련 데이터에서만 `nanmean`·`nanstd(ddof=0)`을 계산한다. 29,134행과 9개 헤드
순서를 확인했고 등록 통계와의 절대 차이는 모두 fraction 단위 `1e-7` 이하다.
원본 파일 SHA256·재계산 값·등록 값은 SpCas9 측정 보고서에 보존한다.

IG 기준은 서열 토큰의 word embedding을 0으로 만들고 특수 토큰을 고정한다.
임시 word-embedding hook을 일반 input-id forward에 적용하므로 ESM의
mask-dropout 보정·위치 임베딩·정규화가 일반 예측과 같다. `inputs_embeds`가
전처리를 건너뛰는 Transformers에서도 원함수를 보존한다. hook 설치·계산·해제는
예측·Rollout과 공유하는 잠금 안에서 실행하고 예외에도 해제한다. 상태는 handle
별로 보유한다. 토큰 기여도는 덮는 염기에 균등 배분한다.

설명 프로토콜은 `nt-affine-units-fixed-specials-esm-input-v2`로 바뀐다. 기존
메모리·디스크 키와 다르므로 과거 잘못된 단위/임베딩 경로 결과를 재사용하지 않는다.
SHAP 캐시는 원출력과 단위 메타데이터를 함께 보존한다. 검증에서 새로 계산한
고정 사례 캐시는 `evidence/shap-cache/`에 두며 기본 앱의 캐시 디렉터리에 자동
설치하지 않는다. 기존 네 legacy 산출물과 #205 증거는 변경하지 않는다.

기본값은 IG 32 / SHAP 64샘플·배경 4개다. 기존 설정 변경·재계산·진행률·취소를
유지한다. 사용자는 IG 256 또는 SHAP 128/8 등 계산량이 더 큰 설정을 선택할 수
있다. 설정별 품질은 별도로 판정하며 정밀 설정을 기본값이나 검증 보증으로 바꾸지
않는다. seed 0은 배경과 부분집합의 두 축을 고정한다.

유한한 결과는 완결성·가법성 기준 미달이어도 기존 그래프 구성을 표시한다. 별도
품질 배너·팝업을 추가하지 않고 예측 표도 유지한다. 예외·결과 없음·NaN/inf는
오류로 처리한다. 실행 시 잔차·기준·요청·계산 식별은 내부
`%LOCALAPPDATA%/dashboard-v3/logs/nt-xai.jsonl`에 회전 로그로 남는다.
화면 성공과 수치 품질 통과는 별도 결과다.

수치 조건은 실제 checkpoint 실행 전에 [contract.json](contract.json)에 동결했다.
기존 #181 고정 서열을 Cas마다 하나씩 사용하되 SpCas9의 PAM 호환성과 무관하게
9개 출력 모두를 수치 검증한다. 앱의 헤드 선택 규칙은 유지한다. 이 결과를 다른
서열 전체의 품질로 확대하지 않는다.

기존 NT Cas12a의 구간 색 표시는 교정하지 않는다. 실제 34bp 모델 윈도우에서
TTTV는 0–3번이지만, 기존 공용 `WINDOW_LAYOUT`은 PAM을 4–7번으로 표시한다.
이는 `CONTEXT.md`와 `docs/demo-sequences.md`에 이미 기록된 한계이며 좌표 교정은
별도 범위 합의가 필요한 항목이다. 기여도 배열은 실제 윈도우의 염기 순서와
일치하지만, 이번 그래프 표시 통과를 생물학적 구간 색·PAM별 기여도의 검증
완료로 확대하지 않는다. 전시회 표시 방침에 따라 새 경고 UI는 추가하지 않는다.

검증 명령은 다음과 같다. 각 Cas는 별도 Python 프로세스에서 실행한다.

```powershell
py -3.13 -X utf8 docs/nt-xai/verify.py SpCas9
py -3.13 -X utf8 docs/nt-xai/verify.py Cas12a
py -3.13 -X utf8 -m pytest tests/test_nt_process.py -q -p no:cacheprovider
```

`verify.py`는 계약/소스/checkpoint/환경 식별과 모든 판정·원시 배열을 보존하며
수치 실패가 있으면 종료코드 1을 반환한다. 첫 실패에서 실행을 중단하지 않는다.
SHAP 측정 harness는 헤드·seed 비교에서 동일한 실제 forward 출력 벡터를
메모하여 중복 추론을 줄인다. 같은 seed의 반복 검사는 이 결정적 출력 위에서의
추정기 재현성이다. 독립된 모델 추론 반복의 변동 측정으로 주장하지 않는다.
실제 계산 프로세스·PNG·캐시 재방문·진행률과 예측 동등성은 별도 process 시험에서
검증한다. 전체 저장소 테스트는 실행하지 않는다.

최종 측정 수치와 판정별 통과/실패/건너뜀 수는 [구현 보고서](../issue201-implementation.md)에 정리했다. 수치
실패를 숨기거나 사후에 계약의 허용 오차를 바꾸지 않는다. 패키징·깨끗한 Windows
배포 검증과 정식 RAM 비교는 #203/#204 범위다.
