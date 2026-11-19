# #197 implementation review

Fixed starting point: `99013e011d0e77581db4228fea567f5e7583fc0c`.
Two independent reviewers applied the code-review skill's Standards and Spec
axes to the working changes. They did not edit or run model tests.

The workspace also contains concurrent #196 changes. The #197 review covered
`non_nt_cas12a_predict.py`, the Cas12a entries in `predict.py`, scanner window
configuration, the shared renderer's `_scan` hook and non-NT override, the #200
message, and the #197 tests, reference generator, evidence and implementation
record. Other #196 changes were excluded. `git diff` at the starting point and
the listed new files supplied the scoped review input; #197 is the spec.

## Standards

발견사항 **0건**. #196 변경은 제외하고 #197의 예측기, scanner 설정 확장,
렌더러 scan 경계, 테스트, reference 생성기와 구현 기록을 검토했다.
문서화된 규칙 위반이나 조치가 필요한 코드 냄새는 발견하지 못했다.
모델별 window 차이를 `CasConfig` 설정으로 전달하고 모든 관련 렌더러 경로에서
같은 scan 경계를 사용한다. 기존 NT 기본값을 보존하면서 non-NT 예측기는
PAM·guide 위치까지 확인해 잘못 정렬된 입력을 거부한다.
테스트는 공개 경계와 별도 TensorFlow reference를 사용한다.

## Spec

발견사항 **0건**. “34bp 윈도우와 23bp protospacer” 요구에 맞춰 non-NT Cas12a만
4bp upstream 문맥을 사용한다. 양 가닥·입력 경계 테스트와 PAM·guide 정합성
검사로 잘못 정렬된 입력을 막는다. “특징 순서·값, 단일 출력과 단위”는
보관된 학습 특징 코드와 별도 TensorFlow 프로세스의 실제 checkpoint 기준값
7건 및 두 입력의 바이트 해시로 확인한다.
Best Candidates, 상세, XAI 선택과 CSV는 같은 계열별 탐색·예측 행을 사용한다.
실제 자식 프로세스 검증은 지연 적재, NT·SpCas9 자원 비접근, CSV 재사용과
단일 `Cas12a` 헤드를 확인한다. 설명 계산의 #200 안내도 실제 호출 경로에 반영됐다.
범위 밖 재학습·분할 변경은 없다.

Standards: 0 findings, no outstanding issue. Spec: 0 findings, no outstanding issue.
