# #203 독립 코드 리뷰

`implement`가 요구한 `code-review` 스킬에 따라 두 독립 에이전트가 Standards와
Spec을 병렬 검토했다. 고정점은 작업 시작 커밋
`a1d0b6151fb88979bef9d28f3651e560e73018ff`이며 다음 diff와 커밋을 사용했다.

```text
git diff a1d0b6151fb88979bef9d28f3651e560e73018ff...07094e8
06462ef feat(dashboard): measure full process-tree RAM and family-switch recovery (#203)
07094e8 fix(dashboard): preserve registered RAM evidence bytes in Git (#203)
```

기준은 `AGENTS.md`, `CLAUDE.md`, `CONTEXT.md`, 관련 호출·프로세스 경계 문서와
기존 `.gitattributes`의 바이트 보존 규칙이다. 명세는
[실제 조회한 #203](benchmarks/issue203-ram-20261119-run02/issue.json)이다.
스킬이 기대하는 `docs/agents/issue-tracker.md`는 없어 사용자에게 설정 필요를
알리고 보존한 GitHub 명세로 검토했다. 아래 Spec 수정은 검토자가 수정본까지 확인했다.

## Standards

문서화된 기준 위반과 수정이 필요한 코드 냄새를 발견하지 못했다.

`AGENTS.md`의 작업 위치·보존·선택 회귀 범위를 따른다. 보관된 observer 중복은
측정 당시 코드의 바이트 보존 목적이 명시되어 있어 중복 구현 문제로 분류하지 않았다.

독립 확인: Git 보관 해시 검사 **13 통과 / 0 실패 / 0 건너뜀**, 비교 diff의
whitespace 검사 통과. 무거운 모델이나 전체 테스트는 실행하지 않았다.

Standards: **0건**.

## Spec

Spec 축에서 1건 발견했고 수정본을 확인했다.

- **[P2, 수정 확인]** 결과 문서의 기존 NT SpCas9 Rollout private commit 피크가
  첫 호출의 4,542.2MiB로 기록되어 있었다. 반복 호출까지 포함한 실제 최댓값은
  **4,618.5MiB**다. 명세의 “단계별 비교…피크…보고”와 보고서의 반복 호출 최댓값
  집계 규칙에 맞게 수정했다. 전체 피크와 감소율에는 영향이 없다.

수정 후 지원 XAI 표의 16개 수치 칸을 보존된 단계별 자료와 대조해 모두 일치함을
확인했다. 보관 해시 검사도 13개 통과했다. 주 작업자는 시작·예측·미지원·전체 피크까지
포함한 기본 표 **27개 칸**을 별도로 대조해 모두 일치함을 확인했다.

추가 수용 기준 누락이나 범위 확장은 발견하지 못했다. 전체 프로세스 트리, 실제
XAI 계산과 캐시 적중의 구별, NT 미적재·종료·복귀 상태, 원시 자료와 배포 재사용
절차가 확인된다. 배포본·장기 반복 실행 미검증과 sampling 한계는 적절히 구분되어 있다.

Spec: **최초 발견 1건(P2), 수정 후 남은 발견 0건**.

Standards 0건; Spec 최초 1건(P2) 수정 완료, 두 축 모두 남은 발견 0건.
