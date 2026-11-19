# #204 Windows 배포 실행

`contract.json`은 복사본과 보존 NT 배포본의 실행 전 식별과 고정 작업을
기록한다. 해당 관찰 도구의 소스는 `d95be0b`다. 각 폴더의 `launch.json`,
`window.json`, `sampler.json`은 명령·창 검사·프로세스 트리와 종료 결과를
보존한다. `samples.jsonl.gz`는 동시 working set과 private commit의 원시 자료,
`phases.jsonl`은 단계 경계, `events.json`은 브리지 호출과 이벤트 기록이다.

`legacy/`의 첫 실행은 창 검사를 통과했고 전체 트리 샘플을 저장했다. 브리지가
준비되기 전에 계측을 붙여 예측과 기법별 호출 경계가 비었다. 단계 비교에는
브리지를 기다리도록 보완하고 별도 등록한 [NT 재측정](../issue204-win-legacy-repeat/)을
사용한다.

개발 PC의 WebView2와 Hugging Face 경로에는 오프라인 조건을 적용했다.
깨끗한 게스트의 네트워크 차단 실행은 `sandbox-attempt.json`에 미완료로 기록한다.
[배포 인계와 검증 결과](../../issue204-implementation.md)
