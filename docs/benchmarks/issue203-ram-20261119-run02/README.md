# #203 기본 소스 비교

[결과·환경·명령·한계](../../issue203-implementation.md)와
[재사용 절차](../ram-procedure.md)를 따른다. 기존 NT 창 검사 6개, v3 창 검사 16개가
모두 통과했다. `contract.json`은 실행 전 등록, `issue.json`은 #203 요구사항 조회다.
각 실행 폴더에는 압축 원시 sample, 실제 단계, runtime 적재 상태, 창 검사, 프로세스
목록과 종료 상태, argv·로그가 있다. `observer/`는 이 실행 당시 측정 코드다.

`summary.json`, `comparison.json`, `ram-timeline.png`는 원시 자료의 파생 산출물이다.
추가 SHAP 캐시 미적중 실험은 별도 `../issue203-ram-20261119-cold01/`에 보존한다.
