## Problem Statement

**현재 상태: 설계 확정·구현 준비 완료. Q1~Q17과 테스트 경계·필수 검증 범위에 대한 최종 공동 이해를 사용자와 확인했다. 구현은 시작하지 않았다.**

Part of #191. #193에서 채택한 두 non-NT 모델의 검증에는 필수 실패·검사 누락을 종합 판정에서 놓치는 경로가 있다. 실행 출처와 독립 오프라인 복구 근거가 부족하고, SHAP 동시 호출의 RNG가 충돌한다. 사용자는 현재 설명의 신뢰성과 더 오래 계산했을 때의 개선·비용을 정확히 판단하기 어렵다.

Keras 3.14.1 + torch backend 채택이나 현재 전방 parity 측정이 잘못됐다고 확인된 것은 아니다. 검증 도구를 보수하고 실제 실패와 근거 범위를 보존해 후속 작업에 인계한다. 이전 NT에도 IG 오차·SHAP 불안정 경고가 있어 모델 구조 차이만으로 이번 현상을 단정하지 않는다.

## Solution

모든 필수 검사를 판정하고 실행·계약·코드·모델을 연결한 런타임 검증 묶음으로 두 실제 모델을 재검증한다. 원본 작업 폴더 없이 복구한 자산으로 최소 실행한다. 기본값과 기존 기준을 유지하면서 유한한 추가 실험으로 정밀 재시도 후보의 오차·변동·비용을 비교하고 실패·미검증·원인 미해결을 명시한다.

이 문서는 Q1~Q17과 최종 검토를 반영한 확정 명세다. 사용자는 Q17 승인에 이어 최종 설계·검증 범위 확인에도 "그렇게해"라고 답했다. 제품 코드 보수·새 수치 품질 검증·공식 복구 실행은 별도 구현 작업이며 아직 시작하지 않았다. 구현 준비 완료는 수치 검증 통과나 이슈 해결 완료를 뜻하지 않는다.

## User Stories

1. As a verification maintainer, I want every required check accounted for, so that missing evidence cannot produce a pass.
2. As a reviewer, I want offline failure reflected in overall status and exit code, so that automation detects failed recovery.
3. As a reviewer, I want both parity criteria applied to every layer, so that a summary cannot hide a violation.
4. As a runtime maintainer, I want all six registered versions checked, so that environment differences are visible.
5. As an explanation reviewer, I want finiteness checked across all seeds and settings, so that invalid outputs cannot be omitted.
6. As a verification maintainer, I want frozen criteria preserved, so that results cannot redefine success.
7. As a reviewer, I want each result linked to its run, code, contract, models and environment, so that incompatible evidence cannot be combined.
8. As a future maintainer, I want historical evidence retained, so that later reports cannot rewrite its provenance.
9. As an operator, I want interrupted measurements restarted in a new bundle, so that partial runs are not mistaken for complete evidence.
10. As a reviewer, I want summaries regenerated from immutable raw results, so that report assembly needs no new measurements.
11. As an operator, I want complete commands and explicit outputs, so that evidence can be regenerated from an empty directory.
12. As a build maintainer, I want frozen dependencies declared, so that personal installations do not substitute for build inputs.
13. As a recovery operator, I want independently restored code and assets, so that the original worktree is unnecessary.
14. As a reviewer, I want Python and OS prerequisites stated, so that local staging is not mistaken for clean-machine deployment.
15. As a SHAP caller, I want participating concurrent calls to reproduce serial results, so that request timing does not change explanations.
16. As a SHAP caller, I want RNG restoration after success, exception and cancellation, so that calls do not leave changed global state.
17. As a cache consumer, I want complete computation identity matched, so that stale explanations are rejected.
18. As an NT maintainer, I want legacy cache rejection coordinated with #201, so that regeneration uses the corrected protocol.
19. As a reviewer, I want cache-hit and fresh-result equality checked, so that reuse has numerical evidence.
20. As an explanation user, I want initial settings preserved, so that default latency does not increase silently.
21. As a reviewer, I want IG results separated by step count, so that finer success does not conceal default failures.
22. As a downstream owner, I want verified heads and paths named, so that limited evidence is not presented as full coverage.
23. As an investigator, I want actual failed points and directions reproduced, so that diagnostics address the observed failure.
24. As an investigator, I want a fixed step sweep and controls, so that favorable steps cannot be chosen retrospectively.
25. As an investigator, I want qualified float64 diagnostics for unresolved locations, so that precision effects can be examined safely in a separate copy.
26. As an operator, I want diagnostic budgets and terminal states predefined, so that investigation can finish honestly with unresolved results.
27. As a reviewer, I want background and subset randomness separated, so that their variability can be distinguished.
28. As an operator, I want finite candidate settings and a total budget, so that exploration cannot grow indefinitely.
29. As a downstream owner, I want candidate selection fixed before measurement, so that favorable cases or seeds cannot be selected later.
30. As a reviewer, I want every fixed case and seed pair checked, so that averages cannot conceal failures.
31. As a downstream owner, I want completeness-improvement candidates distinguished from fully verified settings, so that the largest IG run is not ground truth.
32. As an explanation user, I want retry cost, progress and cancellation, so that I can choose whether to wait.
33. As a memory reviewer, I want stages and held models identified, so that minimum-run RSS is not reported as full application peak.
34. As a future operator, I want checksummed raw backups, so that reports remain auditable after temporary files disappear.
35. As a maintainer, I want downstream quality and deployment ownership explicit, so that this repair does not claim unrelated completion.
36. As a returning collaborator, I want approved decisions and open questions separated, so that I can finish the design without repeating settled rounds.

## Implementation Decisions

### Scope, modules and immutable evidence

- Work on `dashboard_v3` or a ticket worktree derived from it. Review baseline: `c7df95b22683fbcefc0e560266bff01ef6190dc2`; #193 commits: `764bdfa`, `1e31c0f`, `cb6b1bd`, `c7df95b`. Distinguish #194 changes and freeze the actual integration commit and relevant code hashes before measuring.
- Preserve the existing desktop program, checkpoints, historical evidence, adopted Keras 3.14.1/torch runtime and frozen contract. Keep user defaults IG 32, SHAP 64 samples/4 backgrounds. No retrospective relaxation, retraining, split changes or silent default increase.
- Repair the version reporter, contract verifier, report assembler, offline preparation/runner, common SHAP function and existing cache interfaces. Reuse public CLI/computation boundaries; introduce no independent test-only interface or additional model cache.
- Before measurement, freeze run ID, contract, relevant code snapshot/hashes, checkpoint/manifest identities, cases, heads, settings and explanation protocol. Register new seed/candidate/diagnostic rules in an additive contract; preserve the original contract. Record actual commands and environments per stage, distinguishing TensorFlow reference and torch target.
- Link reference arrays, core, diagnostics, RAM, staging and raw artifacts by checksum. Reject missing, altered, stale or mixed-run inputs. Timestamp checks or adding the current contract hash to old results do not establish provenance.
- Unexpected measurement interruption or relevant input changes require a new bundle and full required remeasurement; no partial measurement resume. Reassembling a summary from the same complete immutable bundle is allowed. Valid numerical-failure reports do not stop collection of remaining required stages.
- Retain raw results, frozen contract, code and manifests in both agreed backup locations, checking copies. Connect model bytes to the existing pinned vault. Keep execution manifest, checksums, commands, summary and handoff in version control. Local-volume copies do not establish off-host/physical independence.

### Verification correctness, dependencies and recovery

- Derive the required-check inventory from frozen models, cases, heads, inputs, points, settings, layers, versions and outputs. Compare expected and observed items. Every required claim, including offline execution, affects overall status; required failure, omission and identity mismatch return nonzero exit codes.
- Judge every layer against both scaled absolute and relative parity bounds and compare layer sets. Worst-layer selection is only a summary. Keep existing numerical criteria and units; do not substitute a successful claim for another failed claim.
- Check all six versions: Python, Keras, torch, NumPy, SHAP, h5py. Missing keys fail. Check every required output, gradient, attribution, baseline/prediction, completeness/additivity term and diagnostic value, including seeds 0/1/2, repeats and finer runs. NaN/inf cannot disappear during comparison or summarization.
- Declare Keras 3.14.1, h5py 3.14.0 and necessary transitive dependencies in build/offline preparation inputs. An installed development package is not an installation specification.
- Give tools explicit output locations and verify the full documented command sequence from an empty result directory, including saving staging results. Record stage exit codes and continue collecting valid failure reports.
- Restore prepared code, models, manifests and required dependencies to an independent location. Deny original code/model path access during both-model minimum forward/gradient execution. Record checksums, module origins and actual held model state.
- Existing Python, standard library and OS components are declared external prerequisites. Record executable/version and network-block scope. Keep TensorFlow unavailable in the target minimum run; an in-process socket block is not OS-level isolation.
- Distinguish repair completion, execution completion, numerical success, warning behavior and final deployment. Required numerical failure retains overall failure/nonzero exit even if repair and failure handoff are complete.

### Aggregate investigation outcome and exit status — Q17

Keep required-check results and additional investigation states separate. Apply the following precedence; findings never override a required numerical failure.

| Condition | Overall status | CLI exit |
| --- | --- | --- |
| Required check fails/is missing, or evidence identity/integrity mismatches | `fail` | Nonzero |
| Required checks pass, but candidate measurements remain unverified or conditional float64 is `unavailable/time_limited` | `incomplete` | Nonzero |
| Required checks pass and all applicable investigation measurements are complete, but causes remain `unresolved` or no eligible candidate exists | `completed_with_findings` | Zero |
| Required checks and applicable additional investigations are complete and meet their criteria | `pass` | Zero |

- `unavailable` means qualification could not establish usable high-precision evidence, even if the qualification procedure finished. `unresolved` means planned measurements finished but did not resolve the cause. An inapplicable conditional diagnosis is not missing work.
- Prespecified budget exhaustion preserves the current bundle and completed results as a planned terminal state. It neither triggers Q8's full remeasurement rule by itself nor permits partial measurement resume. Unexpected interruption or changed inputs still requires a new bundle and full required remeasurement.
- A zero exit code reports the stated limited outcome, not all-head/path numerical approval. Preserve original failures and hand off findings/unverified work explicitly.

### SHAP concurrency and cache identity

- Serialize NumPy RNG save, seed, subset sampling and restoration with a shared lock. Guarantee participating SHAP calls only; unrelated global RNG consumers are excluded. The desktop serial worker is separate from the common-function guarantee.
- Restore RNG after normal completion and seeded-region exception/cooperative cancellation. Cancellation only during explainer initialization is insufficient evidence.
- Match checkpoint, family, head, window, settings, seeds and explanation protocol at memory/precomputed cache reuse. Reject legacy artifacts lacking identity/protocol evidence and compute fresh results. Compare accepted hits with fresh calculations.
- Independently control background and subset seeds for comparison experiments while preserving the existing single-seed mapping. Record both seeds and background construction.
- Do not blindly regenerate the four legacy NT artifacts at 64/4. NT quality approval, unit-corrected protocol and regeneration remain coordinated with #201.

### Finite candidate experiment and selection

- Limit #205 comparisons to frozen 8 cases, current SpCas9 NG explanation head and current Cas12a dual-input path. Final user-setting quality approval across required heads/adopted paths belongs to #199/#200; NT belongs to #201.
- Compare IG 32/64/128/256 once per case. Compare SHAP sample counts 64/128/256/512 at background count 4/background seed 0 with subset seeds 0/1/2. Compare background counts 4/8/16/32 at sample count 512/subset seed 0 with background seeds 0/1/2. Background expansion shares the same random-sequence prefix.
- Reuse only the identical intersection measurement within the same bundle: first-stage SHAP is 23 calls/case, 184 overall. Select at most one candidate/model, confirm independent 3×3 seeds on all four cases and make one fresh same-seed call/case. Maximum SHAP calls: 240.
- Additional candidate comparison budget: 90 minutes, checked between completed calls. On exhaustion, start no further calls and record unstarted/unverified work. Do not automatically increase budget or keep trying alternative candidates. Required baseline verification, float32 diagnosis, RAM and independent recovery are separate obligations.
- SHAP first-stage eligibility requires all four cases to satisfy criteria on the axes actually measured. Unmeasured axes remain unverified. Select the setting with shortest maximum observed fresh-call wall time; ties prefer lower sample×background count, then lower sample count, then lower background count.
- Confirmation requires all four cases and all runs to have finite required values; additivity error at most SpCas9 5e-3/Cas12a 1e-3; correlation at least 0.9 for all 36 pairs among 9 seed combinations; pairwise maximum difference at most 0.25×maximum absolute attribution at background/subset seeds (0,0); fresh (0,0) attribution difference at most 1e-9. Missing measurements or undefined correlation cannot pass. Confirmation failure ends this candidate search without automatically testing another candidate.
- Label successful SHAP results only as seed-stability-confirmed candidates under fixed cases/head/path. Preserve the original default-versus-finer gate. 512/32 is not ground truth; matching it is not a new approval gate. Three seeds do not establish population stability.
- For IG, among settings finite and complete on all four cases (SpCas9 5e-3/Cas12a 1e-3), select shortest maximum observed wall time, breaking ties by fewer steps. Label it a completeness-improvement candidate. Report 32/64/128 versus 256 resolution separately using existing correlation 0.99, scaled maximum difference 5% and strong-position sign agreement. Comparing 256 with itself does not verify its resolution. No automatic IG 512 expansion; gradient failures remain recorded.
- A non-NT precision retry may target a roughly one-minute wait per selected prediction row. Hand off estimated cost, progress and cancellation. This is neither a per-input guarantee nor a new 60-second numerical gate. NT needs separate measurement. No automatic candidate promotion to defaults.

### Gradient diagnostics and bounded float64 investigation

- Diagnose the union of historical/new failures, preserving model, head, input, point and direction. Coordinate comparisons use the entire affected input gradient; directional checks retain the 32 original directions from seed 193. Predeclare one passing control/model. Do not expand to every interpolation path.
- Fix float32 steps before execution: 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4. Preserve failures at original points/steps and match activation-boundary evidence to the actual failing stencil.
- Allow one conditional float64 investigation only on still-unresolved locations. Use a separate qualified copy with matched architecture/head/inference mode and exact expansion of stored float32 weight values; verify actual compute/variable/gradient dtypes. Input dtype changes alone are insufficient. Preserve original bytes and deployed runtime; no manual LSTM/backward reimplementation.
- Float64 reuses the six steps and adds only 1e-5, 1e-6. Record raw gradients, differences, residuals and existing comparison lines: coordinate cosine 0.999/relative L2 0.02; directional median 0.02/maximum 0.1. Report adjacent-step agreement, same-step precision effects and boundary evidence. Do not choose favorable steps retrospectively or convert original float32 failures to passes.
- Float64 budget: 30 minutes total, sequential models, at most 5 minutes/model for qualification and 10 minutes/model for diagnosis in isolated child processes. These are permitted limits, not measured runtime estimates. Prespecified budget exhaustion is a terminal condition of this conditional diagnosis, recorded as time_limited with completed/incomplete counts. Unexpected required-pipeline interruption or input changes still require a new bundle.
- Distinguish unavailable qualification/unsupported operation, unresolved causes after completed investigation and time_limited work. Report precision-effect support, boundary-crossing evidence, mixed possibilities or unresolved evidence; float64 improvement does not prove a unique cause.

### Reporting corrections and ownership

- Historical IG default failures: 5/8; 256-step success: separate 8/8. Never report combined 5/16 as the default failure fraction.
- Forward parity covered all 9 SpCas9 heads; historical XAI/gradients covered NG only. Do not claim all-head XAI or use current Cas12a dual-input evidence for a different composite path. #195 owns that path contract.
- Approximately 341 MB historical staging RSS covered minimum forward/gradient, not full IG/SHAP/UI peak or continued retention of both models. Record actual stages/held models; whole-tree comparison is #203/#204.
- #199 owns non-NT SpCas9 warning/diagnostic/retry, #200 non-NT Cas12a, #201 NT. Keep the prediction table; hand off failed explanations for warning treatment instead of normal graph presentation. Warning behavior and numerical success are separate.

## Testing Decisions

- Prefer observable behavior at the highest existing boundaries. The main seam is verification CLI/report/exit status, through independent recovery from an empty output location. The other necessary existing seam is the common SHAP callable and cache APIs for concurrency, cancellation and reuse. Existing code/test boundaries were reviewed against Q1~Q17 and the user confirmed the final scope and test plan.
- Add regressions for offline failure, missing required checks/stages, identity mismatch and altered raw inputs causing overall non-pass/nonzero exit. Check immutable summary reassembly and mixed/partial-run rejection.
- At the public reporting boundary, cover every Q17 outcome and required-failure precedence, candidate omissions, float64 unavailable/time_limited, completed unresolved/no-candidate findings, and inapplicable conditional diagnosis.
- Use raw-result fixtures at existing experiment/report boundaries to cover all-case/all-seed candidate eligibility, undefined correlation, deterministic tie-breaking, and no further candidate search after confirmation failure. Verify no new call starts after the budget boundary, completed/incomplete counts are retained, and planned termination preserves the bundle while unexpected interruption cannot resume or mix measurements.
- Inject relative/scaled failure into a layer other than the absolute-error maximum. Test missing layer sets, all six version keys and nonfinite values in nonzero seeds, repeats, finer runs and diagnostics.
- Exercise actual KernelSHAP same-input/seed concurrent calls against serial results and RNG restoration after success/seeded-region exception/cancellation. Control overlap deterministically without placing a barrier inside the serialized section.
- Test legacy-cache rejection, every identity difference, fresh/hit equality, independent seed axes and preserved single-seed behavior at existing cache/computation APIs.
- Preserve the worker-cache regression that the desktop does not load disk artifacts without provenance (`tests/test_worker_cache.py`).
- Reuse directly affected non-NT runtime, explanation, memory-cache, precomputed-cache, worker-cache and cancellation tests as prior art. Add focused verification-tool regressions; test external behavior rather than private implementation structure.
- After repairs and preregistration, rerun actual two-model forward/layer/weight parity, function match, gradients, IG and SHAP. Then execute the approved candidate/diagnostic protocols and record every required failure or unverified result.
- Run independently restored code/assets with original paths denied, verify target TensorFlow unavailability, network probes, executable/module origins, checksums and minimum forward/gradient results. Regenerate required reports using only the complete documented commands.
- Record actual commands, pass/fail/skip counts and unverified work. Candidate/diagnostic budget exhaustion or unavailable investigation is not numerical success. Historical review results/performance calls are not new passing quality tests.
- Explain selected-test relevance before execution. Use affected files/node IDs during development and selected regression scope at the end; expand for concrete failures/shared-boundary impact. A repository-wide suite requires explicit user request. Mandatory actual-model/recovery checks cannot be waived by narrow unit scope.

## Out of Scope

- Retraining, new splits, checkpoint changes, runtime replacement, retrospective tolerance relaxation or silent default-compute increases.
- New explanation-method research, exhaustive parameter search, systematic NT/non-NT research, automatic alternative-candidate fishing or partial measurement resume.
- Treating current candidate comparisons as all-head/path approval, IG 256 or SHAP 512/32 as truth; automatic IG 512 experiments; manual LSTM/backward reimplementation.
- Warning/progress/retry screen implementation and NT unit-corrected cache regeneration (#199/#200/#201); Cas12a path choice (#195).
- Final distribution build, fresh offline Python provisioning, clean Windows/OS-level isolation verification (#204), whole application-tree RAM (#203/#204), independent model-performance evaluation (#202).
- Automatic completion/closure of #193, #191 or other tickets. Failures and unverified evidence remain visible.

## Further Notes

### Current evidence and limits

- Q1~Q17, design-tree/test-boundary review and final shared-understanding confirmation are complete. Product repair, new quality-gate reexecution, official independent recovery and new agreed backup copies have not been performed in this session.
- The originating review reported 26 passing runtime/explanation tests in 21.36 seconds, reproduced offline aggregation/layer-gate defects and a controlled KernelSHAP concurrency mismatch (maximum concurrent difference about 4.9986489e-05; serial difference about 5.7984822e-05). These are historical results, not new validation of the repair. Stored actual layer measurements had zero violations; the finding was a gate defect, not demonstrated actual parity failure.
- Historical Cas12a C3 coordinate relative L2 was 0.02225866 versus 0.02, with passing cosine; its directional maximum also failed. Matching autograd results do not establish activation boundaries as the cause of all differences.
- Exploratory non-NT CPU timings exclude import/load/warmup and cover one sequence/head/model. Three-completed-call medians, seconds:

  | Setting | SpCas9 | Cas12a |
  | --- | ---: | ---: |
  | IG 32 | 0.47 | 0.93 |
  | IG 256 | 2.76 | 4.96 |
  | SHAP 64/4 | 0.52 | 1.14 |
  | SHAP 512/4 | 2.87 | 4.66 |
  | SHAP 512/32 | 19.56 | 40.06 |

  These are not application output latency or NT timing. Sequential sampled RSS included retained allocations and a TensorFlow probe/import for Cas12a, so it is not final TF-free package RAM. Performance calls completed 52/72: one was interrupted by the 180-second process limit and 19 did not start. Every table row has three completed calls.
- The additional candidate batch estimate of roughly 54 minutes extrapolates S1/C1 timings; it is not measured batch duration or a completion guarantee. Candidate 90-minute and conditional float64 30-minute budgets exclude required baseline verification and float32 diagnostics. Float64 timing/RSS is unmeasured.

### Storage and local handoff

- Raw destinations: `C:/project-backups/dashboard-v3/issue205/<run_id>/` and `D:/MC_project-backups/dashboard-v3/issue205/<run_id>/`. Verify both checksummed copies; two local volumes do not establish physical/off-host independence.
- Local records: `docs/planning/issue205-interview.md`, `docs/adr/0002-preserve-runtime-verification-provenance.md`, `docs/runtime/issue205-performance.md`, and `docs/benchmarks/issue205-performance-20261118/`.
- The previous issue body, prior local draft and publication metadata are retained under `docs/planning/issue205-publication/`. Local additions are not committed/pushed; this issue body is self-contained for a remote implementer.

### Final design review — accepted

Q17 was approved with "그렇게해". The existing assembler, contract verifier, offline staging runner, common SHAP callable and cache/cancellation tests were checked against the design. No further unsettled design choice was found. The explicit Q17, candidate-selection and budget-boundary regressions above complete the proposed acceptance coverage; this is a read-only design review, not executed repair validation.

After approving Q17, the user separately answered "그렇게해" to the final confirmation of Q1~Q17, the reviewed public test boundaries, mandatory actual-model/recovery evidence and downstream scope. The design frontier is empty and this issue is ready for implementation. This confirmation does not authorize automatically starting implementation in the design-resumption session.

### Design handoff

Preserve approved Q1~Q17 and the confirmed scope/test plan without repeating settled questions. Keep the final spec, interview and handoff aligned; retain the previous issue body and verify published content/readiness in the publication record. Implementation begins as a separate task using the sequence below.

### Planned implementation sequence after design closure

This sequence is retained as a future plan, not the next session's starting action.

1. Read dashboard-v3 work instructions, glossary, interview, provenance ADR and this issue. Inspect actual branch/worktree and preserve unrelated work.
2. Repair dependencies, verification/report/recovery behavior, SHAP locking and cache identity; add and run focused public-boundary regressions.
3. Freeze integration/code identity and additive protocol. Predeclare passing controls and deterministic configuration/seed order before measurement.
4. From a new empty bundle, run reference, core, float32 diagnostics, RAM and independent recovery; collect required evidence even when numerical gates fail.
5. Run candidate comparisons and conditional float64 protocols within budgets, retaining all terminal states. Assemble and validate the report.
6. Archive/checksum both copies, correct reporting and hand off to #195/#199/#200/#201/#203/#204. Report commands, counts, open failures and unverified work without claiming deployment.
