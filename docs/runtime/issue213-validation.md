# #213 registered investigations and failure handoff

Run ID: `issue212-20261119-required01`. Continue the normal, unstarted
investigation boundary recorded by #212 in the same live bundle at
`C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01/`.
Frozen integration commit: `0ef34155d365660f99b4c05a4a45ead90c549ce4`.
The 32 registered source/input snapshots and six completed required-stage
receipts were verified before execution. No numerical code, contract, input,
checkpoint, default, seed, candidate rule or budget was changed.

The supplementary recording scripts live under `docs/runtime/evidence/`;
they are outside the registered numerical source inventory. They execute the
existing preregistered command and record its outcome, observe process memory
externally, and archive the terminal boundary. They do not introduce a new
measurement, candidate, precision implementation or recovery attempt.

## Actual execution

From `C:/project/MC_project/dashboard_v3`, Windows x64 Python 3.13.7:

```powershell
python -X utf8 docs/runtime/evidence/run_followup_213.py C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01
```

This invokes the exact #212 continuation command:

```powershell
python -X utf8 docs/runtime/verify_contract.py investigation --work C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01/bundle
```

The candidate producer runs fresh IG 32/64/128/256 and both registered SHAP
axes, in order. Conditional confirmation can select at most one setting per
model. Candidate budget is 5400 seconds, checked between completed calls;
SHAP has a maximum of 240 calls. Float64 follows mandatory float32 evidence
and has its separate 1800-second total / 300-second qualification plus
600-second diagnosis per model. There is no automatic budget increase,
candidate fallback, restart or partial resume.

External RSS observation was started after candidate calculation began and
before either float64 child. It samples the controller and its descendants
every 0.25 seconds; it records PID, parent, creation time, executable, argv,
RSS and Windows peak working set. Early candidate/import observations and
subprocesses shorter than the sampling interval may be missing. This is not
a full application/UI/process-tree memory qualification.

## Results

Candidate collection completed in **1395.821 seconds** from the registered
post-warm-up clock through the last call. All 32 IG calls and 184 SHAP screen
calls completed (216/216 applicable calls); no call was left unstarted and
the 5400-second budget was not exhausted. Neither SHAP model had an eligible
first-stage candidate. Consequently both confirmation rows are
`not_applicable`, with zero new confirmation/fresh-repeat calls. These
conditional omissions are not successful 3x3-seed/fresh confirmations.

### IG completeness and separate resolution

Completeness bounds remain SpCas9 0.005 and Cas12a 0.001. All settings must
pass all four cases. Selection uses the smallest observed maximum fresh wall
time, then the smaller step count for a tie. No tie occurred.

| Model | Steps | Completeness pass / 4 | Maximum error | Maximum seconds | Resolution pass / 4 versus 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SpCas9 | 32 | 2 | 0.0410338 | 0.902 | 3 |
| SpCas9 | 64 | 3 | 0.00560066 | 0.945 | 4 |
| SpCas9 | 128 | 4 | 0.00134145 | 1.584 | 4 |
| SpCas9 | 256 | 4 | 0.00129625 | 3.289 | Not a self-test |
| Cas12a | 32 | 1 | 0.00443447 | 0.838 | 4 |
| Cas12a | 64 | 3 | 0.00341759 | 1.673 | 4 |
| Cas12a | 128 | 3 | 0.00130465 | 2.304 | 4 |
| Cas12a | 256 | 4 | 0.000616112 | 4.911 | Not a self-test |

Limited **completeness improvement candidates**: **SpCas9 128**, **Cas12a 256**.
SpCas9 128 also passes the separately registered resolution comparison on
all four cases. Cas12a 256 has no higher-resolution comparison here. These
labels neither promote defaults nor establish quality on other heads/paths.
The default completeness failures remain 5/8; the separate resolution
failure at SpCas9 S2/32 remains visible.

### SHAP axes, errors, variation and fresh cost

S = subset-seed axis at background count 4/background seed 0;
B = background-seed axis at sample count 512/subset seed 0. Every measured
axis uses seeds 0/1/2 and all three seed pairs in each of four cases.
The S+B intersection is measured once. For S-only settings, B is unverified;
for B-only settings, S is unverified. No undefined correlation was observed.

The difference ratio below is the largest observed pairwise maximum absolute
difference divided by that case's registered `0.25 * max(abs(phi(0,0)))`
bound. Every pair must have correlation >=0.9 and ratio <=1. Each row's
case count requires every measured axis and finite/additive execution to pass.

| Model | Samples/background | Axes | Cases passing / 4 | Minimum pair r | Maximum difference/bound | Maximum additivity error | Maximum seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SpCas9 | 64/4 | S | 0 | -0.089065 | 8.57138 | 2.24e-8 | 3.096 |
| SpCas9 | 128/4 | S | 0 | 0.772156 | 3.72884 | 2.24e-8 | 1.100 |
| SpCas9 | 256/4 | S | 3 | 0.907713 | 1.04406 | 2.24e-8 | 1.815 |
| SpCas9 | 512/4 | S+B | 0 | 0.036539 | 5.60140 | 2.24e-8 | 3.613 |
| SpCas9 | 512/8 | B | 0 | 0.566278 | 3.29117 | 2.98e-8 | 5.861 |
| SpCas9 | 512/16 | B | 0 | 0.774783 | 2.24421 | 2.24e-8 | 10.655 |
| SpCas9 | 512/32 | B | 3 | 0.859461 | 1.80352 | 2.98e-8 | 25.439 |
| Cas12a | 64/4 | S | 0 | -0.094748 | 3.99280 | 5.96e-8 | 1.052 |
| Cas12a | 128/4 | S | 1 | 0.374218 | 5.08339 | 5.96e-8 | 1.952 |
| Cas12a | 256/4 | S | 2 | 0.706840 | 2.22173 | 5.96e-8 | 2.558 |
| Cas12a | 512/4 | S+B | 0 | 0.398857 | 3.88425 | 5.96e-8 | 5.309 |
| Cas12a | 512/8 | B | 0 | 0.779519 | 2.98847 | 2.98e-8 | 9.353 |
| Cas12a | 512/16 | B | 0 | 0.837212 | 2.50991 | 5.96e-8 | 16.593 |
| Cas12a | 512/32 | B | 2 | 0.913016 | 1.15212 | 5.96e-8 | 36.220 |

All SHAP additivity errors satisfy their unchanged model bounds; this does
not establish seed stability. No setting passes all four cases, so there
is **no seed stability confirmed candidate** for either model. In particular,
SpCas9 256/4 narrowly fails the difference bound, while Cas12a 512/32 passes
the correlation line but still fails the difference bound. No substitute
candidate, extra seed grid or repeat call was tried.

These are observed fresh wall times including the registered per-call path,
not latency guarantees. They do not create a new one-minute pass criterion.
Exact per-case/per-seed errors, comparisons and call times remain in
`investigations.json`, `investigation-raw.npz`, `candidate-analysis-213.json`
and the final checksummed archive.

### Conditional float64 diagnostics

Both qualified children completed normally, in SpCas9/Cas12a order. Only
the ten already fixed unresolved failure locations entered float64: eight
SpCas9 and two Cas12a. The two passing float32 controls were not eligible.
All eight intervals (0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 1e-5, 1e-6)
were retained at every eligible location: **64/64 + 16/16 observations**.

Qualification verified the same serialized architecture, output heads and
inference mode; exact expansion of all **47 SpCas9 / 30 Cas12a float32
variables** to float64; actual layer tensors `torch.float64`; and actual
output/input-gradient arrays `float64` (one SpCas9 input, both Cas12a inputs).
Frozen checkpoint hashes, variable source/copy hashes, head and exact input
identities remain in the bundle and qualification records. No recurrent
forward/backward implementation or input-only double conversion was added.

| Model | Child exit | Observations complete / planned | Child seconds | Final state |
| --- | ---: | ---: | ---: | --- |
| SpCas9 | 0 | 64 / 64 | 500.708 | `unresolved` |
| Cas12a | 0 | 16 / 16 | 135.148 | `unresolved` |

The float64 batch ended at offset **635.892 seconds**, below 1800 seconds.
Qualification filesystem/OS spans were 21.654 seconds (SpCas9) and 22.117
seconds (Cas12a), below 300 seconds. Diagnosis was below 600 seconds;
the retained diagnosis start/completion file timestamps span 474.281 seconds
(SpCas9) and 110.136 seconds (Cas12a). These filesystem spans supplement the
parent's enforced monotonic deadlines and are preserved in
`precision-timing-213.json`; they are not new timing gates. There
was no budget termination, unavailable model, unstarted precision location
or resumed child. Both causes remain unresolved, not successful causal
diagnoses.

At the original coordinate step 0.01 and directional step 0.001, the full
coordinate gradient or all 32 seed-193 directions were retained. Table
residuals are L2 norms at that original step. Boundary counts below are the
actual crossing stencils and are equal in float32/float64 at that step.
The frozen comparison lines remain coordinate cosine >=0.999 / relative
L2 <=0.02, directional median <=0.02 / maximum <=0.1.

| Model / case / point / input or directions | Sweep float32 passes | Sweep float64 passes | Residual L2 float32 → float64 | Crossing stencils | Interpretation |
| --- | --- | --- | --- | ---: | --- |
| SpCas9 S1 / x / In_Seq | No | No | 0.00641680 → 0.00641664 | 91 | boundary evidence |
| SpCas9 S2 / alpha0.5 / In_Seq | No | No | 0.00774794 → 0.00774849 | 101 | boundary evidence |
| SpCas9 S2 / alpha0.5 / directions | No | Yes | 0.0000837701 → 0.0000643475 | 8 | mixed possibilities |
| SpCas9 S2 / x / In_Seq | No | No | 0.0121499 → 0.0121523 | 99 | boundary evidence |
| SpCas9 S3 / alpha0.5 / directions | No | Yes | 0.0000608702 → 0.0000309280 | 0 | precision-effect support |
| SpCas9 S3 / x / In_Seq | No | No | 0.00717917 → 0.00718123 | 120 | boundary evidence |
| SpCas9 S4 / alpha0.5 / In_Seq | No | No | 0.0228083 → 0.0228081 | 117 | boundary evidence |
| SpCas9 S4 / x / In_Seq | No | No | 0.0143826 → 0.0143825 | 120 | boundary evidence |
| Cas12a C3 / alpha0.5 / directions | Yes | Yes | 0.000143360 → 0.0000908154 | 0 | precision-effect support |
| Cas12a C3 / x / Input_Seq | No | No | 0.000818799 → 0.000820166 | 135 | boundary evidence |

All seven original-step coordinate comparisons still fail in float64.
The two SpCas9 directional sweep comparisons improve to pass. Cas12a C3
already passed the separately batched float32 sweep at that step; its
required original directional maximum failure is still retained. Differences
between sweep snapshots and the original required checks do not reclassify
those checks. No location establishes a unique cause. Boundary evidence
also exists at passing float32 controls, so crossing alone is not proof.
All eight-step comparisons, adjacent-interval agreement, same-step precision
effects, raw gradients/outputs/residuals and pre-activation boundary arrays
are retained, not reduced to a favorable step. `precision-analysis-213.json`
is an auxiliary readable index into those original results.

### Environment and observed memory

The candidate runtime records Python 3.13.7, Keras 3.14.1, torch 2.12.0+cpu,
NumPy 2.3.2, SHAP 0.52.0, h5py 3.14.0, torch backend and four torch threads.
Both float64 child stderr logs contain TensorFlow/oneDNN startup messages;
these are development-environment observations, not the separate TF-free
independent-recovery claim from #212. Actual qualified compute tensors are
torch float64. The original runtime and checkpoint bytes remain unchanged.

External observations run from 01:09:18.238 to 01:43:42.535 UTC, nominally
every 0.25 seconds plus observer overhead. The observer exited 0 and stopped
before archive collection. Complete commands/exits and raw observations are
in `rss-observer-213.command.json`, `process-rss-213.jsonl` and
`process-rss-summary-213.json`.

| Process | Maximum sampled RSS bytes | Maximum observed OS peak working set bytes |
| --- | ---: | ---: |
| Candidate producer | 745,832,448 | 748,396,544 |
| Investigation parent | 845,119,488 | 845,533,184 |
| SpCas9 float64 child | 813,182,976 | 814,317,568 |
| Cas12a float64 child | 764,956,672 | 766,885,888 |
| First report assembly | 1,458,814,976 | 1,458,819,072 |
| Read-only replay | 1,458,270,208 | 1,458,606,080 |

These are per-process observations, not sums, private physical-memory use
or a guaranteed lifetime peak after the final sample. Early candidate
observations are missing; both float64 children were observed from startup.

## Final report and stage boundary

Investigation collection: **exit 0**, 01:08:03.329–01:42:57.596 UTC.
Assembly: **exit 1**; read-only replay: **exit 1**; recording controller:
**exit 1**. The final report and replay are byte-identical, SHA-256
`374fe1f2eece0781c0a80cf5bcbde240f365435d9edb57d8c8b9941d375f130f`.
All original bundle files except the stage journal's expected
transition are unchanged. Reassembly itself changed no bundle file.

The journal is `ready`, active stage null, and all seven registered stages
have completed receipts. Final numerical status is **fail**: core 543 pass /
40 fail, original diagnostics 62 pass / 3 fail; **605 pass / 43 fail / 0 skip**
across 648 required checks. All 43 evidence-error entries are preserved
failed-check entries; there is no added missing/identity/integrity error.

Additional investigation states: two IG `pass` rows with limited candidate
labels; two SHAP `no_eligible_candidate`; two conditional confirmations
`not_applicable`; two preserved float32 and two new float64 `unresolved`.
No investigation is `unverified`, `unavailable` or `time_limited`. The generic
report counter still shows 0 completed / 28 incomplete for each conditional
confirmation definition. Their row states explicitly say `not_applicable`;
those 56 conditional calls are neither mandatory missing measurements nor
passed seed/fresh confirmations. Actual applicable candidate calls are
216/216, with zero unstarted calls. Nothing may resume this completed stage.

## Selected regression checks

The previously agreed public boundaries are candidate selection, gradient
diagnostic reporting, verification CLI/report/exit status, and the normal
float32-to-investigation handoff. No new production behavior was introduced;
existing tests at those boundaries are reused. The recording tools are also
verified by their actual official execution, syntax checks and CLI help.
| Actual command from dashboard_v3 | Passed | Failed | Skipped | Pytest seconds |
| --- | ---: | ---: | ---: | ---: |
| `python -m pytest test/test_candidate_selection.py -q -rs` | 22 | 0 | 0 | 73.72 |
| `python -m pytest test/test_runtime_verification.py test/test_gradient_diagnostics.py test/test_runtime_handoff.py -q -rs` | 149 | 0 | 0 | 277.81 |

Total: **171 distinct tests passed, 0 failed, 0 skipped**. Tests ran after
timed official measurements; fixture/smoke results are not substituted for
official evidence. Coverage includes every-case/every-seed selection,
undefined correlations, ties, failed confirmation without fallback, planned
budget stops versus interruptions, qualified/unavailable/time-limited
diagnostics, raw-evidence re-derivation and Q17 failure precedence. This is
the selected scope, not a dashboard-wide or repository-wide suite.
The 208 warnings in the second run are retained in its stdout (Keras/NumPy
fixture warnings), not counted as failed or skipped tests.

All three supplementary scripts passed `python -m py_compile`; their CLI
`--help` commands and `git diff --check` exited 0. No new test was added for
the ancillary recording scripts; actual official execution and archival
checks verify their observable results.

## Standards

The `code-review` skill reviewed changes since task-start commit
`674ed80736928093c928366171b01b749c6bc3a8` along two independent axes.
Standards review found **0 documented-standard violations**. One accepted
low-severity judgment call is duplicated inventory/hash comprehensions in
`evidence/run_followup_213.py`. The exact executing controller bytes were
recorded before measurement and are preserved rather than refactored after
execution. Final report values match the raw evidence; no new smell was found.

## Spec

Spec review found **0 findings** in the completed execution/report. It
independently verified all seven receipts and live frozen-source hashes,
the original-step raw vectors at all ten precision locations, 80 fixed
interval observations, candidate policy, Q17 statuses and replay checksum.
Final independent Standards and Spec rechecks verified both new archives,
both unchanged #212 archives, all 226 ZIP entries, and all 225 indexed
archived/current payloads against the committed inventory: **0 checksum
mismatches, 0 documented-standard violations, 0 Spec findings**. Both axes
also confirmed the 171 passing regression results. The accepted low-severity
controller duplication remains the only judgment-call smell.

## Archive and final handoff

Actual archival command, exit **0**:

```powershell
python -X utf8 docs/runtime/evidence/archive_followup_213.py C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01 docs/runtime/evidence/issue213-20261119-investigations.json
```

The [versioned execution/checksum index](evidence/issue213-20261119-investigations.json)
links the original integration commit, all 32 frozen sources, same canonical
inputs/run ID, seven-stage journal, exact commands/exits, candidate budget and
call counts, final statuses, artifact inventory and both archive copies.
Full raw/derived results remain inside the archive rather than being replaced
by a success marker. The prepared dependency wheel payload is included;
the original restored installation is reproducible from that payload.

- C: `C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01/issue212-20261119-required01-investigation-evidence.zip`
- D: `D:/MC_project-backups/dashboard-v3/issue205/issue212-20261119-required01/issue212-20261119-required01-investigation-evidence.zip`
- Size: **657,774,005 bytes**; **226 ZIP entries** checked against their source
  SHA-256, including the final archive index.
- Both copies SHA-256:
  `601bee0c02af0fd35e7e8baec4349eda92b6caa7037ecbee755ad5fdfd783ff1`.
- Both previous #212 required-evidence archives remain unchanged, SHA-256
  `64523d30c54892cca632636c90e0a12581dc5754b37b529819b8007f3d72915b`.

These are two local-volume copies, not established off-host/physical
independence. The original #212 handoff/report/archive and all required
measurements remain preserved; this archive extends the same bundle at its
preregistered normal boundary. No input change, unexpected interruption,
budget enlargement, candidate fallback or partial resume occurred.

The planned applicable investigation is complete, with no candidate calls or
float64 intervals left unstarted. Conditional SHAP confirmation is inapplicable
and unmeasured, both causes remain unresolved, original numerical failures
remain open, and other heads/paths/NT remain unapproved. Downstream owners
should review the cost/error/variation tables and raw evidence before choosing
warning or retry UI behavior. No automatic next measurement is scheduled.

## Interpretation and ownership

Original #212 numerical failures remain failures: 10 coordinate-gradient
criteria, five default IG completeness cases, one IG resolution criterion,
16 SHAP cross-seed criteria, eight default-versus-finer SHAP criteria and
three directional-gradient maxima. These are 43 failed checks on the same
eight fixed cases, not 43 distinct cases. Completing this investigation
does not turn them into numerical successes.

Any IG result is limited to a **completeness improvement candidate**; any
SHAP result requires full conditional confirmation before becoming a
**seed stability confirmed candidate**. Resolution is reported separately;
256-step self-comparison is not a success. A missing candidate or failed
confirmation does not authorize searching another setting.

Scope remains SpCas9 NG explanations and the current Cas12a dual-input path
on four frozen cases each. Approximately one minute per prediction row is
a cost target, not a guarantee or a new numerical gate. IG 32 / SHAP 64/4
deployment defaults remain unchanged. All-head/path approval and warning,
progress, cancellation and precision-retry UI belong to #199/#200, NT to
#201, Cas12a path choice to #195, full application RAM and packaging to
#203/#204. These measurements do not approve NT performance or deployment.

Q17 required-failure precedence applies: final overall `fail` and exit 1
must remain even if the planned investigations finish. `unavailable`,
`time_limited`, unstarted and unverified work are not counted as passes.
A planned budget stop preserves completed raw evidence as a terminal state
and does not permit continuation. Unexpected interruption or changed
registered input requires a new bundle and all required remeasurements.
