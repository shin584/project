# #209 implementation validation

Implementation commits: `e429242`, `f9264a1` on `dashboard_v3`, based on #208
commit `bea5772`. Existing unrelated work was preserved. The frozen #193 contract,
historical evidence, torch target/TF reference split and deployed IG 32 / SHAP
64/4 defaults were preserved.

## Selected regression scope

The supported preparation/recovery CLI and immutable report CLI are the agreed
#205/#209 seams. `test_offline_recovery.py` covers explicit preparation, missing
wheels, mixed/corrupt inputs, independent destination checks, failed-stage
receipts and transported execution with unavailable original source paths.
`test_runtime_verification.py` covers recovery identity, checksums, origins,
held models, source/network probes, RSS, case/head, all six versions, missing
evidence, raw NaN/inf and aggregate exit behavior, including preserved Q17
branches. `test_non_nt_runtime.py` covers the directly affected adopted runtime.
No repository-wide or entire dashboard suite was run.

```text
python -m pytest test/test_offline_recovery.py test/test_runtime_verification.py test/test_non_nt_runtime.py -q
132 passed, 0 failed, 0 skipped (341.30s)
```

The network review revision is additionally checked with the focused recovery,
offline-claim and Q17 selection below. The 132-case batch began before the final
reverse-DNS test was added; this extra selection includes that regression.

```text
python -m pytest test/test_runtime_verification.py test/test_offline_recovery.py -k 'offline or complete_bundle or q17' -q
54 passed, 0 failed, 0 skipped, 59 deselected (116.75s)
```

Development red/green records: the original-worktree staging regression first
failed (1 failed), then passed after independent evidence became mandatory.
Missing RSS/model checksum regressions first failed (2 failed), then the focused
selection passed (16 passed). Preparation CLI selection passed (6 passed), and
the expanded preparation/provenance selection passed (23 passed). Reverse DNS
omission first failed (1 failed), then passed after the guard/probe correction
(1 passed). One intermediate batch ran while the raw inventory was being
changed and reported 61 passed / 42 failed: its old frozen fixture was rejected
for inventory mismatch. It was superseded by the complete 132-pass run above.
These fixture results are tool regressions, not numerical model approval.

## Standards

Standards 잔여 지적 0건입니다. `f9264a1`에서 준비 단계가 동결된
`manifest["offline_dependencies"]`를 사용하도록 바뀌어 중복 파싱 지적이
해결됐습니다. 추가 변경에서도 문서화된 규칙 위반이나 새로운 설계 지적을
발견하지 못했습니다.

## Spec

Spec review of `bea5772...f9264a1`: 0 remaining findings. The runner now denies
and probes `socket.getnameinfo`; aggregate validation requires all four named
probes. Documentation distinguishes aggregate failure from an individual
claim's status. No additional missing requirements, incorrect behavior or
scope creep found.

Review totals: Standards 0; Spec 0. Initial findings were one low-priority
duplication heuristic, a P2 reverse-DNS coverage gap and document precision;
all were corrected and separately rechecked.

## Actual recovery and limits

The complete empty-directory command procedure is in
[recovery-209.md](recovery-209.md). The final development validation uses
`C:/project-backups/dashboard-v3/issue209/validation02/` and run ID
`issue209-validation02`, with frozen runtime code from `f9264a1`. The controller
runs from prepared code outside the original worktree, with `python -S` and
`--snapshot-only`; the restored target runs with `-I -S` and denies originals,
preparation, bundle and personal site-packages access. Dependency inputs are
39 pinned wheels acquired by the documented pip-download command. No personal
package tree is copied into the target. Full wheel hashes, installed RECORD and
native file hashes before/after, and imported dependency hashes against RECORD
are retained; this checksum scope is explicit in the report.

The documented `init`, `prepare` and transported recovery commands all returned
0. Preparation copied 65 frozen inputs, including 39 wheels, then pip installed
into the fresh restored tree with no index/cache. The offline receipt completed
in 239.01 seconds with exit 0; target status is `pass`, validation errors empty.
All six versions match the contract (Python 3.13.7, Keras 3.14.1, torch
2.12.0+cpu, NumPy 2.3.2, SHAP 0.52.0, h5py 3.14.0; torch backend, four threads).

Both pinned actual models ran finite forward/gradient computations: SpCas9 NG/S1
returned 0.3871998787 with input gradient absolute sum 1.4824836254;
Cas12a/C1 returned 0.7837454677 with sequence/derived-feature gradient sums
0.4517872930 and 4.4481654167. Both model objects remained held after Cas12a
loaded. All seven retained raw arrays are finite. Original source/model probes:
25/25 blocked; unexpected access violations: 0. Network probes TCP, DNS,
reverse DNS and UDP: 4/4 blocked. TensorFlow discovery is false, actual import
raises ModuleNotFoundError, and no TensorFlow module is held after model loads.
All 3,425 file-bearing module origins were recorded, with no unexpected origin.
367 installed RECORD/native files and 3,152 imported dependency files were
checksummed, with imported files matching their installed wheel RECORD hashes.

Minimum-run RSS, bytes (same process, both model objects retained):

| Stage | RSS |
| --- | ---: |
| Interpreter/psutil | 27,291,648 |
| Runtime import | 398,331,904 |
| SpCas9 loaded | 404,721,664 |
| SpCas9 after gradient | 408,031,232 |
| Cas12a loaded | 409,501,696 |
| Cas12a after gradient | 410,902,528 |
| Peak working set | 411,299,840 (392.25 MiB) |

The following existing aggregate CLI returned **1**, as required: the offline
claim is `pass`, overall is `fail`, and exactly five errors identify the absent
reference/verify/diagnose/RAM/investigation receipts. No additional checksum,
raw-array or provenance error was present. This is an expected incomplete
minimum-run bundle, not a complete official numerical verification bundle.

```text
python docs/runtime/assemble_verification.py --work C:/project-backups/dashboard-v3/issue209/validation02/bundle --output C:/project-backups/dashboard-v3/issue209/validation02/summary.json
```

The versioned [execution/checksum record](evidence/issue209-validation02.json)
retains exact commands, exits, frozen code/model identities, preparation/wheel
checksums, raw-array inventory, held models, RSS, module origins/probe summaries
and archive identity. Full reports, logs, receipts, frozen snapshots, raw arrays
and prepared wheels are retained in the archive on both local volumes:

```text
C:/project-backups/dashboard-v3/issue209/validation02/issue209-validation02-evidence.zip
D:/MC_project-backups/dashboard-v3/issue209/validation02/issue209-validation02-evidence.zip
SHA-256: 5f9286a67d4301b0c6fdd5c5274f32c7c8a6ea45ae7c94278cd3775fb4844e55
261,680,527 bytes; 103 archive entries verified against source checksums;
both archive copies have the same SHA-256.
```

Two local volumes do not establish physical or off-host independence. The
restored installed tree also remains at the recorded independent C: location.

Earlier development attempts remain separate: `dev01` rejected nested vendored
wheel metadata until the parser was corrected; `dev02` was intentionally
interrupted during exhaustive installed-file hashing and was never resumed.
`dev03` and `validation01` completed two-model recovery successfully under their
earlier frozen code; their three-probe network results do not substitute for the
final four-probe run. No interrupted or old measurement was restamped into the
new run.

This is #209 implementation/minimum-run evidence. #212 must produce a new
official complete bundle at the final integration snapshot. Full numerical
quality and investigation remain #212/#213; final distribution, fresh offline
Python and clean Windows/OS validation remain #204. Both models are retained
here; the RSS scope excludes full IG/SHAP/UI and the application process tree.
Python/stdlib/OS/pip are declared external prerequisites. Socket/file guards
are Python-process audit guards, not OS isolation or a hostile-code sandbox.
