# #212 integration validation

Implementation base: `b358c9a`. Measurement integration commit:
`0ef34155d365660f99b4c05a4a45ead90c549ce4`. #206–#211 are integrated in its
history. #193 baseline `c7df95b` and #194 NT-child change `ae299ed` are distinct;
the measurement uses the final integrated code without changing either runtime,
model checkpoint, historical contract/report, desktop program or deployed
IG 32 / SHAP 64/4 defaults.

## Selected regression scope

The agreed boundaries are the verification CLI/report/exit status, immutable
stage journal, existing diagnostic experiment and independent recovery CLI.
Tests were selected and explained before execution. Scope additionally covers
the integrated actual-model adapter, SHAP concurrency/seed protocol and cache
identity. No dashboard-wide or repository-wide suite was run.

| Actual command from dashboard_v3 | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| `python -m pytest test/test_runtime_handoff.py -q -x` (initial tracer) | 0 | 1 | 0 |
| `python -m pytest test/test_runtime_handoff.py -q -x` (after implementation) | 4 | 0 | 0 |
| `python -m pytest test/test_runtime_handoff.py test/test_runtime_verification.py test/test_gradient_diagnostics.py test/test_offline_recovery.py -q` | 156 | 0 | 0 |
| `python -m pytest test/test_non_nt_runtime.py test/test_shap_rng.py test/test_cache_identity.py -q` | 81 | 0 | 0 |
| `python -m pytest test/test_runtime_handoff.py -q` (final followup-preservation assertion) | 4 | 0 | 0 |

The first failure is expected red: the new combined protocol was absent. Final
selected coverage is **237 distinct passing tests**, no remaining failures or
skips. The 156-test batch took 328.55 seconds; the 81-test batch 35.08 seconds;
the final four handoff cases 20.80 seconds. Fixture results are implementation
regressions, not official two-model quality measurements.

The handoff regressions check normal incomplete reporting, missing raw arrays,
changed frozen inputs, interrupted journals, duplicate stage refusal and a real
child-process fixture returning unsupported qualification. The latter confirms
float64 followup consumes and preserves the completed float32 evidence.
Keras/NumPy deprecation and fixture input-structure warnings do not represent
skipped or failed checks. `git diff --check` and CLI help passed.

## Standards

Independent review of `b358c9a...bad1668`: no documented-standard violation;
one duplication heuristic in separately declared/executed stage ordering.
Registration now derives from the actual execution tuples. Recheck found the
vault record's digest field could be overwritten by its model-manifest field;
the record digest now uses `vault_record_sha256`. **0 remaining findings**.

## Spec

Independent premeasurement review: **0 findings**. Completed float32 receipts
bind core/diagnosis and the fixed plan. Deferred candidate/float64 work reports
unverified/incomplete. Followup preserves all saved float32 rows and arrays;
changed/interrupted evidence retains the new-bundle/full-remeasurement rule.

Official execution and archive results are recorded below after collection;
these review/test passes do not assert numerical quality or investigation
completion.

## Fresh official measurements

Run ID: `issue212-20261119-required01`. New base:
`C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01/`.
The exact invocation was:

```powershell
python -X utf8 docs/runtime/run_required_212.py --base C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01 --run-id issue212-20261119-required01 --wheelhouse C:/project-backups/dashboard-v3/issue209/wheels
```

Preregistration completed before initialization and the first measurement.
The wheelhouse contains the 39 already downloaded declared wheels; these are
dependency inputs, not reused numerical/recovery results. Every model measurement
and restored installation in this run is new. Both independent reviews found
the frozen registration complete, and all 32 source snapshots matched live
inputs. The contract SHA remains
`8c172c0865a9ef4138bbdaeed0f6e8f191eb117cd9e77c281710497512aea252`;
combined protocol SHA is
`9419f9ce56b026090f1a69f70d7d4992b503a4f96f9de9230d4ed82fe6470dd8`.

The target reports Python 3.13.7, Keras 3.14.1, torch 2.12.0+cpu, NumPy 2.3.2,
SHAP 0.52.0, h5py 3.14.0, torch backend and four threads. The separate reference
uses Keras 3.14.1/TensorFlow 2.21.0.

Fresh core: **583 observed / 583 expected, 543 passed, 40 failed, 0 skipped**;
exit 1. Both weight inventories (47 SpCas9 and 30 Cas12a variables) are bitwise
equal to reference. All four cases and every forward head pass. Maximum forward
absolute differences are SpCas9 `5.960464477539063e-08` and Cas12a
`2.9802322387695312e-08`; corresponding maximum relative differences are
`5.426328546105705e-05` and `7.156031396658392e-08`. All 23/19 non-input layers
pass both scaled absolute and relative bounds on all four cases.

The 40 required core failures comprise 10 coordinate-gradient criteria, five
default IG completeness cases, one IG resolution criterion, 16 cross-seed SHAP
criteria and eight default-versus-finer SHAP comparisons. Default IG fails
**5/8 cases**, separately from finer 256-step IG passing **8/8**. These fresh
results reproduce the historical failure pattern; the historical report and
performance calls are not counted as new quality measurements.

Fresh original diagnosis: **65 observed / 65 expected, 62 passed, 3 failed,
0 skipped**, exit 1. All 48 cross-runtime gradient comparisons pass. The three
failures are directional maximum relative error at SpCas9 S2/S3 and Cas12a C3;
all original directional arrays are retained.

The pre-sweep union has 12 locations: nine SpCas9 and three Cas12a, including
the fixed passing controls SpCas9 S3/alpha0.5/In_Seq and Cas12a C2/x/Input_Seq.
Every location completed all six float32 steps: **54 + 18 = 72 observations**,
exit 0 for collection. Coordinate sweeps use full input gradients; directional
sweeps retain all original 32 seed-193 directions. Each actual stencil's
pre-activation arrays, outputs, analytic/numeric gradients and residuals are
retained. Original failures are still `unresolved`; boundary crossings also
appear in passing controls and do not establish a unique cause.

Conditional float64 eligibility is fixed for #213: SpCas9 S1/x, S2/x,
S2/alpha0.5, S3/x, S4/x, S4/alpha0.5 (In_Seq coordinate locations) and S2/S3
alpha0.5 directional locations; Cas12a C3/x/Input_Seq and C3/alpha0.5 directional.
The two controls have no original failure and are excluded from float64. No
float64 child or candidate call has started in this official bundle.

RAM collection exited 0. Bytes below are same-child observations; each child
holds only its named non-NT model through forward and IG 32.

| Stage | SpCas9 | Cas12a |
| --- | ---: | ---: |
| Interpreter/psutil | 18,894,848 | 18,812,928 |
| Runtime import | 422,522,880 | 423,108,608 |
| Model loaded | 431,386,624 | 583,999,488 |
| After forward | 433,008,640 | 585,789,440 |
| After IG | 440,750,080 | 601,051,136 |
| Peak working set | 441,520,128 | 601,616,384 |

Both children retain finite forward/IG results. TensorFlow is importable in
this development environment and was actually imported in the Cas12a RAM
child (not the SpCas9 child). These are observed development-process RSS,
not TF-free package or full SHAP/UI/process-tree RAM. The independent recovery
target separately requires unavailable TensorFlow and retains both models.

## Independent recovery and collection outcome

Preparation copied 72 declared inputs (including 39 pinned wheels) into the new
preparation. A new pip installation from those wheels used no index/cache. The
controller executed from prepared code with `-S --snapshot-only`; the target
executed restored code with `-I -S`. Original source/model/bundle/preparation
and personal site-package access was denied. Original source/model probes:
**32/32 blocked**; unexpected denied accesses: **0**. Network probes TCP, DNS,
reverse DNS, UDP: **4/4 blocked**. These are Python-process audit guards, not
OS isolation or a clean-machine distribution claim.

The fresh recovery stage exited **0**. All six target versions match the frozen
contract. TensorFlow discovery is false, actual import raises
`ModuleNotFoundError`, and no TensorFlow module is held after loading both
models. All 3,425 file-bearing module origins were recorded without unexpected
origins. Checksums cover 367 installed RECORD/native files and 3,152 imported
dependency files against their wheel RECORD hashes, as well as complete wheels
and restored code/models/manifests. Python 3.13.7, stdlib, pip and Windows/native
OS components remain declared external prerequisites.

Both actual models remain held after Cas12a loads. SpCas9 NG/S1 returns
`0.38719987869262695` with gradient absolute sum `1.4824836254119873`.
Cas12a C1 returns `0.7837454676628113` with sequence/feature gradient absolute
sums `0.4517872929573059` / `4.448165416717529`. All seven output/gradient raw
arrays are finite. Minimum-run RSS, bytes:

| Stage (same target, both models retained after Cas12a load) | RSS |
| --- | ---: |
| Interpreter/psutil | 27,533,312 |
| Runtime import | 402,690,048 |
| SpCas9 loaded | 408,076,288 |
| SpCas9 after gradient | 411,111,424 |
| Cas12a loaded | 412,364,800 |
| Cas12a after gradient | 413,327,360 |
| Peak working set | 413,839,360 (394.67 MiB) |

The actual commands are expanded in each `<stage>.command.json` and the
versioned evidence index. The controller invocation above executes these in
the preregistered order without success-only chaining:

| Stage | Exit | Seconds | Result |
| --- | ---: | ---: | --- |
| init | 0 | 1.59 | New empty bundle; inputs frozen |
| reference | 0 | 22.29 | Separate TF reference arrays/weights |
| verify | 1 | 297.01 | 543 pass / 40 fail / 0 skip |
| diagnose | 1 | 475.72 | 62 pass / 3 fail / 0 skip |
| float32 | 0 | 252.79 | 72/72 step observations collected; unresolved causes |
| ram | 0 | 45.10 | Both model-specific children complete |
| prepare | 0 | 6.54 | 39 declared wheels and frozen assets copied |
| offline | 0 | 188.03 | Independent two-model minimum execution passes |
| assemble | 1 | 11.71 | Required numerical failures retain overall `fail` |
| replay | 1 | 10.19 | Byte-identical report, unchanged bundle |

Expected/observed inventories match exactly: **583/583 core, 65/65 diagnosis**.
Overall is **fail**, with **605 passing / 43 failing numerical checks, 0
skipped**. All 43 `evidence_errors` are the intentionally retained failed-check
entries; there is no missing stage/check, identity, raw-shape or checksum error.
The collection controller correctly exits **1**, while `handoff.json` records
completed required collection and the ready journal. No measurement stage was
interrupted or resumed.

Read-only replay used:

```powershell
python -X utf8 docs/runtime/assemble_verification.py --work C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01/bundle --output C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01/summary-replayed.json
```

All bundle file hashes were compared before/after replay, and report bytes are
identical (`reassembly-check.json`). Replay performs no new measurements.

## #213 handoff

The completed stage set is `reference, verify, diagnose, float32, ram, offline`;
journal state is `ready`, active stage is null, and no investigation receipt or
candidate/float64 directory exists. This is the preregistered stage boundary.
The same frozen code, protocol, model bytes and run ID must be retained.
Changing those inputs or unexpectedly interrupting a future measurement
requires a new bundle and full required remeasurement.

Still unstarted/unverified: IG candidate comparison 16 calls/model; SHAP screen
92 calls/model; conditional confirmation up to 28 new calls/model (20 for the
shared intersection candidate); conditional float64 one qualified investigation
per model on the ten eligible locations above. Candidate budget remains 90
minutes / maximum 240 SHAP calls; float64 remains 30 minutes total, at most
5 minutes qualification + 10 minutes diagnosis/model. No candidate/default
promotion, numerical pass, completed investigation or unique cause is claimed.

Use the exact #213 continuation commands in [required-212.md](required-212.md).
Keep the #212 archive immutable when extending the live bundle. Non-NT quality
and warning/retry UI remain #199/#200, NT #201, Cas12a path choice #195, full
application-tree RAM #203/#204 and final distribution/clean Windows #204.

## Immutable backup and final review

The auxiliary recorder was independently reviewed. It now verifies the frozen
manifest, live registered inputs and every completed receipt before copying;
uses the snapshotted registered vault record; refuses existing outputs; and
uses explicit checksum exceptions that remain active under Python optimization.
All original recorder findings are resolved. This ancillary recording tool
does not change the numerical source frozen at `0ef3415`.

Actual archival command (exit **0**):

```powershell
python -X utf8 docs/runtime/evidence/archive_boundary_212.py C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01 docs/runtime/evidence/issue212-20261119-required01.json
```

The [versioned execution/checksum record](evidence/issue212-20261119-required01.json)
contains preregistration, full commands/exits, all frozen source/model hashes,
the complete generated report, immutable artifact index, journal/handoff and
archive checksums. Full raw arrays, frozen code/contracts/manifests,
diagnostic checkpoints, logs, reports and 39 prepared wheels are retained in:

```text
C:/project-backups/dashboard-v3/issue205/issue212-20261119-required01/issue212-20261119-required01-required-evidence.zip
D:/MC_project-backups/dashboard-v3/issue205/issue212-20261119-required01/issue212-20261119-required01-required-evidence.zip
SHA-256: 64523d30c54892cca632636c90e0a12581dc5754b37b529819b8007f3d72915b
412,698,030 bytes; 172 archive entries checksum-verified against source.
Both archive copies have the same SHA-256.
```

Both existing model-vault archives were also checksum-verified, and their
checkpoint entries match the actual measured model hashes. The frozen vault
record identifies those existing locations. The new ZIP excludes the installed
dependency tree, which can be rebuilt from its included wheels; installed/imported
file hashes and origins remain in the recovery evidence. Two local volumes do
not establish off-host or physical independence.

Final independent Spec review verified snapshots, stage/auxiliary/dependency
hashes and handoff identity with **0 mismatches / 0 findings**. Standards and
recorder recheck: **0 remaining findings**. Required execution is complete;
numerical status remains **fail/1**, causes unresolved, and #213 candidate and
float64 measurements remain unverified/unstarted.
