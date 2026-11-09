# #146 returned GPU evidence audit

Date: 2026-11-09. Code at inspection: `51fe1b3`.

Input: `saved_evidence/saved-evidence146-20261109T070250927929Z.zip`,
24,739,402 bytes. User-supplied and independently computed SHA256:
`64ee60767b415daaa9af1823d61847b0dffb6055d05135c60bf3937d97860334`.
The archive and all original study files were read without modification.

## Integrity and scope

All 1,551 manifest-listed archive members matched their SHA256 values. The
614 journal records have contiguous indices, matching payload hashes and
previous-record links, nondecreasing charges and the expected study identity.
Every referenced snapshot/artifact matched its committed hash. No termination-
evidence file is referenced. No unreferenced study files were found.

All 613 retained receipts match their original journal positions. The sole
unacknowledged tail is record 613, `budget-overrun-blocked`; record 612 is the
last retained acknowledgement. This is consistent with the driver's deliberate
refusal to acknowledge an overrun. It is not permission to regenerate a receipt.

The scientific freeze and snapshotted partition/code/config identities were
checked. Existing numerical readers independently recomputed predictions,
ISM arrays/masks/summaries, IG endpoint/completeness/convergence and rollout
parity/mass/diagnostics. A read-only forensic adapter exposed only captured
bytes and state to those readers; it did not clear execution guards, register
a run, load a model, settle a job or write a replacement receipt.

The capture is from real validation on a Tesla T4, native FP32/eager,
autocast/TF32 disabled; saved torch is `2.14.1+cu130`. The real 32-row batch
parity checks passed. All saved populations are validation-only. No Test
coverage freeze or new FP32 Test predictions are present.

## Independently checked coverage

| Phase | Predictions | ISM sources | IG sources | Rollout sources | Closed |
| --- | ---: | ---: | ---: | ---: | --- |
| Correctness | 8 | 8 | 8 | 8 | Yes |
| Warmup | 2 | 2 | 2 | 2 | Yes |
| Cost 1 | 32 | 32 | 32 | 32 | Yes |
| Cost 2 | 32 | 32 | 22 | 22 | No |
| Cost 3 | 0 | 0 | 0 | 0 | Not started |

Totals: 74 checked validation predictions, 7,548 checked non-WT mutants,
64 checked IG sources and 64 checked rollout sources. No saved numerical
failure was found. Every accepted IG source converged at 400 steps, retaining
both its 200- and 400-step attempts. For cost 1, the maximum final completeness
residual / allowed limit is 0.0036332 and maximum convergence L1 / allowed
limit is 0.0644058; both are below 1. These are fractions of the predeclared
acceptance limits, not standalone scientific error percentages.

Only committed sources count. Cost 2's remaining ten IG/rollout sources and
all of cost 3 remain incomplete. Successful partial sources do not complete
the interrupted measurement or authorize Test analysis.

## Failure and measured cost

Cost 2 started on 2026-11-07 at 16:42:34.156 KST. Its recorded elapsed time
reached **1,800.072676502 seconds** at approximately 17:12:34.229 KST,
exceeding the 1,800-second phase reservation. The journal stored the overrun
and blocked further work. The secondary `manager is poisoned` exception
comes from `_Meter.measure` calling `guard()` again while handling that
already-poisoning budget exception. A deterministic fake-clock harness
reproduced this exact exception chain locally.

| Completed phase | Acknowledged elapsed, s | IG, s | Save/reload, s | Verification, s |
| --- | ---: | ---: | ---: | ---: |
| Correctness | 381.35 | 42.44 | 45.90 | 186.86 |
| Warmup | 209.73 | 11.80 | 36.90 | 71.59 |
| Cost 1 | 1,711.44 | 193.86 | 785.46 | 612.71 |

Cost 1 spent **81.70%** of its acknowledged phase time in the combined
save/reload and verification categories. Tokenization and synchronization are
nested timings and are not added again. The final receipt self-write latency
is excluded from the acknowledgement elapsed, as documented in the contract.

Two mechanisms in the implementation explain large avoidable costs:

1. `Study._ensure` reconstructs the full journal and hashes all referenced
   files on every checkpoint and many artifact-reader calls. Counting only
   checkpoint-triggered scans, cost 1 entails at least 57,268 journal-file
   reads (6.00 GB) plus 32,818 artifact-file reads (1.48 GB). Cost 2 already
   entails at least 74,721 journal-file reads (10.44 GB) plus 39,704 artifact
   reads (1.68 GB), despite fewer completed IG/rollout sources. Additional
   reader/register/start checks make logical read totals larger. These are
   filesystem read requests, not measurements of remote network transfers;
   OS/Drive caches may serve them. Bytes are decimal.
2. The driver performs a full-model state hash after both IG and rollout for
   every source, plus additional checks. Cost 1 has 67 explicit full-model
   state checks in the verification category, plus a full export hash check.
   Repeated GPU-to-CPU state copies and hashing are much more expensive than
   the small CPU fixture used during preparation revealed.

Thus the trace and completed measurements support excessive verification and
storage overhead as the main confirmed execution problem. They do not support
a numerical failure or an out-of-memory diagnosis for this stop. Loading
memory peaks also rise across phases (3.69, 7.35, 11.05 GiB); this is a separate
observation needing investigation. A synthetic CPU weak-reference probe did
not reproduce retained native/reload models, so a CUDA leak is not confirmed.

## Budget and recovery limits

Closed-job policy charges are 4,500 seconds: 1,800 correctness, 900 warmup,
1,800 cost 1. These are full reservations, not their smaller actual elapsed
times. Cost 2 is still open with a recorded elapsed lower bound of
1,800.072676502 seconds. Minimum prospective accounting is therefore
6,300.072676502 seconds; the true conservative settlement charge is unknown.

The user explicitly answered **unknown** when asked for the old runtime's
confirmed termination time. The collection manifest also has no termination
confirmation. Its 2026-11-09 collection timestamp is not a termination timestamp.
Heartbeat, error text and browser disconnection are not substituted for the
external process/VM termination evidence required by the existing contract.
The remaining six-hour budget cannot be certified. No receipt was rebuilt,
job settled, blocked flag removed or new study initialized.

There is also no accepted affordability decision: only one of three required
production groups completed. As an engineering illustration, even setting all
common verification/storage costs to zero and extrapolating only cost 1 under
the existing maximum-resolution ladder and 1.5 margin gives about 18,856 seconds
for 400+15 IG sources alone, before other future work and past charges. This
is **not** an approved forecast or a full/fallback rejection decision. It shows
why increasing the 30-minute phase timeout alone does not establish feasibility.

## Result and next decision

#146 remains incomplete. Preserve all existing data, numerical evidence,
receipts, charges and historical limitations. Repeating the unchanged notebook
would consume more resources without addressing the confirmed scaling costs.

The [proposed execution revision](../planning/issue146-execution-revision-proposal.md)
is for review and human approval before significant architectural changes.
CPU implementation/testing and any later GPU execution/budget authorization
are separate decisions. This audit does not approve new GPU time or relax
quality, coverage, dataset partitions or the historical six-hour policy.

Forensic script and machine-readable output are retained locally at
`.scratch/audit_issue146_return.py` and `.scratch/issue146-return-audit.json`.
The audit command used the user-supplied ZIP checksum and existing checked
numerical readers. No model inference or retraining was performed.

Independent read-only `code-review` agents reviewed this audit and the proposed
revision against `51fe1b3`: Standards 0 remaining actionable findings; Spec 0.
The Standards wording clarification about shared mutate/restore blind spots
was incorporated. Documentation links and staged whitespace checks passed.
Only these documentation files change; production code, returned evidence,
models, datasets, ADRs and GitHub issues remain unchanged.
