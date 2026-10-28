# Frozen NGG study execution and reporting (#88)

The final execution uses the existing `StudyProtocol` boundary. `freeze_execution`
now seals verified essential fixtures, the source/interpreter/environment,
published input hashes, stored-v2 identity, and the original cumulative synthetic
history before any full fit. Changed or incomplete evidence refuses fitting.
The immutable draft config remains as originally recorded; the separate
`pre-fit/freeze.json` records final readiness without rewriting its identity.

Run from `xai_test/` using the verified Python 3.13 interpreter:

```powershell
python -m SpCas9.spcas9_research_run verify --destination ../.scratch/issue88-readiness-final
python -m SpCas9.spcas9_research_run execute --destination ../.scratch/issue88-execution --readiness ../.scratch/issue88-readiness-final --synthetic-ledger ../.scratch/issue83-synthetic-budget.json --scanner docs/evidence/issue88-scanner-v2-parity.json
```

`verify` explicitly selects only metrics, NGG selection/reload, study evidence,
fixed-order/budget/failure, acceptance/CI, finalist/Test, and final-report fixtures.
It does not run the repository-wide suite, actual-split pilots, or additional
synthetic optimizer checks. Its command, log, JUnit result, source and environment
are preserved with hashes. The original #83 check allowance remains exhausted;
the approved zero-update recovery exception is retained, with all
395.993219800191 seconds and six previously reserved synthetic updates recorded.

`execute` creates one fresh study and reserves all six named runs. Only the actual
Keras baseline 42/43/44 then fixed ranking candidate 42/43/44 can run, with their
full epoch ceiling and shared NGG selection. The durable scheduler includes
synthetic time in its 21,600-second allowance. Any failed, interrupted or timed-out
attempt stops every subsequent fit. Existing artifacts, checkpoint deployment,
published partitions and paused #50 experiments are unchanged.

Stored-v2 validation inference and partial reporting continue after training
stops. Both validation gates must pass before a finalist freeze and guarded Test
inference on the same rows for the six frozen checkpoints plus stored v2.
Supplementary bootstrap quota failure is reported separately and never gates
adoption. Neither a negative validation nor a negative Test triggers a new recipe,
seed, threshold, retry or checkpoint replacement.

`StudyProtocol.research_report(fresh_destination, validation=..., test=...,
scanner=...)` reconstructs per-run/population/head metrics and counts, histories,
selection/reload evidence, comparisons and CIs from verified persisted bytes.
It records implementation completion, V2-A and independent V2-B separately:

| V2-A status | Meaning |
| --- | --- |
| `improved` | Both validation and both guarded Test comparisons pass. |
| `negative_validation` | A complete validation comparison fails; candidate Test is not read. |
| `negative_test` | Complete guarded Test fails the fixed criteria. |
| `incomplete` | Required seed/metric/evidence is unavailable, damaged or interrupted. |
| `inconclusive` | The available decision has an explicitly indeterminate interpretation. |

Every outcome except `improved` keeps V2-A unresolved and selects stored v2.
Fixed-seed point estimates and protospacer resampling do not establish performance
over arbitrary training seeds or a new external population. Stored-v2 comparison
combines the ranking change with different checkpoint selection/stopping/LR
policies. Historical Test exposure is explicitly retained.

Scanner evidence is checked against its stored-v2 and current scanner identities,
all nine heads, both strands and the established numerical tolerance. A completed
parity failure is `failed`; missing or damaged evidence is `incomplete`. Neither
blocks V2-A fitting. The #88 actual-checkpoint parity run passed independently;
its evidence is [issue88-scanner-v2-parity.json](evidence/issue88-scanner-v2-parity.json).

Report recovery never trains:

```powershell
python -m SpCas9.spcas9_research_run report --study ../.scratch/issue88-execution/studies/ngg-issue88 --destination ../.scratch/issue88-recovered-report --validation ../.scratch/issue88-execution/validation --test ../.scratch/issue88-execution/test --scanner docs/evidence/issue88-scanner-v2-parity.json
```

Changed current source/interpreter identities are recorded as reconstruction
context; they cannot replace the original frozen execution identities. Original
evidence still verifies before a saved Test result is reused. A missing or corrupt
attempt journal leaves time and run states explicitly unknown, forbids training,
and still permits an incomplete report and independent scanner evidence. No
earlier partial artifacts are overwritten.

## Execution record, 2026-10-28

Implementation readiness passed 80 essential fixtures; the separate actual
stored-v2 scanner test passed once. Ruff passed all seven changed Python files
and mypy passed the six changed source modules. The full suite was not run, and
no additional synthetic optimizer updates were performed. Independent Standards
and Spec source reviews ended with zero unresolved findings.

The study was frozen at revision `8574f24` and baseline seed42 started. During
that fit, another shared-workspace commit, `034a24b` (#91), changed the revision
and converted three tracked source files from LF to CRLF. Their new bytes no
longer matched the frozen identities. Only the active fit worker was terminated;
the parent recorded one failed attempt and stopped all five subsequent fits.
There was no retry, resume or replacement study.

The attempted fit was charged **259.879503 seconds**, for **655.872723 cumulative
training seconds** including the original synthetic history. Evaluation took
23.439832 seconds and bootstrap bookkeeping 0.000048 seconds, separately charged.
Unused time does not authorize restarting after a failure. No selected research
checkpoint or complete new-run comparison exists. Candidate Test was not read.

| Result axis | Actual outcome |
| --- | --- |
| Implementation | Complete, essential fixtures and final pre-fit freeze verified. |
| V2-A | `incomplete`, still unresolved; one interrupted/failed run and five unstarted runs. |
| V2-B | `verified`, actual stored-v2 scanner parity on both strands and nine heads. |
| Supplementary CI | `unavailable`, zero valid draws because required paired-seed predictions are absent. |
| Adoption | None; stored v2 retained. |

This is an operationally incomplete study, not a completed negative result or
evidence of improvement. Stored-v2 validation inference and independent scanner
reporting were still completed after training stopped. All 104 originally pinned
source files were preserved with matching frozen hashes. Three LF originals were
reconstructed from the current CRLF files and verified against their original
digests; working-tree source files were not reverted.

The full local execution is under `.scratch/issue88-execution/`, including
the journal, six plans, failure state, original configuration/freeze,
stored-v2 validation predictions, partial reports and recovered source snapshot.
The tracked [artifact manifest](evidence/issue88-artifact-manifest.json) records
171 local artifact hashes and the execution log hash; it does not imply those
large/local artifacts are supplied by a checkout. Compact reviewable records:

- [Final research results](evidence/issue88-research-results.json)
- [Execution configuration](evidence/issue88-execution-config.json)
- [Final pre-fit freeze](evidence/issue88-pre-fit-freeze.json)
- [Implementation readiness](evidence/issue88-implementation-readiness.json)
- [Essential test results](evidence/issue88-essential-tests.xml)
- [Source-drift and preserved-source evidence](evidence/issue88-source-drift.json)
- [Independent review](evidence/issue88-code-review.md)

## Subsequent user-authorized retry

The user subsequently requested a V2-A retry. The original failed attempt and
time were retained under a one-time seven-attempt aggregate exception, with the
same six-hour cumulative limit. Six fresh runs completed in a detached worktree;
both validation comparisons failed the frozen adoption criteria. The new outcome
is `negative_validation`, with V2-A unresolved and stored v2 retained. Candidate
Test was not accessed. The original incomplete execution above remains intact;
see the [retry authorization, protocol and final results](issue88-retry.md).
