# #141 implementation and verification

Date: 2026-11-06. Branch: `model_analysis`. Starting point:
`a8cd28ba5c36bba1831a6cb39605a73587056022`.

The user requested `implement #141` and confirmed the proposed A/C/G/T schema
and computation, save/read and durable-driver test boundaries with `go ahead`.
Independent Standards and Spec design reviews found no blockers before coding.
The [confirmed contract](../planning/issue141-ism-contract.md) records the schema
and integration clarifications. This is CPU implementation work; the scientific
freeze and actual validation/Testset execution remain #146/#147 work.

## Implementation

`nt_ism.compute_ism` retokenizes every full mutant with the saved tokenizer and
checks reconstruction, actual spans and embedding-range token IDs. Each source
evaluates 102 distinct non-WT substitutions plus one WT parity check against the
saved prediction. Batches and tokenization boundaries use the existing budget
guard. The public injected-forward boundary requires finite scalar FP32 output.

Predictions use `(source_row, position, alternative)` arrays, fixed A/C/G/T
alternatives, no head axis and preserved WT predictions/zero deltas. Native raw
and endpoint-clipped deltas remain separate. Relative delta divides by abs(WT);
abs(WT)<=1e-4 preserves NaN plus a validity mask in NPZ and JSON null plus a
reason in the saved sidecar summaries. Defined outputs must be finite.

WT/non-WT, base-valid, PAM-breaking and TTTN-preserving masks retain the actual
denominators. Only substitutions breaking invariant T positions 4,5,6 have the
PAM-breaking tag. Zero eligible alternatives produce undefined sensitivity,
not a zero estimate. This input-pattern tag makes no biological-validity or
in-distribution claim. Predictions remain native fraction outputs, unrestricted
to [0,1]; derived delta arithmetic is float64.

`save_ism` checks source IDs and complete saved prediction bindings before
committing immutable NPZ/JSON pairs. Sidecars bind actual array schemas, WT and
mutant token records, source sequence/hash, original prediction checksum,
layout, formulas, units, model/tokenizer/protocol/code/environment hashes,
quality metadata and the producing session. `read_ism` reconstructs all derived
arrays, masks, denominators and JSON summaries without loading a model. Requested
ID selection/reordering never changes the full-run coverage denominator.

The existing `Study` adds typed ISM sidecars and NPZ attachments while preserving
its prediction defaults. Both files are acknowledged before a single ledger
commit references them. Duplicate coverage is scoped by method, so predictions
and ISM coexist within the same run. Unreferenced files remain ignored;
committed missing, corrupt or incompatible shards fail checking.

`analyze.run_ism` reuses the prediction run's identity, ledger and cumulative
budget. Each executing session repeats native parity and model-state checks;
completed source shards are reused after explicit interruption settlement.
Whole-source commits limit interrupted recalculation to unfinished sources.
Complete coverage cannot hide an unresolved job. Shared execution preparation
checks fixture/device, export hashes, code snapshots and package identity before
loading. Earlier code identities cannot silently acquire this protocol; preserve
their artifacts and historical charges when creating replacement runs.

## Validation

Focused red/green cycles first demonstrated absent computation, storage and
driver behavior, followed by storage-poisoning and saved-null-summary failures.
CPU fixtures cover mutation enumeration, clipping at both range ends, relative
thresholds, actual mutant retokenization, budget interruption, schema/source
binding, ID joins, duplicate coverage, damaged/missing committed files,
unreferenced files, NPZ/sidecar write failures, and post-scan read/hash failures.

A subprocess blocks torch/Transformers imports and reads the committed study
after the synthetic model, source and partition become unavailable. A tiny saved
ESM driver fixture interrupts during the second source, settles using external
termination evidence, reuses the first source, completes the same run, and
reuses full coverage without loading again. Changed batch protocol or export
files fail before a new load. The reservation total remains study-wide.

Commands run from the repository root:

```text
python -m pytest xai_test/cas12a_NT_analysis/test -q
92 passed in 73.82s (0:01:13)
```

Ruff check and format verification passed for the six changed Python files.
`git diff --cached --check` also passed. Only the touched analysis
subtree is tested, following CLAUDE.md; the full repository suite is outside
this task's test scope.

## Implementation review

### Standards

The independent reviewer found one storage-recovery violation: direct
vocabulary/NPZ reads after an integrity scan could fail without poisoning the
manager. Regression tests reproduced both failures. `_read_checked` now checks
the immediate bytes and poisons the manager on I/O or hash failure. Re-review:
**0 remaining actionable findings**, including no remaining baseline smells.
The shared execution preparation removes the initial duplication concern.

### Spec

The independent reviewer confirmed the same recovery gap and rechecked its fix.
The final implementation covers mutation count, delta views, PAM masks,
provenance, source-ID readback and shared-run resume. Sidecars persist and verify
JSON null/reason summaries and quality metadata; GPU status derives from ISM
producing sessions. Re-review: **0 remaining findings**, with no scope creep.

Final totals: Standards 0; Spec 0. Neither axis has a remaining issue.

## Execution boundaries and limits

No real GPU execution, new FP32 Testset observation or retraining was performed.
Existing datasets, splits, checkpoints, evaluation artifacts, historical ADRs
and GitHub issue states remain unchanged. CPU fixture success does not complete
the #138 study or produce its required 153,000 actual Testset mutations.

The implementation inherits #139's sequential-writer assumption, mounted-storage
acknowledgement limits, independently retained receipt trust anchors, conservative
reservation charges and explicit uncertain-time settlement. It does not prove
remote replication or forcibly stop a hung GPU operation. Actual throughput and
the production cost forecast must be measured and accepted in #146. Hashes and
parity checks establish fixture/integrity behavior, not scientific validity.
