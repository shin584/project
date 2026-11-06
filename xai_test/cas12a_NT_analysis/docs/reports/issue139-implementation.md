# #139 implementation and verification

Date: 2026-11-06. Branch: `model_analysis`. Starting point:
`648b22e027fedb8eb6ab6ffc5c20688bea82c9c8`.

The user's latest confirmation that design review was completed and request to
implement supersede the stale waiting status in earlier planning files. The
accepted interview's concurrency removal, mounted Drive storage, uncertain-time
recovery, GPU-change revalidation and study-wide budget inheritance are preserved.
The historical planning/publication files and GitHub issue are unchanged.

## Independent design review

Two independent code-review agents reviewed Standards and Spec before coding.

- Standards: clarified current authorization versus stale implementation gate;
  retained ADR 0010's scientific pre-validation freeze. Added independently
  retained expected-tip receipts to reject ledger suffix rollback.
- Spec: pinned the historical selected-export receipt's own SHA256, added
  expected-tip receipts and post-storage acknowledgement timing checks.

Both reviewers rechecked the revised execution contract and reported zero
remaining CPU implementation design blockers.

## Implementation

Read-only verification reuses existing Cas12a library/manifest readers and their
production pins. The new path loads only saved config/tokenizer/model files,
checks actual tokens against embeddings, verifies FP32 parity and unchanged
weights, and stores validation shards with provenance and run identity. The
sequential ledger spans replacement runs and reserves time before model loading.
Closed work charges the full reservation, with no refunds. Uncertain interruption
requires external termination evidence and a conservative bound. Model-free
readback validates hashes, source binding, coverage and derived fields and
recomputes raw/clipped performance and PAM-stratum metrics.

## Validation

Focused red/green cycles cover the durable-run, verified-input, native prediction,
artifact and thin-driver boundaries selected by the acceptance criteria. CPU
fixtures use a tiny scalar ESM, real vocabulary/embedding dimensions, synthetic
partitions and a directory standing in for external storage.

Final subtree command, run from the repository root:

```text
python -m pytest xai_test/cas12a_NT_analysis/test -q
19 passed in 32.76s
```

Ruff check and format checks passed for all 12 added Python files. The CLI help
command passed. The model-free test renames its synthetic model directory,
removes its synthetic source/partition, blocks torch/Transformers imports in a
fresh subprocess, and successfully exports independently reconstructed metrics
and a checked prediction CSV/hash manifest. The full repository suite was not
run; only the touched analysis subtree was tested, as instructed by CLAUDE.md.

Environment: Python 3.13.7, torch 2.12.0+cpu, Transformers 5.8.1, NumPy 2.3.2,
pandas 2.3.3, SciPy 1.16.1. CUDA unavailable.

## Implementation review

### Standards

The independent reviewer found two correctness violations: checkpoint files
could change between verification and loading while retaining old provenance;
and the final elapsed sample could cross the reservation after the preceding
storage check. Both were reproduced with disposable fixtures. The loader now
checks the expected file set/hashes before and after native/fresh loading and
before successful closure. Final acknowledgement checks its own elapsed sample
and records an overrun before exposing a receipt. Re-review: **0 remaining
Standards findings**, with no actionable smell findings.

### Spec

The independent reviewer found three issues: the same final-acknowledgement
overrun, missing budget guards between individual parity operations, and read
failures not poisoning the manager. Regression tests first demonstrated the
failures. Budget guards now cover forwards, reloads, transfers and state hashing;
read/integrity failures require explicit reopen. Re-review: **0 remaining Spec
blockers** for the CPU implementation. Actual GPU/scientific validation remains
unverified.

## Execution status and limits

**Real GPU not executed. New FP32 Test predictions not generated. No training.**
The existing source, partitions, checkpoint and evaluation artifacts were not
modified. CPU fixture success does not complete the actual #138 study. The
common scientific specification freeze remains necessary before #146 validation.

Mounted Drive acknowledgement is not proof of remote replication. Independently
retained receipts and operator termination evidence are trust anchors. An
in-process timer cannot forcibly stop a hung GPU operation; reservation overruns
block continued execution. Simultaneous writers are unsupported and undetected,
as agreed. Coordinated rewriting of the ledger and all external trust anchors
cannot be detected by this filesystem protocol.
