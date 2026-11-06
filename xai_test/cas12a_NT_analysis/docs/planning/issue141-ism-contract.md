# #141 CPU ISM contract

The user confirmed this schema and the computation, save/read and durable-driver
test boundaries with `go ahead`. Independent Standards and Spec design reviews
found no blockers. Their integration clarifications are included below.

Scope: implement saturation mutagenesis and durable storage for verified Cas12a
34bp inputs. Actual validation and 153,000 Testset mutations remain #146/#147
work. This contract supplies no IG tolerance, forecast or scientific approval.
Existing partitions, checkpoints, historical results and issues are preserved.

## Public boundaries

`nt_ism.compute_ism(rows, forward, tokenizer, embedding_rows, *, batch_size=32,
guard=None)` consumes checked prediction rows (source ID, sequence, saved raw
WT prediction). The injected forward returns finite scalar FP32 raw predictions.
Each full mutant sequence is retokenized and checked with `encode_sequences`.
One original forward per source checks the saved WT with #139's parity limits;
102 non-WT substitutions are evaluated in bounded batches. No clipping precedes
the native forward or raw-delta computation. All source IDs must be unique.

`save_ism(study, run_id, result)` commits an immutable NPZ and JSON sidecar in
the active existing durable run. `read_ism(study, run_id, *, source_ids=None,
require_complete=False)` validates committed files and reconstructs derived
arrays without model loading; optional source IDs select/reorder by identity.
`analyze.run_ism(inputs, study, run_id, *, reserved_seconds, batch_size=32,
device='cpu', loader=None)` requires complete checked prediction coverage,
reuses checked ISM coverage and runs session loading/parity/accounting before
calculating missing rows. It shares #139's study, run identity and budget.

## Single-output array schema, version 1

Rows are saved in ascending source-ID order. Positions are 0-based 0..33;
alternatives always have A/C/G/T order. There is no head axis. Predictions use
native fraction output units without restricting their range to [0,1].

- `source_row`: int64 `(n,)`; `wt_prediction`: float32 `(n,)`.
- `prediction`: float32 `(n,34,4)`. The WT slot contains the saved original
  prediction; only the other three slots count towards coverage (102 per row).
- `raw_delta`, `clipped_delta`: float64 `(n,34,4)`. Raw is mutant minus saved
  WT; clipped is clip(mutant,0,1) minus clip(WT,0,1). WT deltas are exactly zero.
- `relative_delta`: float64 `(n,34,4)`, raw delta / abs(WT). Where
  abs(WT)<=1e-4 every entry is undefined, represented as NaN in NPZ with an
  explicit validity mask and JSON null plus `abs(WT)<=1e-4` reason per row.
  NaN is forbidden in all defined outputs; infinity is always forbidden.
- `reference_base`: int8 `(n,34)` indexing A/C/G/T. `wt_mask`,
  `non_wt_mask`, `base_valid_mask`, `pam_breaking`, `tttn_preserving`:
  bool `(n,34,4)`. All base slots are valid. PAM-breaking means non-WT
  substitutions at invariant T positions 4,5,6. TTTN-preserving is its
  complement, including WT. These are input-pattern tags, not distribution
  or biological validity claims.
- `relative_valid`: bool `(n,)`. `denominator`, `preserving_denominator`:
  int64 `(n,34)`, respectively 3 everywhere and 0 at 4,5,6 / 3 elsewhere.
  Sensitivity is sum(abs(raw_delta) over eligible non-WT alternatives) divided
  by the appropriate denominator. A zero denominator yields null plus reason
  in model-free summaries, never a fabricated zero sensitivity.

The sidecar contains actual axes/dtypes/shapes, source sequences and hashes,
original prediction bindings, lossless token records for WT and 102 mutants,
layout, raw/relative/clipped formulas, quantity/units, quality flags and hashes
of NPZ, model, saved tokenizer, protocol, code and environment. Its full run
identity binds source/partition/export and execution settings. Run/session
records retain actual FP32 execution/parity. Readers validate token IDs and
spans against the snapshotted saved vocabulary without importing Transformers.

## Commit and recovery

Extend #139's typed artifact access with an ISM kind while preserving its
prediction defaults. Write NPZ first and sidecar second, verify byte readback,
then reference both in one ledger commit. Only ledger-referenced shards count.
Unreferenced files are preserved and ignored; missing/corrupt/incompatible
committed files and overlapping IDs block reuse. Partial source coverage is
explicit; an incomplete source shard is never committed. An interrupted source
is recalculated after termination evidence/settlement; completed sources are
reused. There is no automatic deletion, overwrite, refund or concurrent writer.

Storage errors poison the manager. Guard tokenization, forwards, serialization
and acknowledgements with the existing reservation checks. Resume verifies
identity and complete prediction inputs before loading a model. Completed ISM
coverage still cannot declare completion while its job remains unresolved.
During an active ISM job, prediction `coverage_complete` is required, rather
than prediction `complete`, which also requires that job to be closed. Duplicate
coverage is checked within each method. Each shard binds its producing session,
including the actual execution and environment; resumed GPUs may differ.
`require_complete=True` always checks full expected coverage, even when the
caller requests selected IDs. Existing runs with earlier code identity cannot
be silently upgraded; replacement runs retain the same study charges.

## Verification

The proposed test seams are the injected-forward computation, public artifact
save/read, and durable driver resume. CPU fixtures check 102 unique full mutants,
WT correspondence, clipping, relative undefined values, PAM masks/denominators,
actual retokenization, ID selection/order, corruption/missing files, duplicate
coverage, interruption and settlement. Tiny saved ESM fixtures exercise native
integration. No actual Test inference or retraining is part of this ticket.
