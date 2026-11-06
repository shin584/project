# #142 implementation and verification

Date: 2026-11-06. Branch: `model_analysis`. Starting point:
`ad563ccaa5976af5fd1a753e9f9417e1c80a205f`.

The user requested `implement #142` and confirmed the proposed computation,
artifact save/read and durable-driver test boundaries with `진행해`. The
[CPU contract](../planning/issue142-ig-contract.md) passed independent Standards
and Spec design review before implementation. Actual NT quality thresholds
remain explicit external policy inputs; synthetic test values are not scientific
approval. The actual scientific freeze/GPU validation remains #146 and actual
Testset IG remains #148.

## Implementation

`nt_ig.compute_ig` preserves the native ID/mask forward and replaces only the
word-embedding output. ESM position construction, token-dropout, full encoder
and scalar regression head therefore remain on the native path. Temporary
hooks are removed on every exit. Alpha=1 must match both the freshly computed
native FP32 output and saved prediction at atol=1e-5, rtol=1e-4. The reference
zeros all word embeddings, including CLS and padding, with native masks and
positional processing retained. No clipping or output standardization occurs.

Midpoint integration starts at 200 steps and doubles through an explicitly
supplied cap. FP32 forwards/gradients run in eval mode with autocast/TF32
disabled; gradient/mass accumulation uses float64 and does not populate
parameter gradients. Completeness sums all 12 signed token masses. Explicit
raw-unit policy controls absolute/relative tolerances and near-zero handling;
relative residual has no denominator floor. Convergence compares consecutive
signed vectors with an L1 rule. Every completed resolution retains actual
integration batch sizes and its reconstructed quality. Final DNA and special
mass remain separate; no base projection is created.

Disconnected/nonfinite outputs, gradients or attribution produce saved failed
diagnostics, with stage, reason and attempted resolution. Connected zero
attribution is valid; zero norm and near-zero relative quantities retain null
values and explicit reasons. Numerical quality failure is distinct from a
budget, storage or interruption fault. The latter preserve unresolved-job or
poisoned-manager recovery semantics.

`nt_ig_artifacts.plan_ig` binds policy, integration batch, prediction checksum,
IDs and optional checked #140 coverage/case snapshots. Full/subset plans verify
population, partition/source/model/protocol provenance, compute the unique
union, retain overlapping case roles and preserve statistics membership.
Exactly one immutable plan is committed per run before model loading; even a
replacement plan with disjoint IDs is rejected. Every IG shard binds its plan
digest, source/prediction/token identity, actual spans, reference shape, outputs,
quality, axes/units, producing session and provenance hashes.

Whole-source NPZ/JSON pairs are acknowledged before one ledger commit references
them. Readers verify committed bytes and reconstruct quality, verdicts,
DNA/special vectors and output differences without loading a model. The
reference shape is checked against the hash-verified saved model config.
Missing/corrupt/incompatible shards fail and unreferenced files are ignored.
Diagnostic coverage and accepted coverage remain separate. Failed sources
cannot satisfy completion or trigger automatic retry under the same plan.

`analyze.run_ig` reuses the existing prediction run, identity, cumulative study
budget, per-session parity and explicit interruption settlement. It commits
sources separately and reuses checked shards on resume. Handled numerical
failure closes/charges the job before the driver reports failure; execution
faults require recovery. Shared explanation-session and artifact-read helpers
preserve the existing ISM behavior. The selection fixture was moved unchanged
to conftest so planner and selection tests share the same synthetic population.

## Verification

Focused red/green cycles established missing calculation, artifact and driver
behavior. Analytic cubic fixtures verify signed mass and special contribution;
connected-zero, detached and nonfinite-gradient variants test zero handling and
gradient defects. Tiny saved ESM fixtures verify endpoint identity and native
token-dropout/position/mask integration. Other checks cover retry exhaustion,
stored output reconstruction, model-free reads with torch/Transformers blocked,
ID ordering, plan immutability, full/subset case-overlap reuse, provenance
mismatch, corrupted/missing files, partial write failures, failure blocking,
interruption settlement, shard reuse and cumulative charges.

Final subtree command from the repository root:

```text
python -m pytest xai_test/cas12a_NT_analysis/test -q
120 passed in 138.24s (0:02:18)
```

Ruff check/format and staged whitespace verification passed for the changed
files. Only the touched analysis subtree was tested; the full repository suite
was not run.

## Implementation review

### Standards

One finding: reference width metadata could disagree with the actual saved
config. A regression first reproduced acceptance of a false width of 999.
Save/read now require `[12, hidden_size]` from the hash-checked config snapshot.
Independent re-review: **0 remaining actionable findings**. Shared execution
and read helpers also address the design review's duplication concern.

### Spec

Independent review and final re-review: **0 findings**, including no missing
CPU requirements or scope creep. Config-checked reference shape, saved failed
refinement and driver-level gradient-failure tests are included.

Final totals: Standards 0; Spec 0. Neither axis has a remaining issue.

## Execution limits

No real NT GPU execution, new FP32 Test prediction or retraining was performed.
Existing splits, checkpoints, evaluation artifacts, historical ADRs, GitHub
issue states and unrelated user changes were preserved. Current execution
entry points remain validation-only; the full/subset planner is CPU-tested
independently and rejects Test coverage attached to validation predictions.
Actual Test prediction/IG execution remains follow-up work. Earlier run code
identities cannot be silently upgraded; replacement runs retain study charges.
Storage acknowledgement, sequential-writer and external termination-evidence
trust limits remain those documented for #139.
