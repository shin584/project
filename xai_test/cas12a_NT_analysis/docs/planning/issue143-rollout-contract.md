# #143 CPU token rollout contract

Scope: implement computation, checked storage and durable validation driver.
Starting point: `f2c51e2`. No real NT execution, new Test predictions, scientific
threshold approval or changes to existing checkpoints/partitions are included.
Actual validation/freeze remains #146 and actual 1,500-row rollout remains #147.

`compute_rollout(row, predictor, *, policy)` preserves native IDs/masks and
uses the native saved ESM regressor in eval FP32 eager attention, with disabled
autocast/TF32. Record Transformers/torch versions, actual attention class,
backend, layer/head counts and device. Compare fresh native output with the
stored prediction and attention-enabled output with fresh native, both at
atol=1e-5, rtol=1e-4. Missing/nonfinite/wrong-shaped probabilities fail with a
stage and reason. Execution, budget and storage exceptions propagate.

Keep original positions 0..9 (CLS and nine DNA tokens); exclude padding 10,11
before propagation. Save head-mean attention on these valid coordinates for
each actual encoder layer. Validate all returned probabilities as FP32,
finite, nonnegative, correctly shaped `(1,heads,12,12)` with stochastic rows
and zero padding-key mass (atol=1e-5, rtol=1e-4). Padded query rows are checked
as returned but never propagated. For each layer in encoder order, form
`B=0.5*mean_heads(A)+0.5*I`, row-normalize B and update `R=B@R`, starting at I.
Use float64 for accumulation. CLS remains in all products.

Save full valid-token R and the nine-by-nine DNA block R[1:,1:] without
renormalizing. Record original token positions, IDs, valid/special masks and
actual half-open DNA spans. Save each DNA query's removed CLS mass R[1:,0],
retained row mass and their sums. Sequence-only row sums need not equal one.
No output-head axis or base-coordinate expansion is introduced.

Policy is explicit, with no production defaults: `query_l1_threshold` must be
finite in [0,2]. Compare all 36 unordered DNA-query pairs over the full ten
valid-key columns: d(i,j)=sum_k abs(R[i,k]-R[j,k]). Save pair positions and
distances plus min/mean/max and fraction at or below the threshold. Classify
`near_identical_queries` iff max distance <= threshold, otherwise `distinct_queries`.
Also record the same diagnostics on the unnormalized DNA block, as a display
diagnostic. Degeneracy is valid evidence and an interpretation limit; rollout
describes attention flow, not output attribution or selective token-pair
interaction. Scientific selection of the threshold remains #146.

`save_rollout`/`read_rollout` bind immutable policy/protocol, complete prediction
checksum, source identity, token mapping, model/tokenizer/code/environment hashes,
backend/config versions, producing verified session and array axes/quantity/
normalization. Model-free readers recompute R, blocks, mass, parity and query
diagnostics from saved attention means and outputs. Numerical failures are saved
with null arrays/diagnostics and a stage/reason, block completion, and cannot
automatically retry under that plan. No fabricated zero matrix is persisted.

`run_rollout` uses the existing prediction run, execution identity, sessions,
cumulative ledger and sequential-writer/settlement semantics. Commit one
immutable per-run plan before model loading; expected IDs are all checked
prediction IDs. Verify committed shards before reuse/model loading. Commit
whole-source shards only after model-state checks; reuse verified sources after
explicit interruption settlement. Corruption, changed policy/identity and
failed-source evidence block reuse/completion. Include protocol/contract in
execution snapshots; earlier code identities require a replacement run.

Proposed test boundaries: CPU small attention model and saved tiny ESM;
artifact save/model-free read; durable driver/resume. Verify noncommuting layer
order, padding, CLS mass, near-identical and distinct rows, parity and attention
failures, tampering, source ID ordering, interruption/settlement and shard reuse.
Only synthetic CPU fixtures are run; they do not establish actual NT quality.

The user confirmed these test boundaries and explicit policy approach with `y`
before implementation. Native prediction baseline: 3 CPU tests passed.

## Independent design review

Standards: 0 blockers. Bind saved layer/head counts to the hash-checked export
config and package versions to the producing session environment; preserve
narrow numerical-failure classification and unresolved-fault semantics.

Spec: 0 blockers. Verify actual returned layers against config; retain finite
observed outputs and failing stage/reason in diagnostic failures. Scientific
freeze and actual Test execution remain distinct from CPU implementation.
