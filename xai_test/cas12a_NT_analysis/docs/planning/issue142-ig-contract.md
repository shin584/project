# #142 CPU IG contract

Scope: implement raw scalar token IG and checked durable storage. No real NT
GPU execution, new Test prediction, checkpoint change or scientific tolerance
approval is included. Actual validation and the scientific freeze remain #146;
actual Test IG remains #148. Starting commit: `ad563cc`.

## Proposed public boundaries

`compute_ig(row, predictor, *, policy, integration_batch_size=16)` computes one
checked prediction row. `predictor` supplies the existing native model,
tokenizer, synchronization and budget guard. Keep the native input IDs and mask
and replace only the output of `get_input_embeddings()` with the interpolant.
This preserves ID-derived positions, token-dropout, full encoder and scalar
regression head. Remove the temporary hook even on failure. Use eval FP32,
disabled autocast/TF32, enabled input gradients and no parameter updates.
Alpha=1 must match the saved and freshly computed native FP32 prediction with
atol=1e-5, rtol=1e-4. Zero every word embedding, including CLS and padding, at
the reference; retain the native IDs/mask throughout.

`policy` has no production defaults. It requires finite nonnegative raw-unit
`completeness_atol`, `completeness_rtol`, `near_zero`, `convergence_atol`,
`convergence_rtol` and an integer `max_steps` in {400,800,1600}. Midpoint steps
start at 200 and double through max_steps. At least two resolutions are needed.
For delta=f(input)-f(reference), residual=sum(all 12 signed masses)-delta.
Completeness requires abs(residual)<=atol+rtol*abs(delta); when
abs(delta)<=near_zero, use atol alone and report relative residual as null with
a reason. Otherwise relative residual is abs(residual)/abs(delta), with no
denominator floor. Convergence compares consecutive 12-token signed vectors:
sum(abs(current-previous))<=convergence_atol+convergence_rtol*sum(abs(current)).
Both current completeness and convergence must pass to accept a resolution.
Persist all attempted resolutions and their quality; exhausted retries fail.
These are configurable mechanics, not approval of any real NT threshold.

Disconnected or nonfinite outputs/gradients/masses fail with stage and reason.
A connected, finite, zero derivative is valid. All-zero final mass has zero
norm and undefined normalized attribution with an explicit reason; it is not
rejected for being zero. Never require every token attribution to be nonzero.
No optional base projection is needed; primary results remain nine actual DNA
spans plus separate CLS/padding signed mass. Accumulate midpoint gradients and
embedding-dimension mass in float64 from FP32 forwards/gradients.

`save_ig` / `read_ig` commit and verify one source's JSON/NPZ pair using #139's
ledger. Persist token IDs/masks/spans, reference definition, native/endpoint/
reference outputs, attempted signed vectors, actual steps and integration batch,
source/prediction identity, axes/units, policy, quality, producing session and
model/tokenizer/code/environment hashes. Model-free readers recompute quality,
output reconstruction, final DNA/special vectors and coverage from saved values.
Finite successful integration attempts are stored as float64 `(attempt,token)`
arrays, with token positions 0..11. Failed stages use JSON null and a reason,
never fabricated zeros or nonfinite JSON. Failed sources are saved but block
completion and automatic retries in the same immutable plan.

`run_ig` shares the verified prediction run, sessions, budget and recovery rules.
Require complete prediction inputs and verify all committed shards before model
loading. Commit one immutable per-run IG plan before loading, binding explicit
policy and integration batch. With no frozen coverage, require explicit
validation-probe IDs. With #140 coverage/case inputs, verify the manifests,
partition/model/protocol provenance and exact population IDs against stored
predictions; use `ig_work`'s unique union and preserve statistics/case roles.
Never choose cases here. Existing validation-only prediction runs cannot be
silently combined with a Testset coverage manifest. The same planner/save/read
interfaces support full/subset plus case-overlap reuse once checked Test
predictions are provided by follow-up execution work. Missing/extra/duplicate
IDs, changed policy/plan, corrupt files and unresolved jobs block reuse.

Each source shard is committed only after model-state checks. Whole-source
interruption requires explicit termination settlement and recalculates only
uncommitted sources. Saved failed sources remain diagnostic evidence; changing
their policy requires a replacement run retaining the same study charges.
Storage failure poisons the manager. Only explicitly classified numerical
failures become failed source results; budget/storage/interruption exceptions
propagate and retain unresolved-job recovery semantics. No locks or concurrent
writers are added. Bind every shard to its plan digest and enforce exactly one
plan per run, including when a replacement plan would have disjoint IDs. Include
this contract and protocol in source snapshots/execution identity.

## Verification

Proposed seams: CPU small scalar encoder computation/gradient-defect detection;
public artifact save/model-free read; durable driver planning/resume. Known
analytic nonlinear and zero examples check signed mass, special contribution,
endpoint, completeness/convergence and retry exhaustion; tiny saved ESM checks
the actual ID-preserving adapter. Storage checks cover corruption, failed
writes, altered derived values and policy/ID mismatch. Driver checks cover
interruption/settlement, reuse, quality-failure blocking and budget continuity.
Check both full/subset work plans and overlap-role reuse at the public planner
boundary, preserving statistics IDs while computing the unique union. Use
focused red/green tests, then only the touched analysis subtree.

## Independent design review

Standards: 0 blockers. Preserve narrow numerical failure handling, snapshot and
immutable plan bindings; reuse the existing execution preparation/lifecycle
where practical without a broad driver redesign.

Spec: 0 blockers. Enforce one plan per run and shard-to-plan digest binding;
verify both coverage modes and overlap reuse independently of the current
validation-only execution entry point. Scientific quality approval stays #146.

The user confirmed the proposed test seams and explicit policy approach with
`진행해` before implementation. Existing native prediction baseline: 3 passed
(CPU). Policy values in synthetic tests do not approve real NT thresholds.
