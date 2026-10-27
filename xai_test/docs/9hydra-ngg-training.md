# NGG ranking training adapter (#83)

`SpCas9.spcas9_study_backend.KerasNggBackend` implements the fixed candidate
on the existing `TrainingRows`/`FitResult` boundary. `KerasV2Backend` retains
the original objective. Neither is an authorization to run a full fit.
The historical `StudyProtocol` still describes v2 only; do not inject the
candidate into a v2 run. The shared NGG checkpoint monitor (#84) and frozen
recipe/run/budget protocol (#85) must be implemented before study execution.

The training-only wrapper shares the functional model's weights and overrides
loss computation, leaving Keras' single batch forward pass, Adam(0.001), array
shuffle and batch32 unchanged. Both methods reset the same seed before model
construction; no extra random sampling forms pairs. The last batch uses its
actual number of rows. Initial-weight SHA-256 and optimizer-update count are
included in backend evidence.

The retained MSE sums all nine heads' measured squared-error sums divided by
the actual batch size. Missing entries are removed before arithmetic, even
when their placeholders are NaN/Inf. Measured zeros and negative values remain
measurements. Masks must be binary; measured non-finite values are refused.

Each unordered pair occurs once within its original batch. For a head, both
labels must be measured, both PAMs must end in GG, exact 30mers must differ,
and labels must differ. One-hot equality compares all 30 positions. There is
no hard gap cutoff, margin, NGG oversampling or repeated network inference.
The pair penalty is `abs(y_i-y_j) * softplus(-sign(y_i-y_j)*(p_i-p_j)/0.1)`.
Each head divides by its eligible gap-weight sum (empty means finite zero).
The fixed nine-head average, including empty heads, is multiplied by 0.01
and added to MSE. Weight zero is supported only to verify MSE equivalence.

History records `masked_mse`, `ngg_ranking`, and each output's
`ngg_pairs_<output>`/`ngg_gap_weight_sum_<output>`. The latter two are epoch
sums of batch counts and normalization denominators, not a new epoch-wide
normalization. Validation history has the same statistics with a `val_`
prefix. The historical reference-MSE callback remains in place until #84.

Only the ordinary functional model is saved. Its input remains one sequence
of shape `(30, 4)` and its nine linear outputs retain their names and scale.
Loading for inference needs no custom object or training adapter. Training
resume from these checkpoints is not supported by this study.

Actual-backend checks use synthetic inputs and disposable checkpoints only:

```powershell
python -m SpCas9.spcas9_ranking_checks --ledger '<persistent absolute ledger path>'
```

Use the same ledger across attempts/processes. It reserves optimizer updates
before fitting and accumulates check elapsed time, including failures. The
parent runner kills/reaps the child at its remaining cumulative wall-time
limit, reserving one second for termination/accounting. An unfinished parent
session blocks further checks. An exclusive ledger lock prevents concurrent
runners. A killed parent can leave that lock behind: reconcile its session and
confirm the owner is stopped before removing the stale lock. The checks are opt-in so
the ordinary full suite cannot accidentally spend another six updates.
Do not reset the ledger to retry. Before #85 execution, import its
`training_seconds_charged` into the 21,600-second training budget; its six
synthetic updates do not count as full fits. Reconcile every unfinished session
using a defensible elapsed upper bound before importing this scalar; if no
bound is available, block subsequent training. The synthetic-check watchdog
does not implement the production full-fit scheduler.

The implementation-session ledger is `.scratch/issue83-synthetic-budget.json`.
Scalar MSE/all-trainable-gradient parity passed first at rtol=1e-5, atol=1e-6;
pair eligibility, missing-placeholder invariance, near-ties, weighted penalty
and correct-order behavior passed. The subsequent three-model fit/checkpoint
integration check was interrupted at the cumulative check-time boundary;
its six optimizer updates remain reserved and its completion is unverified.
The interrupted process was created at 18:50:03 KST and confirmed stopped by
18:53:33.2870555 KST on 2026-10-27. Charge a conservative ceiling of 211 seconds
for that session, in addition to the previously recorded 128.071598 seconds:
339.071598 seconds in total. This upper bound includes imports/evaluation;
the actual interrupted-session duration is unknown. The conservative charge
exhausts the 300-second allowance, so no further synthetic fitting is permitted.
That original check was incomplete. Preserve its interrupted-session charge
for #85 accounting; the recovery below does not reset or replace that history.

The tracked handoff is `docs/evidence/issue83-synthetic-checks.json`, including
source/checkpoint digests and the conservative charge. Disposable checkpoint
copies are preserved under `.scratch/issue83-interrupted-checkpoints/`; they
are local artifacts, not replacements for trained v2 checkpoints.
The final ordinary regression suite passed 690 tests (7 skipped, including
the four opt-in synthetic checks); the final watchdog/study recheck passed
six tests. Ruff and mypy passed for the changed sources. Independent review
initially found no remaining Standards issue and one unresolved Spec verification
finding: completed actual-backend gradient and final reload evidence.

## Approved verification recovery

The user subsequently approved up to 120 additional seconds for verification
of the preserved checkpoints, with no optimizer updates and no full fit. The
external-timeout recovery completed in 56.921622 seconds. This is an explicit
exception to the original check-time allowance, not evidence that all checks
finished within the original 300 seconds. The conservative cumulative charge
is now 395.993220 seconds; optimizer updates remain at six reserved, with zero
added during recovery. Import the full cumulative charge into #85.

All seven recovery checks passed:

- All three checkpoints loaded as ordinary functional models without custom
  objects, with finite saved weights and preserved original file hashes.
- Reference and zero-auxiliary fit checkpoints had identical weights, including
  non-trainable state (maximum absolute difference zero).
- The actual backend reproduced all nine outputs from the preserved candidate
  checkpoint at rtol=1e-5, atol=1e-6.
- The combined objective and all 45 trainable-tensor gradients were finite;
  zero-weight scalar and gradient parity with native Keras passed again using
  the same forward pass and masked non-finite placeholders.
- The final training adapter's loss, gradients and weighted metrics were finite
  without applying an optimizer update.
- A one-row partial batch retained native MSE reduction, and empty eligible
  heads contributed finite zero.
- Saving the shared functional inference model and loading it with default
  Keras loading preserved all nine predictions within the same tolerance.

Default loading warns that the candidate inference checkpoint has no fitted
Adam slot state. Those slots belong to the disposable training adapter, so
these checkpoints support inference verification, not training resume. The
recovery checked that optimizer iteration counts stayed unchanged.

The tracked results, approval and accounting are in
`docs/evidence/issue83-synthetic-checks.json`. The exact verifier is archived in
`docs/evidence/issue83-recovery-verifier.txt`; local execution logs and the new
disposable roundtrip checkpoint remain in `.scratch/issue83-recovered-verification/`.
This completes the outstanding implementation verification under the approved
exception. It establishes no model-performance improvement or adoption result;
full-study execution still depends on the subsequent tickets.
Follow-up independent Standards and Spec reviews found no unresolved findings
under the approved verification exception.
