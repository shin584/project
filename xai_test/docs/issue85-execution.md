# Fixed NGG study execution (#85)

The existing `StudyProtocol` boundary owns the fixed #79 study. No full fit,
optimizer update, checkpoint replacement, or partition rewrite is part of #85.

`StudyProtocol.plan(..., checkpoint_selection="ngg", execution_mode="full",
synthetic_ledger=Path(...))` validates the published inputs and records an
immutable **draft** `execution-config.json`. It includes source revision, dirty
state and source hashes, interpreter, installed dependency versions, explicit
environment settings (including unknown/default settings), split hashes,
recipes, thresholds, bootstrap rules, fixtures, budgets, order, and seed42 as
the representative model. Planning does not assert execution readiness.

`execution_configuration()` inspects that identity. `freeze_execution()` and
full training remain closed until #86's acceptance/CI and #87's finalist/Test
implementation are verified and #88 checks reporting readiness before the
final pre-fit freeze. Existing historical reports remain readable. Training
cannot bypass readiness through the historical `v2` path.

`execution_mode="verification"` exercises the same scheduling and evidence
path with an explicitly injected, picklable disposable fake backend. It rejects
the built-in real Keras backends. These are behavioral scheduler checks and
provide no evidence of model performance or actual-backend compatibility.

In fixed studies, reserve runs with `plan_run(recipe="baseline", seed=42)` and
execute with `train_validation(run_id, backend=...)`. The only order is baseline
42/43/44, then candidate 42/43/44. At most six attempts and 21,600 cumulative
training-execution seconds are available. The synthetic ledger's previous
elapsed time is included without consuming a full-fit attempt. Previously
authorized exceptions remain visible as exceptions; they do not increase the
original synthetic allowance or authorize further checks.

`budget_status()` reports durable attempts and training/evaluation time. The
`evaluation_timing(bootstrap=True)` context records bootstrap work separately;
omit the flag for evaluation work. `report_only()` records its own evaluation
time. Each timing context owns the ledger, so use it around metric computation
without nesting other study operations.

The hash-chained journal is flushed and synced throughout the fit. Exclusive kernel
ownership prevents simultaneous fits; process death releases the lock. A
separate worker and watchdog bound an unresponsive backend. Any failure,
interrupt, or timeout consumes its attempt and stops every later fit. Start
receipts detect a missing/truncated attempt history. Reusing the same synthetic
history to initialize another study is refused.

After an abnormal parent exit, use `recover_reports(fresh_destination)`. An
available child termination receipt supplies a conservative elapsed bound;
without it, elapsed time remains explicitly unknown and training stays blocked.
Recovery reuses only hash-verified completed evidence and writes reports to a
new location. It does not repair, overwrite, retry, or resume a run. Existing
inputs and source files are not required to remain available for historical
report inspection, but all consumed input/source/environment identities must
still match before any new training attempt.

The readiness gate intentionally needs extension by the downstream tickets.
They must verify the historical checkpoint identity and their own complete
acceptance, CI, report, and finalist/Test contracts before enabling the final
pre-fit freeze. They must retain the ledger, fixed recipes/order/seeds and
failure policy, and record bootstrap time separately from training.

Verification on 2026-10-28: 29 focused scheduler/check-runner tests passed;
the full `xai_test` suite passed with 739 passed and 7 skipped. Ruff lint and
format checks and mypy on the four changed source modules passed. Independent
Standards review found 0 issues; Spec review ended with 0 unresolved issues
after correcting absolute worker deadlines, consumed source/environment
verification, and synthetic-handoff locking. All new fit-scheduling checks used
fake backends. The final pre-fit freeze and real full fits remain blocked.
