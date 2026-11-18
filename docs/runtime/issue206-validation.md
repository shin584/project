# #206 implementation validation

Scope: verification CLI/report/exit status and the directly changed six-version
runtime reporting boundary. Tests use synthetic raw bundles or a synthetic
runtime adapter, except the existing non-NT runtime tests, which load the pinned
models. None of these results is an official TF/torch parity, candidate,
float64, RAM or independent recovery remeasurement (#212/#213).

Tests were selected and explained before execution, following `AGENTS.md`.
The selected scope does not include the repository-wide suite or unrelated XAI
cache/concurrency work being implemented separately in this worktree.

| Actual command from dashboard_v3 | Pass | Fail | Skip | Meaning |
| --- | ---: | ---: | ---: | --- |
| `python -m pytest test/test_runtime_verification.py -q` (initial missing-evidence tracer) | 0 | 1 | 0 | Expected red: old assembler ignored explicit paths and exited 0 |
| Same initial tracer after repair | 1 | 0 | 0 | Missing evidence produces a failure report and exit 1 |
| `python -m pytest test/test_runtime_verification.py::test_missing_candidate_measurements_are_incomplete -q` (before repair) | 0 | 1 | 0 | Expected red: missing additional work needed incomplete rather than required failure |
| `python -m pytest test/test_runtime_verification.py::test_measurement_cli_records_failures_and_keeps_collecting -q -x` | 3 | 0 | 0 | Actual CLI collection with fixture adapters: nonrepresentative layer, seed 2 NaN, missing version; remaining cases/stages retained |
| `python -m pytest test/test_runtime_verification.py -q` (development regression) | 51 | 0 | 0 | Q17, missing/mixed/changed bundles, all layers/versions/seeds, replay and restart rules |
| `python -m pytest test/test_runtime_verification.py::test_planned_budget_stop_preserves_results_and_counts test/test_runtime_verification.py::test_identity_and_registered_investigation_inventory_mismatch_fails -q` | 7 | 0 | 0 | Planned stop/replay and model/head/window/case/seed inventory failures |
| `python -m pytest test/test_runtime_verification.py::test_missing_required_stage_measurements_cannot_pass -q` (before repair) | 0 | 2 | 0 | Expected red: RAM peak alone and an offline pass string were insufficient |
| `python -m pytest test/test_runtime_verification.py::test_complete_bundle_can_be_reassembled_without_measurement test/test_runtime_verification.py::test_missing_required_stage_measurements_cannot_pass -q` | 3 | 0 | 0 | Complete replay still passes; partial RAM/offline records fail |
| `python -m pytest test/test_runtime_verification.py test/test_non_nt_runtime.py -q` (selected regression before final producer guard) | 80 | 0 | 0 | Relevant verification and existing actual-model runtime boundary |
| `python -m pytest test/test_runtime_verification.py::test_registered_producer_cannot_restamp_another_runs_result -q` (before guard / after guard) | 0 / 1 | 1 / 0 | 0 | Expected red then green: foreign identity cannot be overwritten with current provenance |
| `python -m pytest test/test_runtime_verification.py::test_registered_producer_cannot_restamp_another_runs_result test/test_runtime_verification.py::test_changed_frozen_input_blocks_all_further_measurements -q` | 2 | 0 | 0 | Guard and permanent invalidation; local code import closure is frozen |

Development also caught an investigation-ID comparison bug (9 failed, 34 passed
in an intermediate file run); a later run had 42 passed and one failure because
the concurrently edited SHAP source correctly invalidated the frozen fixture.
The ID comparison was repaired, and duplicate/interrupted-stage rejection now
precedes live-source checking. Subsequent relevant runs above passed.

`git diff --check` passed. `contract.json` and historical `verification.json`
have no changes. Current source hashes are never attached to those old results.
Official measurements, archive-copy verification and independent recovery remain
unperformed and belong to the follow-up execution tickets.
