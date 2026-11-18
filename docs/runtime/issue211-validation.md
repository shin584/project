# #211 implementation validation

Scope: the existing verification CLI, diagnostic experiment and report/exit
boundaries. Selected tests cover actual failure identity, full-input coordinates,
all 32 directions, fixed steps, stencil-boundary evidence, actual dtype
qualification, same-step/adjacent comparisons, process budgets/partial evidence,
abnormal exits and Q17 precedence. `test_runtime_verification.py` is directly
affected by history snapshots, producer merging, plan receipts and assembly.
Its shared fixture builder accepts an optional protocol argument so both
diagnostic and candidate regressions reuse the existing CLI fixture seam.
That fixture extraction landed in the concurrent #210 commit `9f86daf` before
this commit; #211 does not depend on an uncommitted test helper.
No unrelated subtree or repository-wide suite is run.

All new measurements here use analytic or small Keras fixtures (including LSTM,
BiLSTM and a real isolated float64 child). CLI fixture initialization snapshots
the pinned assets but supplies synthetic results. These are implementation
regressions, **not official two-model diagnostics, quality measurements or
recovery runs**. Those remain #212/#213 work.

Commands below were executed from the `dashboard_v3` worktree. Counts are actual
pytest results, with expected development failures identified explicitly.

| Command | Passed | Failed/error | Skipped | Result |
| --- | ---: | ---: | ---: | --- |
| `python -m pytest test/test_gradient_diagnostics.py -q` (first tracer) | 0 | 1 collection error | 0 | Expected red: diagnostic module absent |
| Same file after coordinate sweep implementation | 1 | 0 | 0 | Original failure/step and actual stencil retained |
| Same file after union-plan tracer | 3 | 1 | 0 | Expected red: location planner absent |
| Same file after planner | 4 | 0 | 0 | Historical/current union and 32 directions retained |
| Same file with initial dtype tracer | 4 | 1 | 0 | Serialization dtype metadata normalization needed |
| Same file after dtype repair | 5 | 0 | 0 | Separate exact-weight float64 graph qualified |
| Same file after process/report slices | 19 | 0 | 0 | Real child, timeout, unsupported, crash, partial evidence and report corruption |
| `python -m pytest test/test_gradient_diagnostics.py::test_diagnostic_inventory_and_q17_reach_existing_assembler_cli -q -x` | 5 | 0 | 0 | Unavailable remains incomplete; missing/dtype/nonfinite/inapplicable faults fail |
| `python -m pytest test/test_gradient_diagnostics.py::test_completed_float64_report_checks_qualification_and_same_step_effects -q` | 6 | 0 | 0 | Qualification, weights, head, budget and precision interpretation replay |
| `python -m pytest test/test_gradient_diagnostics.py::test_qualified_copy_expands_weights_and_uses_actual_double_compute test/test_gradient_diagnostics.py::test_input_only_double_is_not_a_qualified_runtime -q -x` (before/after BiLSTM path handling) | 2 / 4 | 1 / 0 | 0 | Graph-order variable correspondence preserves exact values despite wrapper-path normalization |
| `python -m pytest test/test_runtime_verification.py -q` | 106 | 0 | 0 | Existing verification/report/Q17/recovery regression |
| `python -m pytest test/test_gradient_diagnostics.py test/test_runtime_verification.py -q` | 143 | 0 | 0 | Integrated selected regression before the final direction-normalization check |
| `python -m pytest test/test_gradient_diagnostics.py::test_nonfinite_qualification_weights_have_an_explicit_nonfinite_reason -q` | 1 | 0 | 0 | Nonfinite source weights cannot be hidden as an ordinary unavailable qualification |
| `python -m pytest test/test_gradient_diagnostics.py -q` (direction preservation slice) | 38 | 1 | 0 | The new expected analytic literal differed by one last-place digit; corrected the fixture literal |
| `python -m pytest test/test_gradient_diagnostics.py::test_original_seed193_normalized_direction_values_are_preserved_in_float32_diagnosis -q` | 1 | 0 | 0 | Exact original normalized directions and projection retained |
| `python -m pytest test/test_gradient_diagnostics.py -q` (final) | 39 | 0 | 0 | Final diagnostic scope after direction/qualification corrections; 86.62 seconds |

The final direction check found that NumPy 2.3.2 normalizes float32 random draws
with a float64 norm. Diagnostics now preserve those original normalized values
and cast only model stencil inputs. Final selected scope: 39 diagnostic tests
plus the existing 106 CLI tests, **145 distinct passing tests**, zero remaining
failures or skips. The existing CLI scope is unchanged by the diagnostic-only
direction correction. `git diff --check` and `git diff --cached --check` passed.
Keras/NumPy deprecation and single-input probe structure warnings do not indicate
skipped tests or numerical gate failures.

## Standards

Review found no documented-standard violations and one duplication heuristic:
failure/control target metadata was constructed twice. A shared constructor now
keeps those identity fields consistent. The shared CLI fixture dependency is
available in parent commit `9f86daf`, not left in another ticket's uncommitted
implementation.

## Spec

Review found two P1 gaps in report replay: raw diagnostic inputs were not bound
to the frozen failure point, and missing boundary arrays could be presented as
zero crossings. Both were repaired: assembly checks exact frozen input values,
shapes and full coordinate vectors, and derives the required selected-head
boundary layer/chunk inventory from frozen graph bytes. CLI corruption tests
cover changed inputs, missing coordinates and deleted boundary arrays.

A subsequent P2 observation identified a trusted `boundary_support` summary
flag. Replay now derives support directly from the original actual stencil and
rejects a summary flag that disagrees. Precision interpretations are regenerated
from the raw residuals/crossings and cannot establish a unique cause.

Initial findings: Standards 0 hard/1 heuristic; Spec 2 P1 and 1 P2. All are
addressed in the final implementation. No required numerical failure is changed
to a pass. Frozen `contract.json`, historical `verification.json`, checkpoints
and deployed `non_nt_runtime.py` are unchanged.

`python -X utf8 docs/runtime/verify_contract.py --help` passed. The same help
command without UTF-8 mode hit the existing Windows cp949 console encoding
limit in the Korean docstring; no CLI implementation change was needed.
