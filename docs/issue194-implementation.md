# #194 implementation and verification

Work is confined to the independent `dashboard_v3` branch/worktree. The
preserved `xai_dashboard_desktop` / `dashboard_core_v2` and completed #177
contract were not edited. Concurrent #193 runtime work is excluded from this
commit.

## Result

The window keeps a lightweight renderer proxy and the existing request
scheduler. A single spawned child owns NT predictors, tokenizers, model/runtime
references, XAI arrays, matplotlib and bounded prediction/chart caches. Requests
and responses use JSON bytes on a private pipe with request ID and generation.
The chart-transport comparison was recorded before implementation in
[computation-process.md](computation-process.md).

Cancellation travels back to child progress callbacks. Uninterruptible work may
finish, but obsolete output is discarded. All four event types, including
preparation ready/error, are gated by generation in Python and in the window.
Restart invalidates pending work, terminates and joins the old child before a
new child can start. A broken pipe also marks a worker unusable: Windows can
report EOF before `is_alive()` observes process death. This race is covered by
the death/retry test. Closing the app cleans up an active child.

Preparation errors are escaped and can be retried in the same window. Existing
81bp scanning, four views, filters and UTF-8 BOM/CRLF CSV remain. V3's boundary
document, executable fake and `--validate` window driver were updated together.
The frozen entry invokes `freeze_support()` and the packaging spec explicitly
includes dynamically imported worker renderers.

Cache identity includes NT family, checkpoint SHA256, model/output configuration,
dtype/batch size and explanation protocol, followed by head, sequence, method
and calculation settings. Size is excluded. Old disk SHAP files do not identify
their checkpoint/protocol, so the child disables that lookup. In-memory results
and PNGs are retained on method/size changes and die with the child. Checkpoint
replacement during a process lifetime requires restarting it.

## Selected regression scope

These tests exercise the changed process/API/cache boundary and directly affected
prediction tables, CSV, XAI rendering/cancellation and standalone imports. No
subtree-wide or repository-wide suite was run.

```powershell
python -X utf8 -m pytest tests/test_computation_process.py tests/test_worker_cache.py tests/test_desktop_api.py tests/test_desktop_xai.py tests/test_prediction_views.py tests/test_detail_view.py tests/test_shap_cache.py tests/test_nt_cancellation.py test/test_explain_cache.py test/test_explain.py tests/test_integrated_layout.py::test_core_and_fake_work_in_an_isolated_copy -q
```

135 passed, 0 failed, 0 skipped (94.18 seconds). One SHAP warning comes from the
small linear-model disk-cache exclusion test's singular regression; it is not a
claim that NT explanation quality was validated.

```powershell
python -X utf8 -m pytest tests/test_desktop_shell.py::test_shell_startup_inputs_views_and_no_http_server -q
```

1 passed, 0 failed, 0 skipped (18.59 seconds). This opens real WebView2, exercises
all four views, CSV/filtering, rapid replacement/cancel, stale generation events
and the preparation retry button. It verifies no HTTP server/listener is used.

```powershell
python -X utf8 -m pytest tests/test_nt_process.py -q -s
```

1 passed, 0 failed, 0 skipped (306.11 seconds). Actual local NT checkpoints,
float32, four CPU threads, fixed #181 input, IG 32, SHAP 64/background 4 and
Rollout for both Cas types. The 2,716-byte CSV equals the preserved baseline
byte for byte; fixed site head order is checked. All six methods generate PNGs,
resize returns identical images without progress/computation, the UI process
has no computation imports and the child exits. This test's CSV comparison
rounds raw scores to four decimals and display scores to two; the window
validation additionally checks the unrounded fixed values.

Evidence: [real NT process report](benchmarks/issue194-nt-process.json).
Its process-tree RSS includes the test UI/controller and calculation child;
it does not include WebView2. The window validation measures the WebView2 tree.
RSS sums may count shared pages more than once and are not private committed
memory measurements or a claim of NT/non-NT switching savings.

## Real window acceptance

The final command is:

```powershell
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
python -X utf8 -m desktop --validate .packaging/evidence/issue194-window-final
```

The independent fake-window dry run passed in 19.784 seconds. The deterministic
fake does not model computation-cache reuse, so that assertion is applied to
the real worker. An earlier real-window driver attempt failed when its synthetic
site-change event did not bubble to the form; the driver now dispatches
`change` with `bubbles:true`. The application itself did not need a site-change
fix. Earlier failed output is retained under `.packaging/evidence/issue194-window`.

Final real-window validation passed in 341.697 seconds. Evidence is in
[the final window report](benchmarks/issue194-window-final.json). All 13
unrounded raw/display cases passed the declared `rtol=1e-5`, `atol=1e-8`;
the CSV is byte-identical. All six NT XAI displays loaded in WebView2. Returning
to methods hit the child cache and tab/size changes sent no calculation requests.
The UI remained runtime-free, no HTTP server was created and the calculation
child exited. Peak RSS across UI, computation and WebView2 was 2,690,129,920
bytes in this run; no RAM-reduction comparison is inferred from this number.
The [fake window report](benchmarks/issue194-window-preview.json) is separate.

Final pytest total across the selected regression, shell and NT commands:
137 passed, 0 failed, 0 skipped. `git diff --check` passed. Ruff passed for the
18 changed Python files in the scoped lint command; the two other touched files
(`desktop/detail_view.py`, `test/test_explain_cache.py`) retain 11 pre-existing
lint diagnostics outside the changed lines. No unrelated lint cleanup was made.

## Standards

Independent Standards review: no documented-standard violations.

One non-blocking judgement call: possible Feature Envy / scheduler coupling in
`desktop/computation.py`. The renderer proxy reads the owner's request context,
and the dispatcher subclass invalidates inherited scheduler state. Process
lifetime and scheduling therefore depend on each other's private state.
Consider an explicit request context and scheduler-owned invalidation method
when extending this boundary. This is a maintainability suggestion, not a
confirmed lifecycle defect; the restart/late-event/death/cancel tests cover the
current behavior.

## Spec

Independent Spec review found one verification gap: byte-identical CSV alone
checks rounded raw/display scores. The requirement “기존 예측 값·헤드 순서·표시
점수를 고정 사례로 확인한다” also merits unrounded values. `data-raw` now preserves
raw scores alongside existing `data-eff`; the real window driver compares both
against the preserved full-precision fixture. The final run passed and the
independent reviewer confirmed the gap is resolved. No additional implementation
defects or scope creep were found.

Review totals: Standards 0 hard violations / 1 non-blocking suggestion; Spec
0 remaining findings (1 verification gap resolved by the final window check).
No cross-axis ranking.

## Limits

Existing NT explanation unit labels and numerical quality are not justified by
old XAI equality; correction/quality validation remain #201. No non-NT model
switching or RAM-reduction claim is made. A new frozen v3 build, clean offline
Windows validation and other-PC validation belong to later v3 tickets and were
not run here. The existing deployment remains preserved.

This worktree has no `.pre-commit-config.yaml`; the shared Git hook points at a
configuration present only in the original worktree. The commit uses
`PRE_COMMIT_ALLOW_NO_CONFIG=1` for that absent configuration after the scoped
checks above. The original worktree's hook/configuration is unchanged.
