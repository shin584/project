# v3 computation boundary (#194)

Decision recorded before implementation, 2026-11-18. Scope is the independent
`dashboard_v3` worktree; the preserved desktop and #177 boundary are unchanged.

## Chart transport decision

| Choice | Serialization and copies | Resize | Lifetime and imports |
|---|---|---|---|
| Child renders PNG inside HTML | JSON UTF-8 over a local multiprocessing pipe; PNG buffer, base64 string (about 4/3 PNG bytes), HTML string, pipe buffer and decoded UI string coexist transiently | CSS figure width; reuse identical image | Child owns matplotlib, arrays and bounded result/fragment caches; UI owns only displayed HTML and WebView image |
| Child sends numeric arrays | JSON lists copy arrays to Python lists, wire bytes and UI arrays; binary transfer would need a second protocol | UI must implement all existing plots and coordinate labels | UI renderer must avoid importing predictor/runtime modules; separate plotting/cache ownership and new chart parity work |

Choose child-rendered PNG/HTML to preserve the existing charts. The UI imports
only the process controller, API, static screen and window libraries. All model,
tokenizer, numpy, SHAP and matplotlib imports occur in the spawned child. No
arrays, predictors, exceptions or Python callables cross the pipe.

## Execution and lifetime

Keep the existing JS functions and six slots. Add `generation` to every event.
IPC requests carry `generation`, `request_id`, `operation`, `args`; replies carry
the same identity, `type` and `value` or progress counters. JSON messages travel
via `send_bytes`/`recv_bytes` (no pickle for commands or results). Spawn bootstrap
passes only the private pipe, cancellation event and renderer factory name.

One scheduler thread and one calculation child serialize all renderer calls.
Progress callbacks observe cancellation; noninterruptible loading/rollout finish
and their obsolete results are discarded. Restart first invalidates all old
events (including ready/error), terminates and joins the child, then permits a
new generation to start. Close also terminates and confirms child exit. A dead
child produces a visible error; the next submission starts a fresh generation.
Preparation failures can be retried from the window.

The child owns the predictors, runtime references and memory caches. NT cache
identity includes family, checkpoint SHA256, score configuration and explanation
protocol, dtype and batch size, then method, head, window sequence and calculation settings. Graph
size is excluded. Old disk SHAP results have no checkpoint/protocol provenance
and are disabled in this process. PNG fragments and predictions are bounded and
die with the child. The UI keeps the current DOM images; a resize changes CSS.
Checkpoint files and runtime definitions are immutable during a process's
lifetime; replacing weights requires a restart. The predictor cache and both
render caches therefore always refer to the same loaded checkpoint generation.

Tests use the existing renderer/API seam specified in #194 plus spawned-process
integration at that same boundary. Fixed NT prediction checks use the preserved
equivalence baseline; XAI unit correctness remains #201's separate scope.

The v3 `--validate` command observes public window events and the actual DOM;
the old in-process predictor hooks cannot observe a spawned worker. It checks
unrounded raw/display scores against 13 fixed candidate rows with tolerances
declared before execution (`rtol=1e-5`, `atol=1e-8`), a CSV whose existing
columns are byte-identical to the preserved baseline (#196 appended the model
and model-version columns, so the whole file no longer is; see
`docs/desktop-boundary.md` and `detail_view.csv_against_baseline`), all four
views, six real NT XAI displays, repeated-method hits and CSS-only resizing.
Model checkpoints and head order remain unchanged. This is not an XAI quality
claim or completion of the later frozen/offline v3 packaging gate.
