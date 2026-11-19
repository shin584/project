# non-NT forward and gradient runtime (#193)

The #195 [Cas12a IG decision](cas12a-ig-decision.md) selects composite IG for the
single-sequence screen, with a separate [frozen contract](cas12a-ig-contract.json)
and retained measurement evidence. Numerical quality still **fails**, including
C2 completeness at 256 steps. Historical measurements on this page and in
`contract.json` remain dual-input evidence.

The completed #205 validation bundle and reporting corrections are recorded in
[#214 archive replay and handoff](issue214-validation.md). Its required numerical
outcome is **fail / exit 1**, with 605 passing and 43 failing checks. The historical
#193 files below retain their original identity; they were not stamped with the
new contract/code identity. Forward parity covers all nine SpCas9 heads, while
XAI/gradients cover NG only and Cas12a's current dual-input path. This does not
establish all-head XAI quality, deployment readiness or full UI/IG/SHAP peak RAM.

For new runs, use the explicit bundle commands in
[Runtime verification bundles (#206)](verification-206.md). The commands and
measurements below describe historical #193 evidence; its contract, report and
raw results are preserved. #206 fixture tests do not constitute official
remeasurement or independent recovery (#212/#213).

Declared-wheel preparation and independent minimum recovery are documented in
[Independent minimum recovery (#209)](recovery-209.md).

Work in `C:/project/MC_project/dashboard_v3`, branch `dashboard_v3`.
This ticket picks the runtime that will run the two adopted non-NT checkpoints
on offline Windows and freezes the numeric contract the later tickets verify
against. It wires no screen, builds no distribution and retrains nothing.

`contract.json` is the frozen contract: runtime, versions, inference mode,
dtype, fixed cases, settings and every numeric threshold. It was committed
before the measurements ran. `verification.json` holds measured values only.
A failing criterion is reported as a failure and never relaxed afterwards.

## Adopted runtime

**Keras 3.14.1 on the PyTorch backend** (`KERAS_BACKEND=torch`, torch
2.12.0+cpu), loading the adopted `.keras` files unchanged, with **IG input
gradients from torch autograd in that same runtime**.

- No model conversion. The adopted bytes are the deployed bytes, so there is no
  converted-model SHA, no conversion tool and no conversion command to track.
  `contract.json` still records the checkpoint SHA-256 for both models and
  `model_assets.py` verifies them before a load.
- No auxiliary runtime. The runtime that computes the forward pass also
  computes the gradients, so there is no second model copy to keep alive, no
  second lifetime to manage and no second dependency tree to package.
- torch is **already** in the existing distribution (`_internal/` ships
  torch/transformers for the NT models), so adopting it adds no new large
  dependency. Keras itself is pure Python.

### Why the alternatives lost

| Option | Forward | IG input gradient | SHAP | Windows offline packaging |
| --- | --- | --- | --- | --- |
| **Keras 3 + torch backend (adopted)** | loads the adopted `.keras` directly | torch autograd, same runtime | works through the same forward | adds Keras only; torch already shipped |
| Keras 3 + TensorFlow backend | same | `tf.GradientTape`, same runtime | same | adds a whole second deep-learning runtime next to torch, the largest single cost of any option |
| ONNX Runtime | needs a conversion artifact | **not provided** - inference only | works | adds onnxruntime plus the conversion toolchain, and still needs a gradient runtime |
| Hand-written NumPy | full reimplementation of Conv1D, BatchNormalization, LSTM, Bidirectional, pooling and Dense | hand-derived backward for all of the above | works | smallest bundle, by far the largest correctness risk |

The deciding points, in order:

1. **Gradients are not optional.** IG needs `d output / d input`, so a
   forward-only runtime does not finish the job. ONNX Runtime would have to be
   paired with a second gradient implementation, which means two runtimes to
   package, two lifetimes to manage and a parity obligation between them. That
   is strictly more work and more risk than using one runtime that already does
   both. The same argument retires the NumPy option, which would additionally
   put a hand-derived LSTM/BiLSTM backward pass on the critical path for
   numbers a user reads as biology.
2. **The checkpoints are backend-agnostic.** Both files are Keras 3.14.1
   archives holding `config.json` plus `model.weights.h5`, with no TensorFlow
   graph inside. `.keras` does not imply TensorFlow. Keras 3 reads them under
   whichever backend the process selects.
3. **One of the two candidate backends is already deployed.** Choosing torch
   reuses the shipped runtime; choosing TensorFlow ships a second one. With RAM
   as #191's stated priority, importing a second framework and its allocator
   into the computation process is the wrong direction.

### The one real risk, and what it is

Under the torch backend, constructing a `Bidirectional` layer reaches
`keras.src.saving.serialization_lib`, which asks `tf.available` and therefore
*imports TensorFlow if it happens to be installed*. That is a probe through
`keras.src.utils.module_utils.LazyModule`, not a dependency: when TensorFlow
is absent, `available` returns `False` and loading continues. On this
development machine TensorFlow **is** installed, so a naive measurement here
would silently include it and overstate both RAM and bundle size.

The staged TensorFlow-free run recorded in `verification.json` is what settles
this: the Cas12a BiLSTM model loads, predicts and produces IG gradients with no
`tensorflow` importable at all. Without that run, the TF-free claim would rest
on reading library source, which is not evidence.

## Where the module sits

`non_nt_runtime.py` is a root module next to `predict.py`, and it is deliberately
**not** added to the `core` facade. The facade exposes the abstract surface and
the loader - `BasePredictor`, `ExplainHandle`, `ModelSpec`, `load_predictors` -
while the concrete predictor machinery stays behind it; `NTClassifierPredictor`
is not exported either. This runtime is that same kind of machinery, and
#196/#197 will reach the screens through a predictor that the facade already
exposes. Adding exports now would widen the public surface for a consumer that
does not exist yet. The module keeps `import core` free of model libraries:
its module-level imports are `os`, `numpy`, `model_source` and the lightweight
`predict.ExplainHandle`, with keras and torch imported inside functions.

`RuntimeHandle` extends `predict.ExplainHandle` rather than redeclaring its
fields, so the handle surface has one definition; it adds only `input_totals`,
the per-input signed contribution sums that keep the Cas12a feature-input
contribution from being silently dropped or spread over bases.

## Packaging method

`packaging/desktop.spec` currently excludes `tensorflow` **and** `keras`.
The adopted runtime needs:

- `keras` removed from `excludes`, and collected as a package (it imports
  submodules dynamically, so it needs `collect_submodules`, not a single hidden
  import).
- `tensorflow` **kept** in `excludes`, together with `tensorboard`, so the
  `tf.available` probe resolves to `False` inside the bundle. This is both the
  size decision and the determinism decision: it removes the probe's effect
  rather than relying on it staying harmless.
- `KERAS_BACKEND=torch` set in-process before the first `keras` import, by the
  runtime module rather than by the environment, so a developer shell with a
  different `KERAS_BACKEND` cannot change which backend the app loads.
- The two adopted `.keras` files added as data, at their canonical
  `models/non_nt/...` paths.

This ticket edits the spec and verifies the runtime import path offline from a
local staging tree. It does **not** build a distribution: the actual
PyInstaller build, its measured size delta and a clean offline PC belong to
#204, and the UI/computation process split belongs to #194.

## What the verification found

Commands, in order (from the worktree root):

```
python docs/runtime/verify_contract.py reference   # tensorflow backend, separate process
python docs/runtime/verify_contract.py verify      # torch backend, judges the frozen criteria
python docs/runtime/verify_contract.py diagnose    # addendum 1 gradient diagnostics
python docs/runtime/verify_contract.py ram         # per-stage RSS, one model per process
python docs/runtime/stage_offline.py <path>        # historical #193 staging command
python docs/runtime/assemble_verification.py       # writes verification.json
```

`assemble_verification.py` derives the pass/fail summary from the recorded check
list instead of restating it by hand, and `verification.json` carries a
`corrections` list for places where an earlier draft of this write-up was wrong.
The raw reports stay under Git-excluded `.packaging/issue193/`; every command
above regenerates them.

### Historical runtime parity measurements

| Claim | Measured | Frozen bound |
| --- | --- | --- |
| Forward parity vs the TensorFlow backend, all cases, all 9 + 1 heads | 6.0e-8 (SpCas9), 3.0e-8 (Cas12a) max abs | 1e-4 |
| Layer-by-layer activation parity, worst layer | 1.07e-6 at `lstm`, 1.37e-6 at `bidirectional` | 1e-4 x scale |
| Weight placement | identical variable sets, bitwise identical float32 bytes | bitwise |
| IG path function match | exactly 0.0, all 8 cases | 1e-6 |
| Cross-runtime gradient parity (torch autograd vs `tf.GradientTape`), per input | 2.4e-7 max abs, cosine 0.999999999999, 48 of 48 | 1e-4 / 0.9999 |
| SHAP additivity, and reproducibility at a fixed seed | 8 of 8 each; repeats now bitwise identical | 1e-3 / 5e-3, 1e-9 |
| Finiteness, 24 checks | all finite | all finite |
| Offline, TensorFlow-free, network-blocked staged run | both models load, predict and give gradients; same forward values | pass |

The LSTM and BiLSTM layers matching to about 1e-6 *layer by layer* is what rules
out a transposed kernel or a swapped gate order, which a final-scalar comparison
alone can hide. Across both models and all 8 cases, 164 of 168 layer comparisons
sit below 1e-6, and the four that do not are the recurrent layers themselves at
1.07e-6 and 1.37e-6 - the layers where accumulation order differs most.

### Historical minimum-process RSS

| Stage | TensorFlow absent (staging) | This PC, TensorFlow installed |
| --- | --- | --- |
| Interpreter | 20.9 MB | 18.7 MB |
| After importing the runtime | 330.2 MB | 424.1 MB |
| Both models loaded, gradients run | 341.3 MB peak | 601.5 MB peak (Cas12a) |

The historical staging run with TensorFlow unavailable recorded a 341.3 MB
minimum forward/gradient process peak with **both** models held at that point.
This did not measure full IG/SHAP/UI peak or continued model retention. The
development measurement used a different process and workload, so its difference
cannot be attributed exclusively to the TensorFlow probe. The selected-family
process design still needs whole-tree validation in #203/#204. Bundle
inputs for the same decision: Keras 13.7 MB of site-packages against
TensorFlow's 1408.4 MB, with torch's 492 MB already shipped.

### Three groups of numerical failures remain open

Six registered criteria are open, grouped into the three themes below.
`verification.json`'s `open_failures` is the authoritative list, derived from the
recorded checks rather than written by hand: `shap_cross_seed_stability`,
`shap_default_vs_finer_agreement`, `ig_completeness_at_user_default_steps`,
`ig_resolution_stability`, `coordinate_finite_difference_gradient_check` and
`directional_finite_difference`. Ten criteria pass, listed there as
`verified_claims`.

1. **SHAP at the user's default settings is not a stable estimate.** With the
   seed fixed, 64 samples over 4 backgrounds reproduces exactly (see below), and
   additivity holds in all 8 cases. But changing the background draw changes the
   explanation: pairwise correlation across seeds 0/1/2 runs from -0.13 to 0.68
   against a frozen bound of 0.9, and the default disagrees with a 512-sample /
   32-background reference at correlations of 0.22 to 0.82. All 24 of these
   checks fail. The positions a user would read as "most important" are
   therefore not stable at 64/4. `shap` itself warns about this on every run:
   the 64-sample regression over 30 or 34 features is singular.
2. **IG completeness at the 32-step default fails in 5 of 8 cases**
   (worst 4.1e-2 against 5e-3 for SpCas9 S3; Cas12a C2 4.4e-3 against 1e-3). At
   256 steps the separate eight-case comparison passes 8/8, worst 1.3e-3.
   This is a different setting, not a combined 5/16 default failure rate.
   Resolution is a separate gate: correlation is at least 0.9992, but SpCas9 S2
   misses the scaled max-difference bound by 0.5 percent and therefore fails.
3. **The coordinate-wise finite-difference gradient check fails in 10 of 48
   per-input checks, which is 7 of 24 case/point/input combinations** (worst
   relative L2 0.264, cosine 0.969; 9 on SpCas9 `In_Seq`, 1 on Cas12a
   `Input_Seq`). The registered diagnostics then showed cross-runtime gradient
   parity passing all 48 per-input checks at 2.4e-7 with cosine
   0.999999999999, directional finite-difference median relative error of
   3.2e-4 to 2.4e-3, and hundreds of ReLU/LeakyReLU sign flips inside the
   h = 1e-2 coordinate steps. Cross-runtime agreement and boundary crossings
   do not establish a unique cause or invalidate the failed criteria. The
   coordinate failures and 3 of 16 directional maxima exceeding 1e-1
   (0.286, 0.122, 0.104) remain failures. The fixed-location float32/qualified
   float64 investigation in #212/#213 completed with unresolved causes;
   all seven original-step coordinate comparisons still fail in float64.

   This check is also where a review caught a flaw in my own measurement. The
   first run pooled Cas12a's two inputs into one vector before computing cosine
   and relative L2, so Cas12a appeared to pass. The frozen criterion says
   "every model input", and gating each input separately exposed
   `cas12a/C3` `Input_Seq` at 0.0223 against the 0.02 bound - a real failure
   that the much more accurate `Input_Feat` gradient (1e-4) had been hiding.
   The sequence gradient is the one the screen renders per base, so pooling hid
   exactly the number that matters most.

The criteria were not relaxed. Execution completion does not establish numerical
quality; these failures keep the final required outcome at fail/1. #199/#200 own
non-NT warnings, diagnostics and explicit precision retry settings/costs, progress
and cancellation; #195 owns the Cas12a path contract. Defaults remain IG 32 and
SHAP 64/4. A larger setting must not be silently promoted to a verified default.

One further consequence needs an owner. #191 says that when XAI verification
fails, the prediction table stays and the explanation carries a warning, a
diagnostic value and a retry note - "XAI 검증에 실패하면 예측 표는 유지하고 해당
설명에 경고·진단값·재시도 안내를 표시한다". #199 (SpCas9) and #200 (Cas12a)
now own this requirement; the detailed evidence/cost handoff is in #214's report.

> **Superseded on 2026-11-19, as to the display requirement only.** The screen
> requirement above no longer holds: computable explanations are rendered with
> the previous version's graphs even when they miss the numeric tolerances, and
> no separate quality warning, diagnostic panel or retry screen is added. See
> [XAI 표시 방침](../planning/xai-exhibition-display-policy.md). Everything else
> on this page - the failures, the verdicts, the tolerances, the measurements and
> the ownership of units, cache identity, path contracts and precision cost -
> stands unchanged, and the failures are preserved in these internal records.

### One defect found and fixed

SHAP was **not reproducible at all**: `shap.KernelExplainer` samples its subsets
from numpy's *global* RNG, which `kernel_shap` never seeded, so two identical
calls returned different attributions - measured at up to 1.02 apart on a
Cas12a probability in [0, 1], and 0.22 on SpCas9. The same screen opened twice
would have shown different contributions. `xai/explain.py` now seeds the global
RNG from the existing `seed` argument around the `shap_values` call and restores
the caller's RNG state afterwards, so `seed` now fixes both the backgrounds and
the subset sampling. Repeat runs are now bitwise identical, and three tests in
`test/test_explain.py` hold that. This fix does not address finding 1 above:
reproducibility and sampling adequacy are different properties.

Two consequences to carry forward. The four precomputed NT artifacts in
`data/precomputed_shap` were generated at 64 samples / 4 backgrounds through the
unseeded path and lack calculation identity/protocol evidence. Their equality
to a fresh computation cannot be established. Non-NT instability measurements
do not establish NT quality. #208 now rejects legacy artifacts without
provenance and computes fresh results. #201 owns NT quality approval, unit
correction and coordinated regeneration; these four artifacts must not simply
be regenerated at 64/4 based on non-NT evidence. `make_precomputed_shap.py`
does not independently approve its default settings.

## Measurement scope and limits

The additive fixed-location/conditional float64 producer is documented in
[diagnostics-211.md](diagnostics-211.md). Its fixture validation is separate from
the official #212/#213 measurements.

`verification.json` records, for both models: forward parity against the
TensorFlow backend, layer-by-layer activation parity, bitwise weight placement,
IG path function match, finite-difference gradient checks, IG completeness and
resolution stability, SHAP additivity and repeat stability, process RSS for
import/load/IG, and the offline staged run.

Not established here: full UI + computation + WebView2 tree RAM (#194), the
Cas12a IG path choice and its completeness contract (#195), screen and CSV
wiring (#196, #197), a built distribution and a clean offline Windows PC
(#204), and any statement about biological correctness or held-out model
performance (#202). RSS figures come from this development PC, one process,
CPU only.

Two limits of the offline evidence specifically. The network block is an
in-process socket block, not an OS-level one, so it shows the code path needs no
network rather than proving the machine had none; and the staging tree is
assembled from this PC's own site-packages, so it shows the dependency set is
locally stageable without a network, not that a fresh machine could be
provisioned offline. The `packaging/desktop.spec` change is likewise recorded
and reasoned, **not executed** - no distribution was built from it here, and it
now also hard-fails the build if the two `.keras` files are absent, which is
deliberate fail-fast behaviour that #204 will be the first to exercise.
