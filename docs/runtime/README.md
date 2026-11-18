# non-NT forward and gradient runtime (#193)

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
```

### The runtime decision is verified

| Claim | Measured | Frozen bound |
| --- | --- | --- |
| Forward parity vs the TensorFlow backend, all cases, all 9 + 1 heads | 6.0e-8 (SpCas9), 3.0e-8 (Cas12a) max abs | 1e-4 |
| Layer-by-layer activation parity, worst layer | 1.07e-6 at `lstm`, 1.37e-6 at `bidirectional` | 1e-4 x scale |
| Weight placement | identical variable sets, bitwise identical float32 bytes | bitwise |
| IG path function match | exactly 0.0, all 8 cases | 1e-6 |
| Cross-runtime gradient parity (torch autograd vs `tf.GradientTape`) | 2.4e-7 max abs, cosine 0.999999999999 | 1e-4 / 0.9999 |
| Finiteness, 24 checks | all finite | all finite |
| Offline, TensorFlow-free, network-blocked staged run | both models load, predict and give gradients; same forward values | pass |

The LSTM and BiLSTM layers matching to about 1e-6 *layer by layer* is what rules
out a transposed kernel or a swapped gate order, which a final-scalar comparison
alone can hide. Across both models and all 8 cases, 164 of 168 layer comparisons
sit below 1e-6, and the four that do not are the recurrent layers themselves at
1.07e-6 and 1.37e-6 - the layers where accumulation order differs most.

### RAM, in the configuration that ships

| Stage | TensorFlow absent (staging) | This PC, TensorFlow installed |
| --- | --- | --- |
| Interpreter | 20.9 MB | 18.6 MB |
| After importing the runtime | 298.8 MB | 423.7 MB |
| Both models loaded, gradients run | 313.6 MB peak | 601.3 MB peak (Cas12a) |

The gap is the `tf.available` probe, not the adopted runtime: with TensorFlow
not importable, the Cas12a BiLSTM path peaks at 313.6 MB with **both** models
held at once, where the shipped design holds only the selected family. Bundle
inputs for the same decision: Keras 13.7 MB of site-packages against
TensorFlow's 1408.4 MB, with torch's 492 MB already shipped.

### Three failures stand, and they are not about the runtime

1. **SHAP at the user's default settings is not a stable estimate.** With the
   seed fixed, 64 samples over 4 backgrounds reproduces exactly (see below), but
   changing the background draw changes the explanation: pairwise correlation
   across seeds 0/1/2 runs from -0.13 to 0.68 against a frozen bound of 0.9, and
   the default disagrees with a 512-sample / 32-background reference at
   correlations of 0.22 to 0.82. All 24 of these checks fail. The positions a
   user would read as "most important" are therefore not stable at 64/4.
2. **IG completeness at 32 steps misses the frozen bound in 5 of 16 cases**
   (worst 4.1e-2 against 5e-3 for SpCas9 S3; Cas12a C2 4.4e-3 against 1e-3). At
   256 steps every case passes, worst 1.3e-3. Per-position resolution is fine
   either way: correlation between 32 and 256 steps is at least 0.9992, and one
   case (SpCas9 S2) misses the scaled max-difference bound by 0.5 percent.
3. **The coordinate-wise finite-difference gradient check fails in 9 of 32
   case/point combinations** (worst relative L2 0.264, cosine 0.969). The
   registered diagnostics then showed cross-runtime gradient parity at 2.4e-7
   with cosine 0.999999999999, directional finite-difference median relative
   error of 3.2e-4 to 2.4e-3, and hundreds of ReLU/LeakyReLU sign flips inside
   the h = 1e-2 coordinate steps. So the gradients are right and the coordinate
   difference quotient was the wrong instrument at the kinks; 3 of 16 directional
   maxima still exceed 1e-1, consistent with individual kinked directions. The
   original failure is kept on the record rather than relabelled.

None of the three blocks the runtime adoption, and none was relaxed. 1 and 2 are
decisions for the user, because #191 forbids raising the user-facing compute
without saying so: raising IG to 256 steps is 8x the IG work, and a SHAP setting
that is actually stable costs more again. #195 through #197 should not wire the
screens to 64/4 and 32 steps as if they were verified; the numbers above are
what those tickets need to argue from.

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
unseeded path, so they are one unreproducible draw from the unstable
distribution that finding 1 describes; a cache hit and a fresh computation of
the same window will not agree. They are not wrong in any new way - that
mismatch existed before, invisibly - but whoever acts on finding 1 should
regenerate them under the settings that get chosen. And `make_precomputed_shap.py`
carries the same 64/4 default, so it inherits the decision rather than making it.

## Measurement scope and limits

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
