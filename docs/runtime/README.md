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
