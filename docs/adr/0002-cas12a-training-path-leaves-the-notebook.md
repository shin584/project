# The Cas12a training path leaves the notebook

`cas12a/model.py` and `cas12a/train.py` now hold the Cas12a model definition and
the 5-fold loop, and `python -m cas12a.train` runs training end to end on a
local CPU. `cas12a/model/Cas12a_model_final.ipynb` keeps a Colab wrapper that
mounts Drive and calls that same entry point.

**This reverses #91's decision.** #91 moved the feature functions and the loader
into `cas12a/` but deliberately left the model and the training loop in the
notebook, on the grounds that the notebook was the only place training had ever
run. That reasoning held until the consequence became concrete: with the
training path behind `google.colab.drive.mount`, #92 — which exists to fix the
training configuration the review's P1 and P2 describe — could not be worked by
an agent at all, and was labelled `ready-for-human` for that single reason. The
cost of the split was not duplication; it was that the part of the code most in
need of change was the part nothing in the repo could reach.

The measured trade-off is a real one, and it is not free.

**Given up: GPU access.** Colab supplies a GPU; this machine does not have one.
TensorFlow 2.21.0 here reports an empty `list_physical_devices('GPU')` and
`is_built_with_cuda()` False, and TensorFlow dropped native-Windows GPU support
at 2.11 — so local training is CPU-only, and will stay that way short of WSL2.

But measured, the GPU turns out not to be worth much here. On the full Cas12a
library (15,000 usable rows, 34 timesteps, 96,513 parameters) on an i5-8265U,
one epoch takes **about 10 seconds** and the whole 5-fold run **257.5s — four
and a quarter minutes**. The model is small and the sequence is short; there was
less for a GPU to do than the Colab habit suggested.
`cas12a/docs/cas12a-training-cost.md` has the per-fold numbers. That makes the
cost objection to running #92 locally disappear: raising `epochs` to 50 and
attaching `EarlyStopping`, which is most of what #92 is for, lands in the tens
of minutes at worst.

**Gained: the training path is reachable.** It runs from a checkout with no
Drive mount and no `/content/...` path, the dataset location is an argument
defaulting to `cas12a/dataset/`, and the parts of training that are
deterministic are now under test — the fold indices and the layer table, in
`cas12a/test/test_train.py` and `test_model.py`. None of that was testable while
the code lived in a notebook cell.

## Consequences

- **The port changed no training setting.** `epochs=5`, `KFold(5, shuffle=True,
  random_state=42)`, `EarlyStopping`/`ReduceLROnPlateau` still unattached, and
  `fold_results` still returned empty. Each is something the review wants
  changed; changing any of them in the same commit as the port would make a
  later performance difference impossible to attribute to one or the other.
  #92 owns them, and the tests above pin them so that #92's edits are visible.
- **Bit-exact parity with the notebook's training is not available** and was not
  attempted: weight initialisation is random and CPU and GPU kernels do not
  agree bit for bit. What is verified instead is the deterministic part —
  preprocessing (already shown bit-exact on all 15,000 rows in #91), the fold
  indices, and the layer-by-layer structure and 96,513-parameter count. The
  notebook's reported `val_loss 0.0481` (fold 4, epoch 5) is a sanity reference
  only; nothing asserts against it. For the record, the ported path's five folds
  ended between 0.0433 and 0.0467, so the same order of magnitude.
- **The review's P2 reproduces here**, which is a sign the port is faithful
  rather than a new problem: in four of five folds the last epoch is not that
  fold's best, and with no `restore_best_weights` those are the weights that
  survive. #92's territory, recorded in the cost doc.
- **The notebook is not deleted.** It stays until #92 reports metrics from the
  ported path that the project is willing to adopt, because until then Colab is
  the only training path that has produced a reported number. It is now a
  wrapper — the model and loop are imported, not copied, so the drift risk #91
  removed from the feature functions does not reappear here. The wrapper has not
  been re-verified on Colab (neither had #91's change to it); `REPO_ROOT` and
  `base_path` must be pointed at the actual clone and dataset.
- **`e_system` still carries its own copy of the feature functions.** The port
  does not change that, and `cas12a` must keep importing nothing from
  `e_system`; the dependency, when it is wired, points `e_system` -> `cas12a`.
  `cas12a/docs/cas12a-model-review.md` §5 tracks it.
- **Reversing this is cheap**, which is the main reason it is safe to decide
  now. `cas12a/train.py` is a module with a `--dataset` argument; running it on
  Colab means a mount and an import, which is exactly what the notebook wrapper
  already does. The expensive direction would have been the other one — writing
  #92's configuration work against notebook cells and then porting it.

## Considered Options

- **Leave training in the notebook, work #92 by hand** (#91's decision):
  rejected. It keeps GPU access and costs nothing to implement, but it leaves
  the under-trained, unmeasured part of the model as the one part no agent can
  touch, and every change to it stays untestable and unreviewable as a diff.
- **Port to a module and delete the notebook**: rejected for now. It is where
  this should end up, but deleting it before #92 has produced adoptable numbers
  from the ported path would leave the project with no training path whose
  output anyone has seen. The deletion is #92's call.
- **Port to a module targeting Colab GPU only** (module in the repo, run only
  from a mounted notebook): rejected. It gets the reviewability and the GPU, and
  it is the fallback if the measured CPU cost turns out to be impractical — but
  as the default it keeps every run gated on a human mounting Drive, which is
  the thing that blocked #92.
- **WSL2 with GPU support**: not evaluated. It would recover the GPU locally and
  is the obvious move if CPU cost proves to be the binding constraint; it is a
  change to the development environment rather than to this repo, so it is out
  of scope here and noted as the next thing to try.
