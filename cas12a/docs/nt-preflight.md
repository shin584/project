# NT-Cas12a model validation and resource preflight

After [run preparation](nt-preparation.md), use the same public workflow:

```python
from cas12a.nt_workflow import ModelSource

# `workflow` and `run` come from NTWorkflow(...).prepare().
run = workflow.preflight(run, model=ModelSource())
assert run.status["state"] == "preflight_passed"
assert run.status["model_completed"] is False

# The later training stage must use this fresh initialization, never a model
# retained from a preflight attempt. This call performs no training updates.
initial = workflow.initialize_fit(run)
```

Production resolves `InstaDeepAI/nucleotide-transformer-500m-human-ref` to a
40-character immutable commit, then loads its configuration, tokenizer and
generic encoder weights at that same revision. Pass `ModelSource(revision=...)`
to resolve a specific commit or reference. The standard
`EsmForSequenceClassification` head has one unconstrained regression output and
MSE loss. Every encoder/head parameter remains trainable. There is no adapted
SaCas9 initialization, NTv2 substitution, LoRA, encoder freezing or sigmoid.

The actual tokenizer must reconstruct every usable 34 bp input exactly after
excluding its special-token IDs. Unknown tokens, missing bases and inputs longer
than 12 token positions fail before training. Only then is each input padded to
12 positions with `truncation=False`. Testset sequences participate in this
representation check; Testset labels never enter the training/validation probe.

Production requires CUDA. Native BF16 is preferred when the GPU and runtime
support it; T4 uses FP16 with gradient scaling. CPU float32 is available only
for an explicitly marked fixture run. The probe begins with training microbatch
32, accumulation 1 and evaluation microbatch 32. On an out-of-memory error,
training microbatch halves through 16/8/4/2/1 with accumulation `32/microbatch`.
Evaluation OOM halves evaluation microbatch independently. After training
microbatch 1 fails, gradient checkpointing is enabled for the final memory-saving
attempt. Failure there raises an explicit `ResourceConstraint`; it does not
change the backbone or training intervention.

Each candidate reloads the generic initialization and seeds Python, NumPy and
PyTorch with 42. It creates a new AdamW optimizer, linear warmup/decay scheduler
and precision scaler, runs two complete effective-batch-32 optimizer updates,
and validates up to one evaluation microbatch. The second update exercises
forward/backward after Adam moments have been allocated. Losses, predictions,
gradients and updated parameters must be finite. Microbatch MSE contributions
are weighted by example count. The schedule uses optimizer updates, at most
10 epochs and a 0.05 warmup ratio; production has 3,750 planned updates and
188 warmup updates. Candidate selection uses feasibility, never validation-loss
ranking or final-Testset performance. No GPU fit or speed is guaranteed.
FP16 gradient overflow triggers the scaler's native skip/backoff, up to 32
overflow retries; skipped steps do not advance the scheduler or count toward
the two required optimizer updates. Persistent nonfinite gradients fail.

The successful candidate freezes execution settings. `initialize_fit` checks
source/partition evidence, matching local/durable artifacts, tokenizer and
initialization hashes, and unchanged GPU/runtime/package settings before
returning a fresh model, optimizer, scheduler and scaler at global step zero.
It never reuses preflight updates, optimizer moments or RNG progression. A
passed preflight cannot silently be rerun to select another configuration;
create a separate recorded run to repeat it.

Both run destinations receive verified matching evidence:

| Artifact | Evidence |
| --- | --- |
| `tokenization.json` | Complete-input count, actual unpadded length histogram, vocabulary size/hash, tokenizer class and special-token metadata |
| `preflight.json` | Training/validation identities, candidate settings/outcomes, initial-weight hashes, completed updates, post-Adam backward check, measured losses, timings and CUDA peak allocated/reserved bytes |
| `execution.json` | Frozen model revision, trainability counts, precision/device/batch/checkpointing settings, actual package versions, resolved AdamW/scaler/scheduler settings and evidence hashes |
| `status.json` | `preflighting`, `preflight_passed` or `preflight_failed`, configuration/execution hashes and `model_completed: false` |

CUDA peaks include optimizer-state allocation and validation, with a reset
between attempts. CPU fixture memory fields are null, not GPU measurements.
Failures record their error and surviving evidence in both destinations where
storage is writable. Checkpoints and model weights are not written to Drive
by this stage.

For offline checks, prepare a synthetic fixture split, save a tiny local ESM
masked-language model and matching six-mer tokenizer, then call
`workflow.preflight(run, model=ModelSource.fixture(path))`. This uses the same
public workflow and real forward/backward/optimizer operations. Production runs
reject fixture weights/tokenizers. CPU fixture evidence is labeled
`offline_fixture` with `gpu_preflight_executed: false`; it does not establish
full-library original-tokenizer validation, GPU compatibility or trained-model
completion. A tiny CUDA fixture still does not establish production-model fit.
Its evidence is labeled `gpu_fixture`, separately from `production_gpu`.

Run `python -m pytest cas12a/test/test_nt_preflight.py -q`, then
`python -m pytest cas12a/test -q`. The CUDA integration check skips explicitly
when CUDA is unavailable. The implementation was checked locally with Python
3.13.7, PyTorch 2.12.0+cpu, Transformers 5.8.1, tokenizers 0.22.2,
huggingface-hub 1.15.0, safetensors 0.7.0 and NumPy 2.3.2. This is CPU fixture
evidence, not a validated Colab CUDA dependency set. The real Colab session must
pass the production preflight and retain its actual compatible dependency and
GPU evidence before fit. This ticket provides preflight and fresh initialization;
full training, checkpoint selection and final evaluation are later stages.

API references: [original NT configuration](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref/blob/main/config.json),
[ESM regression model](https://huggingface.co/docs/transformers/model_doc/esm#transformers.EsmForSequenceClassification),
and [PyTorch mixed precision](https://docs.pytorch.org/docs/stable/amp.html).
