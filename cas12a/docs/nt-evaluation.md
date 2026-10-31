# NT-Cas12a final Testset evaluation

Issue [#107](https://github.com/shin584/project/issues/107) evaluates the frozen,
reload-verified selected export on the reserved NT-Cas12a Testset. Continue the
same run after [durable selected export](nt-durable-export.md):

```python
run = workflow.verify_durable_export(run)
run = workflow.evaluate(run)
assert run.status["state"] == "final_evaluation_validated"
assert not run.status["final_evaluation_completed"]
run = workflow.persist_evaluation(run)
run = workflow.verify_durable_evaluation(run)
assert run.status["final_evaluation_completed"]
```

Production evaluates exactly the 1,500 reserved targets in the saved partition's
order. Before inference it rechecks data/partition identities, durable evidence,
local selected weights and matching tokenizer, and fresh validation-sample
reload predictions. Inference uses the frozen execution device, precision and
evaluation microbatch. It never creates an optimizer, chooses a checkpoint,
changes warmup/batching, fits a baseline or changes the clipping policy. Model,
tokenizer and frozen evidence hashes are rechecked after inference too. Existing
evaluation destinations are preserved; final results do not initiate another fit.

The local `final_evaluation/` bundle contains:

- `predictions.csv`: source row, original source identity/metadata as JSON,
  sequence and its SHA-256, protospacer, target label, native `raw_prediction`,
  `bounded_efficiency` clipped to [0, 1], and `baseline_train_mean` prediction.
- `metrics.json`: separate `raw`, `clipped` and `baseline_train_mean` namespaces,
  target count, diagnostics, frozen baseline fit provenance and model/partition
  identities. Each namespace includes Spearman, Pearson, MAE and MSE.
- `evaluation.json`: validated/failed outcome, failure reasons, ordered Testset
  source rows, fixed prediction policy, comparison caveat, timestamp and SHA-256
  links to both files and the selected checkpoint/artifact/partition identities.

The baseline is the same training-label mean frozen during preparation, with its
training count and training-partition hash. It predicts that constant on exactly
the model's final targets. Its expected constant output is never model collapse.
Finite constant predictions retain ordinary MAE/MSE. Undefined correlations use
JSON `null` with `spearman_undefined_reason`/`pearson_undefined_reason`, including
constant labels, constant predictions or fewer than two targets.

Diagnostics include each view's minimum, maximum, population variance,
out-of-range count, finite checks/nonfinite count and constant flag.
`raw_collapse` distinguishes constant native outputs from
`clipped_only_constant`, where a varying raw model saturates the bounded view.
Clipped-only constant output and valid low metrics are reportable. Raw collapse
or nonfinite outputs record `final_evaluation_failed` and retain local numerical
evidence with `validated: false`; publication is refused. JSON never contains
NaN/Infinity. Failed nonfinite predictions remain visible in CSV, while error
metrics that cannot be computed are `null`. Missing or changed identities also
fail explicitly without claiming successful model completion.

`evaluate` validates local evidence only. `persist_evaluation` checks the report
identity, selected durable export, file hashes and verified copied bundle before
recording `durable_evaluation.json` and final durable completion. Reverification
revokes completion if saved evidence changes. Fixture runs always keep
`model_completed: false`; passing offline tests does not establish a trained
production NT model.

The historical CNN five-fold mean Spearman of approximately 0.751 is historical
context only. Its evaluation population/procedure differs from this NT holdout,
so the values are not directly comparable. Historical CNN checkpoint training
populations overlap the NT holdout: applying those checkpoints would not give a
leakage-free comparison. Every final report preserves this caveat. There is no
CNN/Ridge superiority gate or train-mean error threshold.

`cas12a.nt_evaluation.prediction_metrics` can also compute JSON-safe metrics
without importing PyTorch, Transformers or TensorFlow. Offline workflow tests use
the real tiny saved model/tokenizer and controlled external model outputs for
boundary/degenerate cases. Run `python -m pytest
cas12a/test/test_nt_evaluation.py -q`, then `python -m pytest cas12a/test -q`.
Actual NT-500M GPU training, final Testset measurements and mounted Drive
execution remain **unexecuted** until a real Colab run saves that evidence.
