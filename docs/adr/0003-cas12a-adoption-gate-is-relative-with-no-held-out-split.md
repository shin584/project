# The Cas12a adoption gate is relative, with no held-out split

Whether the Cas12a CNN+BiLSTM is adopted is decided by **one comparison**: its
5-fold mean Spearman must exceed RidgeCV's 0.738 — same features, same
`KFold(5, shuffle=True, random_state=42)` split
(`cas12a/docs/cas12a-model-review.md` §2) — **by at least one fold standard
deviation.** `cas12a/metrics.py` holds the baseline constant and
`gate_verdict`, and `cas12a/artifacts/cas12a_5fold_metrics.json` records the
verdict of each run.

Three things follow from the gate being *relative*, and they are the decision
this ADR exists to record.

**No held-out test split.** Five-fold cross-validation over all 15,000 usable
rows is the whole of the evaluation. The usual reason to hold rows back is that
a number reported on data the model selected against is optimistic — but both
sides of this comparison are computed the same way on the same folds, so the
bias applies to the ridge baseline too and cancels in the margin. #91 verified
there are no duplicate sequences, so there is no leakage path between folds
beyond the one every cross-validation has. Spending 20% of the rows to measure
an absolute number nothing consumes would make the comparison that *is*
consumed noisier.

**Strict op determinism is not enabled.** Python, NumPy and TensorFlow seeds are
fixed (`cas12a.train.set_seeds`, `SEED = 42`, per-fold `SEED + fold`), so a
fold's initialisation is reproducible and the library versions that produced a
run are written into the JSON. `tf.config.experimental.enable_op_determinism()`
is deliberately **not** called. The gate's threshold is the fold standard
deviation, so run-to-run jitter smaller than that spread is already inside the
decision's tolerance; buying bit-identical reruns at the cost of slower
CPU-only training would be paying for a precision the gate does not read. What
the artifact promises is therefore **traceability, not bit-reproducibility**:
the JSON beside a `.keras` says which numbers that file scored.

**DeepCpf1's ≈0.77 is a reference line, not a second gate.** Kim 2018 split and
preprocessed differently, so falling short of it cannot be attributed to this
architecture rather than to the conditions. `metrics.py` keeps it under
`reference_only` so it cannot quietly become a pass/fail threshold.

## Consequences

- **Spearman decides, and MAE plus calibration are reported beside it.** Rank
  correlation is what the field reports and what the ridge baseline is quoted
  in, but the app compares *absolute* scores across Cas types, so a model that
  ranks correctly at the wrong level would pass a rank-only gate and still
  break cross-type comparison. `calibration_slope`, `calibration_intercept` and
  `mean_bias` are in every fold's record for that reason, and
  `cas12a/test/test_metrics.py` pins the case they exist to catch — a uniformly
  shifted prediction scoring Spearman 1.0.
- **A tie goes against the model.** `margin >= std` means a mean that beats
  0.738 by less than the folds disagree among themselves does not pass. This is
  intentional and it is the common case to expect: §2 found this architecture
  performing at ridge level, so a small positive margin is the likely outcome
  and it is not evidence of a better model.
- **There is a contamination trigger, and it must be acted on when it fires.**
  The moment this gate is missed and the response is to try another
  architecture, these fold metrics start selecting the architecture — and a
  number used for selection no longer estimates performance. **The second
  architecture attempt is when a held-out split has to be introduced**, before
  it is run and not after. One retune measured against the folds is not
  selection; a search is.
- **A failed gate does not mean the model is broken.** The review's §2 already
  established that it learns signal — MSE less than half the mean predictor's.
  Failing means the CNN+BiLSTM does not earn its complexity over a linear
  model, which points at P6 (the 352-dim dinucleotide features are fully
  derivable from the one-hot sequence the CNN already reads) and at
  `GlobalAveragePooling1D` averaging away the BiLSTM's positional information.
  Those are separate issues, opened on a failure, not fixes to smuggle into the
  run that measured it.
- **Changing the split invalidates the gate**, not just the model's numbers. The
  0.738 was produced on `random_state=42`; `cas12a/test/test_train.py` pins the
  fold indices by digest so that a change to them is a deliberate edit to a
  test. Re-deriving the ridge baseline is the price of changing the split.
- **The per-fold JSON is the single source of these numbers**, not a console log
  and not notebook output. #92's second issue comment is the reason: an executed
  notebook's stored training logs cost a later reader tens of thousands of
  tokens to read past and carry nothing the JSON does not.

## Considered Options

- **Hold out 20% before any training, report on it** — rejected for now, for the
  reasons above: the comparison is relative, so the held-out number would be
  more honest in absolute terms while making the only decision anyone acts on
  noisier. It is the right move the moment architecture search starts, which is
  the trigger recorded above.
- **Gate on beating DeepCpf1's 0.77** — rejected. It is the number a reader
  will reach for, but a miss would be unattributable between architecture and
  the different split and preprocessing, which makes it useless as a decision
  rule even though it stays useful as context.
- **Gate on mean Spearman alone, ignoring fold spread** — rejected. It would
  pass a 0.74 mean whose folds range from 0.68 to 0.82, which establishes
  nothing about a 0.002 win over a linear model.
- **Enable `enable_op_determinism()` and report bit-reproducible runs** —
  rejected. It is the stronger reproducibility claim, and worth revisiting if
  the project ever needs to defend an exact number rather than a comparison,
  but it slows CPU-only training for a precision below the gate's own
  resolution. Fixed seeds plus recorded library versions are what the artifact
  needs to be traceable.
- **Gate on MSE or R² instead** — rejected. Both are reported, and the ridge row
  has them, but the field reports Spearman for guide-efficiency models and the
  app's ranking of candidate sites is what users actually see.
