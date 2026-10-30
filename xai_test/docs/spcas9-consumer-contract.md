# Unchanged e_system consumer contract (#97)

[Issue #97](https://github.com/shin584/project/issues/97) verifies the existing
consumer through the model argument of its actual imported `run_prediction`.
The [retained observations](evidence/issue97-consumer/report.json) include the
runtime, inputs, controlled outputs, per-protein results, candidate results,
exceptions, source hashes and complete arrays received by the model argument.
The accompanying `source/` directory preserves the five consumer source files
verbatim. `verification-code.py` preserves the exact verifier identified by the
report's code hash. Final review additionally tightened the current CLI so
rejected ordinary finite cases cannot publish verified evidence; the retained
observations already contain complete successful finite cases. No e_system
source, configuration, operating artifact, loader, cache or UI was changed.
The #97 characterization loads neither SaCas9/Cas12a models nor a real SpCas9
checkpoint. The separately reported broader subtree tests include existing
checkpoint integration checks.

## Reproduction

Use a Python environment with NumPy, TensorFlow, Keras, PyTorch, Streamlit and
pytest. The consumer source and dependencies are required; missing inputs fail
explicitly rather than becoming skipped acceptance. From `xai_test/`:

```powershell
python -m SpCas9.spcas9_consumer_check --source ../e_system --destination C:/exports/consumer-observations-001 --prior-evidence docs/evidence/issue97-consumer/prior-review-evidence.json
```

The destination's parent must exist and the destination must be new and outside
the consumer directory. Failed attempts must use another destination. Only a
`report.json` with `status == "verified"` records completed checks; snapshots
alone do not. Earlier results are never overwritten. This is a synchronous
offline check: importing e_system temporarily isolates its absolute sibling
imports, restores the caller's imports and disables bytecode writes during that
import. The app and `load_models` are never invoked.

For historical reproduction, supply `--source docs/evidence/issue97-consumer/source`.
For current-source verification, supply the current local e_system directory.
Current selection behavior is classified separately as `known_defect_present`
or `fixed`; unexpected or partial repairs fail explicitly. A corrected future
consumer is accepted without requiring the historical bug to return. Tests also
exercise an illustrative finite-selection repair in a disposable source copy;
that fixture neither implements nor validates the #95 follow-up.

## Positional values, encoding and scale

| Index | Consumer protein | Controlled raw value |
| --- | --- | ---: |
| 0 | SpCas9(WT) | 0 |
| 1 | SpCas9-NG | -0.25 |
| 2 | VRQR | 1.5 |
| 3 | xCas9 | 0.3 |
| 4 | Sniper | 0.4 |
| 5 | SpCas9-HF1 | 0.56 |
| 6 | eSpCas9(1.1) | 0.67 |
| 7 | HypaCas9 | 0.8 |
| 8 | evoCas9 | 0.9 |

All nine results retain the supplied raw values, including zero, negatives and
values above one. In particular, HF1 and eSpCas9 remain distinct and correctly
ordered. At distance 10, Gaussian penalty is approximately 0.6065306597;
the 1.5 raw value becomes approximately 90.979599 after multiplication by 100.
Every final score is checked against an independent exponential calculation,
and the tests use literal expected values. No clipping or sigmoid is applied.

The actual `one_hot_encode_dna` emits an integer NumPy array, observed locally as
`int64`, reaching `model.predict` as `(1, 30, 4)` with A/C/G/T channel order.
Each observation records the exact window and its SHA-256. The 30 bases are four
left-flank bases, twenty protospacer bases, three PAM bases and three right-flank
bases. SpCas9's configured two-base NG match occupies positions 24–25; the third
PAM base is position 26 of the model window. A deliberately unusable
`CandidateSite.encoded_seq` confirms this path uses neither cached encoding nor
the separate scanner encoder. No test-side float32 conversion is performed.

The supplied model is a controlled Python object returning NumPy arrays. These
observations **do not verify Keras integer-input acceptance, checkpoint parity,
model accuracy or trained-model trigger frequency**. Actual checkpoint
save/reload and consumer parity belong to [#98](https://github.com/shin584/project/issues/98),
using the export from [#96](https://github.com/shin584/project/issues/96).

## Negative-score defect and retained history

The retained source initializes `best_score` to -1 and assigns `best_raw` only
when a head's final score is strictly greater. The trigger is **all nine**
`raw * gaussian_penalty(distance) * 100 <= -1`, not one negative head.

| Scenario (distance zero) | True final scores | Observed candidate result |
| --- | --- | --- |
| First candidate: nine raw values -0.02 | Nine values -2 | `UnboundLocalError` after nine variant results |
| First candidate: nine raw values -0.01 | Nine values exactly -1 | Same `UnboundLocalError` |
| Normal candidate: nine raw values 0.2 | Nine values 20 | raw 0.2, final 20 |
| Next candidate: nine raw values -0.02 | Nine values -2 | stale raw 0.2, fabricated final -1 |
| Next candidate: nine raw values -0.04 | Nine values -4 | stale raw 0.2, fabricated final -1 |
| Zero maximum tied between first two heads | Maximum 0 | raw 0, final 0 |

Per-variant results preserve their actual negative raw/final values even where
the candidate summary crashes or becomes stale. The mixed-sign distinct-head
case succeeds, demonstrating why one negative head alone proves no defect.

The original AST-based review record is preserved byte-for-byte as
`prior-review-evidence.json`; its original Python version, source hashes and
limited verification mode remain explicit. Its predict/scanner hashes match
the source used for this run. It did not load TensorFlow, verify NumPy dtype or
run a real checkpoint. The new report independently identifies the current
source and selection status. No claim is made that an actual checkpoint emits
the controlled trigger values.

## Existing limits and handoff

Actual `scan_pam` calls demonstrate that NG discovery includes model windows
with AGG, AGA, AGC and AGT PAMs. This discovery policy applies to SpCas9-family
candidates; it does not establish each protein's biological PAM compatibility
or the accuracy of ranking proteins against one another.

Historical NGG/non-NGG performance evidence remains in
`docs/planning/evidence/9hydra-pam-group-baseline.md`: measured SpCas9 n=1,907
NGG and n=3,114 non-NGG, descriptive Spearman approximately 0.514616 and
0.829371 respectively. Thus prior validation was not NGG-only. Those existing
population summaries do not validate arbitrary e_system discoveries, all
individual protein/PAM combinations or protein ranking. No Testset metrics
were recomputed for #97.

The current consumer does not reject nonfinite head values. Controlled NaN,
positive infinity and negative infinity are retained in variant results;
positive infinity can become the winning candidate score. The JSON report
represents these as `NaN`, `Infinity` and `-Infinity` strings. Export-time
finite checks do not supply runtime validation. Invalid-output policy and
consumer repair belong to [#95](https://github.com/shin584/project/issues/95).

The app's detailed-search slider permits only final scores from 0 through 100,
inclusive. Its comparison excludes finite negative scores and scores above
100; NaN fails the comparison and infinities fall outside the bounds. A raw
value above one can still have a final score below 100 after distance penalty.
This filter does not clip stored predictions or define the other tabs' scope.

`load_models` hardcodes `models/Multi-Cas_1IN_9Hydra_Divide_Testset.keras` and is
decorated with Streamlit `cache_resource`. Export success or an artifact's
existence does not select or activate it. #94/#96–#98 own export and unchanged
consumer compatibility evidence. #95 separately owns consumer repair and
deliberate model-selection, cache-refresh and return-to-v1 preparation before
operational adoption. No model switch, cache clearing, rollout or rollback was
performed here.

Run the focused checks from the repository root:

```powershell
python -m pytest xai_test/test_spcas9_consumer.py -q -p no:cacheprovider
```

These checks use the already agreed model-argument seam, actual consumer source,
and independent expected encoding, head order and score calculations. Frozen
source checks preserve historical reproducibility; current-source checks accept
correct negative selection. Evidence source bytes are exempt from Git newline
conversion so recorded SHA-256 values survive checkout.

Final focused verification passed all nine checks. The independent
[standards/spec review](evidence/issue97-consumer/review.md) has no outstanding
findings. [Validation and preservation details](evidence/issue97-consumer/validation.json)
record the broader subtree run separately, including failures outside the new
consumer test file, and the protected source/artifact/evidence checksums.
The [final commit checks](evidence/issue97-consumer/commit-validation.json) record
the nine passing focused checks after hook formatting and the passing Ruff
hooks. A local `.ruff.toml` excludes only the byte-preserved Python artifacts;
the implementation and tests remain subject to both hooks.
