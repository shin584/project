# 9-Hydra v2 inference export

[Issue #96](https://github.com/shin584/project/issues/96) adds an inference
representation of an explicitly selected checkpoint. It preserves learned values
and does not establish a performance improvement or select an operating model.

From `xai_test/`, using a Python environment with TensorFlow, Keras and NumPy:

```powershell
python -m SpCas9.spcas9_export --source SpCas9/test_run/9hydra_v2.keras --destination C:/exports/9hydra-v2-inference-001
```

The destination's parent must exist. The destination itself must be new, including
after a failed or interrupted attempt. Existing paths, source collisions and
aliases are rejected. There is no latest-run search, automatic training/export,
fallback or training-data dependency.

The same operation is available in Python:

```python
from SpCas9.spcas9_export import export_checkpoint

manifest = export_checkpoint(
    "SpCas9/test_run/9hydra_v2.keras", "C:/exports/9hydra-v2-inference-002"
)
```

The result directory contains `model.keras` and `manifest.json`. Reload with
`keras.models.load_model(path, compile=False)`, without custom objects. The input
is `In_Seq`, float32 `(N, 30, 4)`, A/C/G/T channels: four flank bases, twenty
protospacer bases, three PAM bases and three flank bases. Window extraction and
unknown-base policy belong to the caller.

`predict` returns exactly a Python list of nine float32 `(N, 1)` arrays:

| Index | Protein | Source layer |
| --- | --- | --- |
| 0 | SpCas9(WT) | `Output_SpCas9` |
| 1 | SpCas9-NG | `Output_SpCas9_NG` |
| 2 | VRQR | `Output_VRQR_variant` |
| 3 | xCas9 | `Output_xCas` |
| 4 | Sniper | `Output_Sniper_Cas9` |
| 5 | SpCas9-HF1 | `Output_SpCas9_HF_1` |
| 6 | eSpCas9(1.1) | `Output_eSpCas9_1_1_` |
| 7 | HypaCas9 | `Output_HypaCas9` |
| 8 | evoCas9 | `Output_evoCas9` |

The exporter resolves these layer identities explicitly. The training model's
dictionary outputs, builder, masking and checkpoint selection remain as before.
All heads remain linear on the published-percentage/100 scale. Finite zero,
negative and above-one predictions are preserved. Distance penalties and the
display conversion by 100 belong to the consumer.

Export validates every required head's uniqueness, shape, dtype and activation.
It saves to separate staging, reloads without compilation, and compares all heads
against source named tensors for one complete synthetic window and a batch of
three distinct windows. Tolerances are fixed at `rtol=1e-5`, `atol=1e-6`; all
verification outputs must be finite. These fixtures are independent of the
spreadsheet, split and Testset metrics.

Success requires `manifest.json` with `validation.status == "passed"` **and** a
`model.keras` whose SHA-256 matches `artifact.sha256`. The marker is published
last, after verification, source identity rechecks and artifact identity checks.
A directory, a model file, a staging directory or `manifest.pending` alone is
incomplete. Failed publication can leave incomplete files; inspect them and use
a fresh destination for retries. Neither a previous export nor the source is
overwritten. The manifest records contracts, head order, fixtures, fixed
tolerances, per-head errors and values, source/output hashes, runtime versions and
the export module's SHA-256.

## Local acceptance evidence

The agreed source is `SpCas9/test_run/9hydra_v2.keras`, SHA-256
`24e31cc6b9df8520e2bf761d33f7bb475c4658a9a479242b67db882946d0d2b8`.
[The retained acceptance record](evidence/issue96-export.json) identifies the
fresh local artifact, its verification manifest and preservation checks. The
large checkpoint and exported model stay local. A checkout without the selected
checkpoint cannot reproduce real-checkpoint acceptance; the real-checkpoint
test fails explicitly when that input is unavailable.

Local acceptance passed 137 focused export/scanner tests. Real-checkpoint CLI
parity had maximum absolute error 0.0 for all nine heads at N=1 and N=3;
45 protected source, checkpoint, split, log, evaluation and consumer files
retained their hashes. CLI exit codes were 0 for success, 1 for an existing
destination and 2 for missing explicit arguments. Independent standards and
specification reviews reported zero findings.

Run only the focused export and existing scanner contract tests from repo root:

```powershell
python -m pytest xai_test/test_spcas9_export.py xai_test/test_multi_cas_scanner.py -q
```

Controlled serializable models cover reordered dictionaries, distinguishable
heads, the HF1/eSpCas9 permutation, input-dependent learned weights and linear
values outside `[0, 1]`. Failure checks cover invalid heads/inputs, ambiguity,
nonfinite values, existing paths, hard-link source aliases, save/reload/parity
failures, interruption, source drift and publication failures. Controlled models
supplement the separately pinned real-checkpoint test.

## Consumer handoff

This is the exporter child of [#94](https://github.com/shin584/project/issues/94).
The e_system loader remains hardcoded and cached on its previous artifact;
exporting does not activate this model. Existing consumer limitations include
all-nine final scores at most -1 causing an unassigned or stale raw score,
unguarded nonfinite values, and the detailed filter's 0–100 range. No clipping or
sigmoid workaround is applied. Consumer repair, explicit model selection, cache
handling and preparation to return to retained v1 are tracked separately in
[#95](https://github.com/shin584/project/issues/95), before operational adoption.
The root scanner's dictionary-only v2 contract is unchanged. This record covers
export parity; parent #94's unchanged-consumer integration is separate evidence.
Export parity also does not establish biological accuracy across NG-discovered
sites, variant-specific PAM suitability or protein-ranking accuracy.
