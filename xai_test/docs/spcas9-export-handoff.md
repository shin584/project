# Real 9-Hydra v2 export and consumer handoff (#98)

The [#98 acceptance record](evidence/issue98-export-consumer/report.json) binds
one exported artifact to the agreed manual-run checkpoint, its saved/reloaded
parity checks, actual unchanged e_system inference and preservation evidence.
It completes the real integration work following [#96](spcas9-inference-export.md)
and [#97](spcas9-consumer-contract.md) under
[#94](https://github.com/shin584/project/issues/94).

## Artifact and provenance

The explicit source is `SpCas9/test_run/9hydra_v2.keras`, SHA-256
`24e31cc6b9df8520e2bf761d33f7bb475c4658a9a479242b67db882946d0d2b8`.
Its training log completed at 2026-10-29 12:31:36 UTC. No latest-by-name search
was used; `test_run2` and the root scanner's older acceptance checkpoint retain
their original identities.

The retained local output is
`C:/project/MC_project/.scratch/issue98-acceptance-handoff/export/model.keras`,
SHA-256 `ddbf86aa34fe12cd6b63a1a556e602e9ef34275a9356fb57ae99247c4472ce65`.
The [matching manifest](evidence/issue98-export-consumer/manifest.json) records
the exact source, ordered protein-to-layer mapping, fixture windows/checksum,
fixed tolerances, per-head errors, runtime and export-code identity.
The model file remains local; the repository retains its verification evidence.
Missing checkpoint, e_system source or dependencies prevents real acceptance.
Synthetic passing tests or skips do not replace these inputs.

The complete synthetic A/C/G/T windows have 4 flank, 20 protospacer, 3 PAM and
3 flank bases. Export's public operation saved and reloaded a fresh inference
artifact without custom objects, then checked N=1 and N=3 against named source
heads at `rtol=1e-5`, `atol=1e-6`. Every output was exactly a nine-element Python
list in WT-through-evo order, with float32 `(N, 1)` arrays. All nine heads had
maximum save/reload absolute error **0.0** in both batches.

The actual imported `run_prediction` then received that reloaded Keras instance
directly as its SpCas9 model argument. A transparent observer on `predict`
recorded and forwarded the exact array without a cast or output substitution.
The consumer's real `one_hot_encode_dna` emitted **int64 `(1, 30, 4)`** arrays;
Keras accepted them at its declared float32 input. The separate cached scanner
encoding was deliberately unusable. All 27 protein results matched source named
predictions within the fixed tolerance; the largest absolute difference was
`5.960464477539063e-08` (single consumer calls versus the source batch).

Distances 0, 10 and 20 exercised independent Gaussian factors 1,
0.6065306597126334 and 0.1353352832366127. The record retains every raw value,
penalized raw value, multiplication by 100, expected/displayed final value and
winning candidate result. These fixtures succeeded under the unchanged
consumer's selection rule. They do not measure the real model's negative-score
defect trigger frequency.

Runtime: Python 3.13.7, TensorFlow 2.21.0, Keras 3.14.1, NumPy 2.3.2,
PyTorch 2.12.0 and Streamlit 1.51.0 on Windows. All 78 protected checkpoint,
training-source, log, split, evaluation, operating-model, scanner, consumer and
prior-evidence files retained their SHA-256. Source size, modification time,
device, inode and SHA-256 were rechecked at the end. No training, optimizer
update, partition regeneration or new Testset performance evaluation ran in
acceptance.

## Reproduce and load

From `xai_test/`, with the listed runtime dependencies and local inputs:

```powershell
python -m SpCas9.spcas9_export_acceptance --source SpCas9/test_run/9hydra_v2.keras --consumer ../e_system --destination ../.scratch/issue98-acceptance-next
```

The destination parent must exist; each attempt needs a fresh destination outside
the protected checkpoint and consumer directories. A failed attempt can leave
export-only evidence but cannot publish a successful acceptance `report.json`.
The acceptance command pins #98's agreed source; use the reusable
[#96 exporter](spcas9-inference-export.md) for explicitly selected future sources.

```python
from pathlib import Path
import hashlib
import json
import keras

directory = Path("../.scratch/issue98-acceptance-handoff/export")
manifest = json.loads((directory / "manifest.json").read_text())
artifact = directory / "model.keras"
assert (
    hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["artifact"]["sha256"]
)
model = keras.models.load_model(artifact, compile=False)
```

This obtains an inference model for explicit offline use. It does not invoke the
app's loader or change an operating model. Keep `model.keras`, `manifest.json`
and its acceptance report together when handing the artifact to #95.

## Consumer limitations and #95 prerequisites

The [published follow-up #95](https://github.com/shin584/project/issues/95) links
back to #94; #94 also links to #95. The
[read-only live linkage check](evidence/issue98-export-consumer/followup-links.json)
records both references. No duplicate follow-up was created, and neither the
parent nor earlier completed issues was modified or closed.

The current consumer source hashes match the
[preserved T2 source](evidence/issue97-consumer/source/predict.py); the acceptance
record identifies both. If current source changes, the command verifies the
preserved T2 source instead and records both identities. It does not require a
future #95 implementation to retain the historical defect.

The [T2 observations](evidence/issue97-consumer/report.json) remain a separate
controlled-model characterization. When **all nine** `raw * Gaussian * 100 <= -1`,
selection initialized to -1 never assigns the current best raw score. A first
such candidate raises `UnboundLocalError`; after a valid candidate, another
can retain the previous raw score with a fabricated final -1. One negative
head alone does not imply the defect. Controlled nine-head raw values -0.02
at distance zero produced -2 and demonstrated the failure; this does not claim
that the agreed checkpoint produces those values for a particular sequence.
Unguarded nonfinite handling and the detailed filter's inclusive 0–100 range
also remain limitations. Valid linear zero, negative and above-one values are
preserved without clipping or sigmoid.

`load_models` still hardcodes `models/Multi-Cas_1IN_9Hydra_Divide_Testset.keras`
under Streamlit `cache_resource`. Export alone does not select or refresh it.
Before operational adoption, #95 must repair selection, settle invalid-output
handling, verify explicit artifact selection and cache refresh, and prepare a
verified return to the retained v1 artifact. This handoff requires #95's existence
and reciprocal links, without waiting for #95 to finish.

The [validation record](evidence/issue98-export-consumer/validation.json) separates
the successful real acceptance CLI and focused integration check from the full
`xai_test/` subtree result: 804 passed, 7 skipped, 8 failed and 13 errors in
29m13s. Every failed/error node matches the prior #97 subtree record; no new
failure entries were introduced. The acceptance integration check passed without
a skip. Four public acceptance
CLI rejection checks cover missing arguments, a missing checkpoint, the wrong
checkpoint identity and an existing destination; none creates a new output.
The [independent review](evidence/issue98-export-consumer/review.md) reports zero
standards findings and zero specification findings.

Format parity does not mean new learning, better performance, equivalence to the
old sigmoid model or deployment approval. NG discovery includes NGA/NGC/NGT
windows; historical measured validation contains both NGG and non-NGG data.
Neither history nor this format check establishes every discovered candidate's
biological suitability, protein-specific PAM compatibility or protein-ranking
accuracy. Operating deployment remains a separate decision.
