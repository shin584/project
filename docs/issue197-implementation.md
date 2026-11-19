# #197 Cas12a manifest prediction

The non-NT computation renderer now uses the adopted Cas12a CNN+BiLSTM
checkpoint, with one prediction row and one selectable `Cas12a` head per site.
The user supplies only the existing 81bp DNA. Explanation calculation is #200;
family selection in the desktop shell is #198.

## Model inputs and units

The manifest CNN uses a 34bp window: 4 upstream bases, 4 PAM bases, the 23bp
protospacer and 3 downstream bases. Its sequence input is float32 ACGT one-hot.
`non_nt_runtime.cas12a_derive` reuses the preserved training extractor and
concatenates GC over the 23bp protospacer, GC over its last 8 bases, then 22
position-major blocks of 16 dinucleotides in ACGT pair order. No feature values
are accepted from the user. The sigmoid's single raw output is a fraction;
the existing score conversion displays its value times 100 as efficiency.

The original scanner's Cas12a NT window starts at the PAM, as already recorded
in `demo-sequences.md`. Reusing it for this CNN would shift the protospacer
features by four bases. `CasConfig.window_upstream` and the optional
`scan_all(configs=...)` let the non-NT renderer use the manifest layout on both
strands while preserving NT's existing inputs and scores. Site keys, guide,
cut position, distance and rank rules retain their existing meanings. A site
without a complete model window remains excluded.

`NonNTCas12aPredictor` checks window geometry, the two graph input names/shapes,
single output name and pinned checkpoint SHA. It loads only on a nonempty
Cas12a batch and reports progress in site counts. Empty batches do not prepare
Keras or NT resources. The process owns the loaded model, with no new module
cache. Nonfinite predictions raise an error.

## Existing views and CSV

The common #196 model identity and renderer work is reused. Best Candidates,
detail rows and XAI selectors all use the same family-specific sites and
predictions. Cas12a has no protein variants; its only head is `Cas12a`.
Explanation requests identify #200 as the implementation owner.

CSV retains the nine original columns, filter predicates, sorting and ranks.
The existing common extension appends `모델` and `모델 버전`; the Cas12a values
identify the manifest CNN and `cas12a-cnn-20261102` with checkpoint digest.
The non-NT worker creates only that family's predictors and its cache keys
include the model/checkpoint identity.

Checkpoint: `models/non_nt/cas12a-cnn-20261102/cas12a_cnn_manifest.keras`.
SHA256: `94ac977865efa1cdad734ae05e033cbaf40b38e109013a6d2fe2b9361aa8b28f`.
Training, data partitions, model weights, XAI defaults and numerical contracts
are unchanged.

## Verification

Implementation-time checks:

- `python -X utf8 -m pytest test/test_non_nt_runtime.py test/test_predict.py -q`:
  32 passed, 0 failed, 0 skipped.

Original-model reference:

```powershell
python -X utf8 docs/verify_issue197_reference.py --output docs/benchmarks/issue197-cas12a-reference.json
```

The separate TensorFlow 2.21.0 / Keras 3.14.1 process loaded the original model
and archived training feature code. It recorded seven fixed windows: the two
actual +/- sites, a Cas12a-only API fixture, and the four original #193 probes.
The predictor regression verifies exact sequence/feature input byte hashes and
both frozen output bounds (absolute <= 1e-4, relative <= 1e-3).

Final selected regression command:

```powershell
python -X utf8 -m pytest test/test_non_nt_cas12a.py test/test_non_nt_runtime.py test/test_predict.py test/test_pam_scanner.py test/test_core_facade.py tests/test_non_nt_cas12a_process.py tests/test_prediction_views.py tests/test_detail_view.py tests/test_worker_cache.py tests/test_computation_process.py tests/test_desktop_xai.py -q --junitxml=.packaging/issue197/regression.xml
```

Result: **131 passed / 0 failed / 0 skipped**, 130.11 seconds. Three existing
SHAP singular-regression warnings appeared in worker cache tests. This is the
ticket's selected scanner, predictor, facade, renderer, process, CSV and XAI
selection regression boundary, not the whole repository suite.

Of these, 15 cases cover the new Cas12a integration. The actual spawned child
verifies preparation and no-site inputs open no weights; Cas12a prediction
opens only its checkpoint while NT weights/tokenizer and SpCas9 weights are
guarded at the file boundary. It verifies rendered fragments, one selectable
Cas12a head, original CSV columns plus model/version, filtering, prediction
reuse, #200 explanation notice and child termination. A mixed-Cas input checks
the Cas12a rows, ranks, score filters and XAI selectors against the same values.

The corrected fixed-input Cas12a rows are 77.501589% and 63.048422% with
`cas12a-cnn-20261102 · sha256 94ac977865ef` identity. Standalone NT scanner
geometry remains unchanged. `git diff --check` passed. Independent
[Standards and Spec review](issue197-review.md): zero findings on either axis.

No new native-window screenshot or frozen deployment was run: this ticket
verifies the existing renderer/API/CSV boundary with actual model processes;
startup/family selection is #198 and final packaging is #204. #193/#195's XAI
numerical failures remain in their original records; prediction integration
does not change their verdicts.

## Commit boundary

The working directory contained ongoing, uncommitted #196 family/identity/CSV
scaffolding before this task. #197 reuses it; those changes were preserved.
At the user's request, this session is committed as work in progress without
including #196's changes. The commit contains the new Cas12a predictor, tests,
reference, records, scanner configuration and the renderer scan hook.

The Cas12a registry entries in `predict.py` and the non-NT scan override in
`desktop/worker_renderer.py` sit inside blocks that #196 is still introducing.
Their #197-only integration diff is committed as
[`issue197-integration.patch`](issue197-integration.patch). The corresponding
source changes remain in the shared working tree for the #196 commit to include;
do not reapply the patch when those changes are already present. The complete
session diff is also saved locally in `.scratch/issue197/changes.patch`.

The 131-test result applies to the combined working tree with the ongoing #196
foundation. This WIP commit alone is not an independently runnable completion
of the family/CSV integration. No #196 implementation files were included in
the commit, and no push was performed.
