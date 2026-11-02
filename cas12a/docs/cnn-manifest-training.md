# Manifest-based single CNN training

Issue [#116](https://github.com/shin584/project/issues/116) adds a second
training entry point beside [the five-fold one](cas12a-5fold-metrics.md). It
exists because [#115](https://github.com/shin584/project/issues/115) compares a
CNN against the standalone NT-Cas12a regressor, and that comparison is only
meaningful if both models were fitted on the same rows and neither was selected
on the rows the comparison is scored on. The five-fold path cannot do that: it
constructs its own `KFold(5, shuffle=True, random_state=42)`, so every row is
in some fold's validation set and the NT-Cas12a Testset overlaps its training
data.

```bash
python -m cas12a.cnn_single \
  --source   cas12a/dataset/kim_2018_cas12a_dataset.xlsx \
  --partition cas12a/models/NT_cas12a/result/20261101-nt500m-full/partition.json \
  --output-root cas12a/artifacts/cnn_manifest --run-id 20261102-cnn-manifest
```

Both defaults already point at those two files. The run directory must not
exist; the five-fold `.keras` and `cas12a_5fold_metrics.json` live one level up
in `cas12a/artifacts/` and are never touched.

This document describes the entry point. The confirmed training run itself is
[#117](https://github.com/shin584/project/issues/117); a fixture rehearsal of
this path is not that run, and `status.json` records which it was.

## Membership comes from the manifest, and only from there

`cas12a/cnn_manifest.py` reads the partition NT froze and never writes one. It
has no access to `partition_library`, `save_or_verify_partition` or `KFold`,
and a test runs the whole path with all three replaced by functions that raise.

Verification is in two halves, because they fail for different reasons.

`verify_manifest` is structure and self-consistency, and needs no data file:
the three partition names, counts against membership length, non-negative
integer identities, no row repeated within or across partitions,
`partition_sha256` recomputed from the membership it claims to describe, row
identities ascending and covering exactly the membership, the frozen training
baseline describing the train partition, and the recorded `algorithm`/`seed`
being the ones `cas12a/nt_data.py` writes. `run_kind` is derived from the counts
rather than read as a field, so an 11-row fixture cannot relabel itself.

`verify_against_library` is identity against the actual Kim 2018 sheet, read
through `nt_data.read_library` so the 34 bp ACGT, TTTN PAM, duplicate-sequence,
duplicate-protospacer and `indel % / 100` clipped-to-`[0, 1]` rules are the ones
the NT run applied rather than a second implementation of them. It compares the
source-file and table digests, every per-row sequence checksum, the
training-partition digest and its frozen baseline mean, and states protospacer
disjointness across the three partitions directly.

### Why the production identity is pinned

Self-consistency is not enough on its own. `partition_sha256` describes
whatever membership the file happens to carry, so anyone who moves a row and
re-digests passes every internal check. `training_partition_sha256` catches a
changed *training* row, because it digests the rows and not their indices - but
a `validation`/`test` swap leaves the training rows untouched, and that is a
Test row entering model selection with nothing left to object.

So `PRODUCTION_IDENTITY` pins the #110 run's values, and they are demanded
whenever the counts are the production 12,000/1,500/1,500:

| Field | Value |
| --- | --- |
| `manifest_sha256` | `e45c8b1128f85ba90b6f2bd17921b64b069081686321bc2809afc1fc8a3bdc9c` |
| `partition_sha256` | `680fc75f023e49ce6baae41a581e00d9ba06da024294691964f4ff1f03942f6d` |
| `training_partition_sha256` | `383bb97d91be31ce8220d95e9c0c39c908a4309f6cd9323dd465809894d23029` |
| `source_file_sha256` | `c465925148023dcfe01e388c79c17ccaf73bc454c815eaef72595ba6ee71d028` |
| `table_sha256` | `25ac3a282b31193a31067168bd3c7ec9cdf2b34ce2ffb30803f2f4bf5e640c3c` |

`manifest_sha256` is the digest of the whole document - the value NT's own
`configuration.json` carries as `partition_manifest_sha256` - so it fixes all
three memberships at once. A fixture manifest has no frozen identity to be
pinned to and is checked for consistency only, which is what lets
`cas12a/test/` exercise this path on 11 synthetic rows.

Only `train` and `validation` are returned as arrays. The `test` membership is
read, because proving the other two exclude it requires knowing what it is, and
is returned as bare identities: no held-out row's features or label is built.

## The recipe is the five-fold path's

Imported from `cas12a/train.py`, not restated - a copy would be a second recipe
that looks like the first. Seed 42, Adam 5e-4 (in `cas12a/model.py`), MSE, batch
32, at most 100 epochs, `EarlyStopping` on `val_loss` with patience 10 and
`restore_best_weights=True`, `ReduceLROnPlateau` at patience 5 / factor 0.5 /
floor 1e-5. Op determinism stays off, as it is for the five-fold path. The
architecture is `create_cas12a_model()` and the inputs are the canonical
one-hot and 354 derived features from `cas12a/features.py`, so the encoding is
the same whether a row arrived through the loader or through the manifest.

Validation rows are passed as `validation_data`, never `validation_split`: the
rows the model is selected on are the manifest's, not a tail of the training
array that happens to be the right length.

`SingleFit` holds the one-model/one-fit rule. "We trained once" is the part a
later reader cannot check from the artifacts - two fits leave the same files as
one - so a second build or a second `fit` raises `SingleFitViolation`, and the
counts reach `history.json` either way.

## Save, reload, and two separate judgements

`best_epoch` is the argmin of the recorded `val_loss` rather than the
callback's report, because it has to describe the weights that are actually in
the saved model; `epochs_ran` and `early_stopped` are recorded beside it, since
one "epochs" field cannot say whether early stopping fired.

It is `nanargmin`, and the distinction is not cosmetic. Keras decides
improvement with `current < best - min_delta`, which is False against NaN, so a
NaN epoch is never the one `restore_best_weights` puts back - while `np.argmin`
returns its index regardless, and would name an epoch whose weights nothing
restored. Any non-finite validation epochs are listed in
`nonfinite_val_loss_epochs`, and a fit where no epoch produced a finite
validation loss fails rather than reporting a selection that never happened.

The reload comparison is two checks, following
[#111](https://github.com/shin584/project/issues/111):

1. **Weight identity, exactly.** Per-tensor SHA-256 over raw bytes, compared
   before any forward pass. A variable the loader re-initialised is a loading
   defect however its predictions land, and a tolerance wide enough to cover
   reduction-order drift is wide enough to hide it. Identity is the weight's
   position, shape, dtype and bytes; its *name* is recorded beside them and
   not compared, because Keras's `Bidirectional` wrapper reports its inner
   cell's variables as `bidirectional/forward_lstm/...` in a live model and
   `forward_lstm/...` in a reloaded one. Name-keyed comparison would report
   six renamed tensors as twelve missing and unexpected ones, which is noise
   loud enough to hide the defect the check exists to find.
2. **The forward pass, at FP32's own resolution.** `rtol = 128 * 2^-24`, with
   `atol` a tenth of that, over the validation predictions at the fit's batch
   size. A deliberate margin against reduction-order drift, not a measured
   error bound - and it bounds forward reproducibility only.

## What a run writes

| File | Content |
| --- | --- |
| `cas12a_cnn_manifest.keras` | The checkpoint: the restored best-epoch weights. |
| `configuration.json` | Frozen recipe, seed, source/table/partition digests, the pinned production identity, train and validation membership, the held-out identities, library versions, CPU settings. |
| `history.json` | Full per-epoch history, `best_epoch` apart from `epochs_ran`, wall clock, validation predictions, `model_builds`/`fit_calls`. |
| `checkpoint.json` | Checkpoint digest and size, per-tensor weight identities, and both reload judgements with their evidence. |
| `validation_predictions.csv` | One row per validation row, joined on `source_row` with its `sequence_sha256` - never on array offset - with target, prediction, reload prediction and their difference. |
| `status.json` | Per-stage states, artifact digests, and `records_confirmed_training`. |

`records_confirmed_training` is false, with reasons, for anything but a
production manifest fitted at the recipe's own seed and epoch ceiling. A fixture
run produces every file a real one does, so the file has to be the thing that
says it is not one.

Two things this path deliberately does not do:

- **It cannot resume.** There is no checkpoint-per-epoch and no restart, so an
  interrupted fit leaves a run directory with no `status.json` and is not
  reportable; rerun it under a fresh `run_id`. The design permits recovery only
  if it "preserve[s] the same run and full compatible training state", which a
  one-shot `model.fit` cannot offer, so the alternative it names applies:
  report incomplete training.
- **It does not choose its own parent directory.** The files above are written
  flat inside whatever `run_dir` the caller names, so #115 gets the `cnn/`
  entry of its artifact table by pointing `--output-root`/`--run-id` at it
  rather than by this module hardcoding a layout for a run it is not part of.

Tests are `cas12a/test/test_cnn_manifest.py` (no TensorFlow: membership,
identity and encoding) and `cas12a/test/test_cnn_single.py` (the fit, its
record and its artifacts). Neither asserts a performance number.
