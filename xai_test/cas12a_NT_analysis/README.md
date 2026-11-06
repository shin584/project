# Cas12a native NT analysis

This is the CPU-tested implementation of the accepted #139 design. It verifies
the existing source, frozen partition and selected standalone NT scalar export;
predicts **validation only** in FP32; writes immutable prediction shards; and
reconstructs performance without loading a model. No training or Test prediction
entry point is provided. Real GPU validation belongs to #146 after its scientific
specification/review freeze. No real GPU execution was performed for #139.

Run commands from the `MC_project` repository root. Dependencies are NumPy,
pandas, SciPy and the source spreadsheet reader; prediction additionally needs
PyTorch, Transformers and safetensors. The saved ESM config and tokenizer are
loaded locally. Installed package versions are recorded and frozen per run;
the historical training environment is not assumed equivalent to the local one.

## CPU implementation checks

```sh
python -m pytest xai_test/cas12a_NT_analysis/test -q
python -m xai_test.cas12a_NT_analysis.analyze --help
```

Tests create synthetic source/partition files and a tiny saved scalar ESM with
4,105 embedding rows and 4,107 vocabulary entries. They never load the real
500M checkpoint. Fixture studies explicitly require CPU and cannot mix with
production inputs.

## Sequential study lifecycle

The study is one fixed #138 budget across every session and replacement run.
Preserve the **same study root**. A new run is a new identity within that root,
not a fresh budget. Production requires the mounted Colab Drive under
`/content/drive/MyDrive`; initialization is always explicit. Do not initialize
another root to recover a lost ledger.

Each command returns a JSON `receipt`. Keep every latest acknowledgement
**outside the study directory**, in storage retained across VM changes. Reopen
checks its expected chain tip, including the final storage acknowledgement.
The `--receipt` argument accepts either that receipt or the full successful
command JSON. Do not redirect a command's output over its own input receipt.

For synthetic CPU inputs (paths below refer to separately prepared fixtures):

```sh
python -m xai_test.cas12a_NT_analysis.analyze initialize \
  --fixture --study-root /tmp/cas12a-fixture-study > /tmp/receipt-001.json

python -m xai_test.cas12a_NT_analysis.analyze predict-validation \
  --study-root /tmp/cas12a-fixture-study --receipt /tmp/receipt-001.json \
  --run-id validation --source /tmp/fixture/source.csv \
  --partition /tmp/fixture/partition.json \
  --selected-export /tmp/fixture/selected_export \
  --export-receipt /tmp/fixture/export-receipt.json \
  --device cpu --batch-size 2 --reserve-seconds 300 > /tmp/receipt-002.json

python -m xai_test.cas12a_NT_analysis.analyze inspect \
  --study-root /tmp/cas12a-fixture-study --receipt /tmp/receipt-002.json \
  --run-id validation --output /tmp/new-model-free-report
```

Production initialization omits `--fixture`; prediction explicitly selects
`--device cuda`. Source and partition paths must identify the existing #110
inputs. Pass the existing run's `durable_export.json` as `--export-receipt`
and its `selected_export/` directory. The receipt's own hash and production
partition/source identities are pinned, not inferred from caller-supplied
checksums. Do not execute this production path before #146's gate is satisfied.

Choose a conservative reservation covering loading, parity, batches, hashing,
GPU synchronization, saving and acknowledgement. Successful and failed closed
jobs charge their **entire reservation without refunds**. This deliberately
trades budget utilization for simple conservative accounting. The cap is
21,600 seconds. Session revalidation and replacement runs consume the same cap.
GPU changes are allowed; code, protocol, source, model, tokenizer and packages
must still match. Native/adapter, singleton/batch and fresh-reload FP32 parity
use the first up to three validation rows in every new executing session.

## Interrupted jobs

An open job blocks new execution, regardless of run name or elapsed time since
its last heartbeat. Inspect the last independently retained receipt and the
durable chain. After an operator independently confirms the old process/VM has
terminated, retain external evidence and use `settle` with UTC Unix seconds and
a conservative start-to-termination upper bound:

```sh
python -m xai_test.cas12a_NT_analysis.analyze settle \
  --study-root /content/drive/MyDrive/cas12a-xai/study-138 \
  --receipt /content/drive/MyDrive/cas12a-receipts/latest.json \
  --termination-evidence /content/drive/MyDrive/cas12a-receipts/termination.txt \
  --terminated-utc <confirmed-UTC-Unix-seconds> \
  --upper-seconds <conservative-duration-upper-bound>
```

Settlement preserves evidence bytes/hash, charges at least the reservation or
the supplied bound, and rejects duplicate settlement. Use its returned receipt
for the next command. Verified committed shards are reused after recovery;
unreferenced shards are ignored. Changed identity, corrupt files, missing or
old receipts and ledger gaps fail before another model load.

If acknowledgement failed or the VM died before a current receipt was retained,
ordinary reopen stays blocked. Manual investigation must independently establish
the actual tip and termination evidence; there is no automatic receipt rebuilding
or history-repair command. A reservation overrun is recorded and blocks further
execution pending budget review. An in-process timer cannot forcibly stop a hung
GPU kernel. A heartbeat is never proof of termination.

## Stored results and limits

`ledger/` contains the immutable state/hash chain. Each run has a hashed source
and tokenizer/config snapshot, session environment/parity records and prediction
shards indexed by source row. Every shard retains input sequence, source identity,
converted label, raw/clipped predictions and errors, TTTV/TTTT stratum, layout,
actual token IDs/spans/masks, fraction units and run identity.

`inspect --output` writes a new `predictions.csv`, recomputed `metrics.json`
and hash manifest. Correlations use null plus a reason for empty/singleton or
constant populations; empty error metrics also use null plus a reason.
Nonfinite outputs/errors fail. Partial coverage and unresolved jobs remain
explicit, and do not become completion merely because some files are present.
Inspection needs neither torch nor Transformers nor the original model/source.

Readback verifies the mounted filesystem's acknowledgement. It does not prove
remote Drive replication. Receipts and operator termination evidence are trusted
external inputs; coordinated rollback of both the ledger and all retained
receipts is outside this integrity guarantee. The journal assumes sequential
writers and provides no concurrency detection, locks or distributed coordination.

See the [execution contract](docs/planning/issue139-execution-contract.md) and
[implementation report](docs/reports/issue139-implementation.md).

## Frozen IG coverage and technical cases (#140)

`nt_selection.py` is a model-free Python interface for #146's future coverage
freeze and #147's subsequent case selection. The implementation has been tested
with synthetic CPU files. It has not frozen a real population or observed new
FP32 Test results. Existing FP16 Test results were known when these rules were
designed; they are not a preregistration against an unseen Testset.

```python
from xai_test.cas12a_NT_analysis.nt_selection import (
    freeze_coverage,
    read_coverage,
    save_cases,
    read_cases,
    ig_work,
)

# #146 supplies the reviewed validation decision and its trusted byte SHA256.
# Use new directories and retain returned checksums independently.
coverage_sha = freeze_coverage(
    partition_path,
    decision_path,
    new_coverage_directory,
    decision_checksum=accepted_decision_sha,
    mode="full",
)
coverage = read_coverage(new_coverage_directory, checksum=coverage_sha)

# Only after the production freeze: #147 verifies complete FP32 Test predictions.
# Each row supplies source_row, sequence, label (fraction), raw_prediction.
cases_sha = save_cases(
    new_coverage_directory,
    new_case_directory,
    coverage_checksum=coverage_sha,
    predictions=verified_prediction_rows,
    prediction_identity=coverage["analysis_identity"],
)
cases = read_cases(
    new_coverage_directory,
    new_case_directory,
    coverage_checksum=coverage_sha,
    checksum=cases_sha,
)
work = ig_work(coverage, cases)
```

The decision JSON schema and review boundaries are defined in the
[#140 selection contract](docs/planning/issue140-selection-contract.md).
Production reads the pinned existing 1,500-target membership. Synthetic
partitions require `fixture=True`; fallback retains its 400-target floor even
in fixtures. This module checks the decision's identity and integrity; #146
must separately accept its measurement quality and forecast. A reference/hash
alone does not establish that full IG is unaffordable.

Full mode stores all Test IDs. Only an accepted `full_rejected: true` decision
permits `mode="subset", size=N`, where 400 <= N < population size. The random
list is a prefix of a seed-42 PCG64 permutation over ascending source IDs.
`selection_ids` preserves that order; `storage_ids` is ascending. No label,
prediction, explanation or PAM quota affects this sample. Copied partition and
decision bytes, counts, versions, membership digests and rules are verified
on readback against the independently retained manifest checksum. New writes
require empty directories; partial or corrupt artifacts are never overwritten.
Preserve all files and the original returned checksum when transferring a list.

Cases use complete-population raw errors and linear Q25/Q75. They allocate
out-of-range, high-error, then low-error cases, at most five each, excluding
previous selections. Shortages and equal-quartile distributions remain
explicit. The stored normalized inputs contain only IDs, sequences, labels
and raw predictions; extra caller fields are ignored. Reopening recomputes the
whole case list. Upstream verification must establish actual FP32 execution
and label provenance; this selector cannot infer those from numeric values.

`work["ids"]` is the unique IG union, `additional_case_ids` contains case-only
work, and `rows` retains every ID's roles. Use **only `statistics_ids`** as the
IG statistics denominator; adding technical cases does not enlarge it.
Full mode reuses all case IG; subset mode adds only cases outside the sample.
The coverage/case checksums accompany the plan for downstream artifact binding.

After selection, `save_annotations(coverage_directory, selection_directory,
new_note_directory, coverage_checksum=..., selection_checksum=...,
annotations=[{"source_row": case_id, "note": "..."}])` writes separate notes.
`read_annotations` takes the same directories and checksums plus the returned
note `checksum`. Notes refer to the original selection checksum and can name
only selected cases. The original case artifact remains immutable.

See the [implementation report](docs/reports/issue140-implementation.md).

## Saturation mutagenesis (#141)

`nt_ism.compute_ism` evaluates all 102 non-WT substitutions of a checked 34bp
input. Every full mutant is retokenized with the saved tokenizer and checked
for lossless reconstruction, actual spans and embedding-range IDs. The native
forward remains FP32 and raw. Arrays use `(source_row, position, alternative)`
with A/C/G/T alternatives and no head axis. WT slots preserve the original
prediction and exact zero deltas; they do not count as mutations.

`prediction` and `wt_prediction` are float32; raw, clipped and relative deltas
are float64. Raw delta is mutant minus WT. Clipped delta clips each endpoint
to [0,1] before subtraction. Relative delta divides by abs(WT); abs(WT)<=1e-4
yields NaN in the NPZ, an explicit validity mask, and JSON null plus a reason
in summaries. All other outputs must be finite. These are native fraction
output units, not predicted percentages.

The non-WT, base-valid and PAM masks preserve sensitivity denominators.
Substitutions at invariant T positions 4,5,6 are PAM-breaking. Other positions,
including variable PAM position 7, preserve TTTN. This is an input-pattern tag;
it does not establish biological validity or an in-distribution input. The
PAM-preserving sensitivity at invariant positions is undefined because there
are no eligible alternatives; summaries retain null and the exclusion reason.

The Python driver extends the existing validation run:

```python
from xai_test.cas12a_NT_analysis.analyze import run_ism
from xai_test.cas12a_NT_analysis.nt_ism import read_ism

# inputs, study and run_id come from the verified prediction workflow above.
# First complete run_validation with the same code, packages and batch size.
# Choose a reservation covering loading, parity, computation and durable saving.
result = run_ism(
    inputs,
    study,
    run_id,
    reserved_seconds=reservation_seconds,
    batch_size=32,
    device="cpu",  # synthetic fixtures only; production requires CUDA and Drive
)
# Independently retain study.receipt after each acknowledged operation.
selected = read_ism(study, run_id, source_ids=[source_id_b, source_id_a])
```

Each completed source commits an immutable NPZ plus JSON sidecar to the same
run and cumulative ledger. The sidecar binds source/prediction identity,
actual array shapes/dtypes/axes, token records for WT and all mutants, model,
saved tokenizer, protocol, code, environment and producing session. Prediction
and ISM coverage are separate. Interrupted sources are recomputed only after
the existing termination-evidence settlement; committed sources are reused.
Unreferenced files are ignored. Missing, damaged, incompatible or overlapping
committed shards fail validation. Storage failures poison the manager.

`read_ism` uses committed study files alone and requires no model, original
dataset, torch or Transformers. It reconstructs masks, deltas and denominators,
checks each source against its original prediction, and selects/reorders by
source ID. `coverage` and `mutant_count` describe all stored results;
`arrays`, `rows` and `summaries` follow the requested IDs. `require_complete`
checks the full run population, even when selecting fewer IDs. An unresolved
job prevents completion even if all source rows have been saved.

Earlier runs cannot silently acquire a new code/protocol identity. Preserve
them and their charges; create replacement runs within the existing study as
needed. This ticket supplies CPU implementation evidence. Actual validation
and 153,000 Testset mutations remain #146/#147 work under the scientific gate.
See the [ISM contract](docs/planning/issue141-ism-contract.md).
