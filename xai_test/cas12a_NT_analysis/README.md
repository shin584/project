# Cas12a native NT analysis

## Stored IG–ISM agreement (#144)

The model-free reader checks committed predictions, IG and ISM, rebuilds the
immutable IG population plan and joins actual source/token records by source ID:

```python
from xai_test.cas12a_NT_analysis.nt_agreement_artifacts import (
    read_agreement,
    save_agreement,
    load_agreement,
)

result = read_agreement(study, run_id)
# New empty directory outside the study; independently retain this checksum.
checksum = save_agreement(study, run_id, new_output_directory)
replayed = load_agreement(new_output_directory, checksum=checksum)
assert replayed == result
```

No model, original dataset, torch, Transformers, GPU reservation or ledger
mutation is required. The standalone export contains normalized `inputs.json`,
recomputed `results.json` and a hash manifest. Reload verifies the exact file
set and trusted checksum, then recomputes every result. Exports never replace
existing files or write into the study. Their provenance is checked at export
time; standalone replay does not independently verify remote study history.

Primary is per-sequence Spearman for t1–t5 over all non-WT alternatives,
comparing absolute token IG / 6 with mean-base mean-alternative absolute raw
ISM delta. Both token sets (five 6-mers and nine tokens), both PAM versions,
mass/span-sum comparisons and population mean-profile correlations are saved.
Every summary records its population IDs, total/valid counts and exclusions.
`populations.statistics` uses only the frozen full/random membership;
`additional_cases` and `validation_probes` are separate descriptive populations.
Source rows retain case roles, including overlap with the statistical population.

Masks, alternative/base counts and actual token spans remain inspectable. PAM
exclusion nulls positions 4,5,6 and leaves t1/t2 with 4bp/5bp ISM support while
their IG still covers the full 6bp. Exact ties use midranks; constant/missing
axes retain null plus a reason. Missing/failed methods or unresolved jobs block
completion; low or normally undefined correlations remain valid results.

For each PAM version, nine-token density ranks retain separate length-group and
pure 3′ flank diagnostics: global rank means/differences, within/between
covariance terms and their contributions normalized by the same global standard
deviation product. Their sum is checked against the nine-token Spearman. These
partitions are descriptive diagnostics, with no causal or confound-fraction
interpretation. The primary five-token ranks cover [0,30), omitting t6–t9.

`sequence_agreement` exposes the numerical boundary; `compare_saved` accepts
reports already verified by `read_results`, `read_ig` and `read_ism` and checks
their ID/identity/token/coverage joins. Prefer `read_agreement` for study access
and committed-byte/quality verification. See the
[agreement contract](docs/planning/issue144-agreement-contract.md).
Only synthetic CPU fixtures were used for #144. Scientific freeze and real
validation remain #146; actual Test agreement remains #148.

Token attention rollout (#143) is available through
`analyze.run_rollout(inputs, study, run_id, reserved_seconds=..., policy=...)`.
It shares the checked prediction run, ledger and sequential resume lifecycle.
Pass an explicit `{"query_l1_threshold": value}` policy frozen before real
validation; there is no production threshold default. The driver commits an
immutable all-prediction-ID plan before loading and reuses verified shards.
`nt_rollout_artifacts.read_rollout(study, run_id, require_complete=True)` reads
the committed results without torch, Transformers, original source or model.
Earlier code identities require a replacement run retaining study charges.

`nt_rollout.compute_rollout` checks eager attention probabilities and both
saved/native and native/attention-enabled FP32 parity (atol=1e-5, rtol=1e-4).
It averages attention heads, mixes `0.5*A+0.5*I`, row-normalizes and left-multiplies
in encoder order using float64 accumulation. Padding is excluded; CLS remains
throughout propagation. Per-source NPZs preserve `(layer,10,10)` valid-token
attention means, the full `(10,10)` matrix, and its unnormalized DNA `(9,9)`
block. JSON sidecars bind source ID, actual token IDs/masks/spans, layer order,
backend class/version, config dimensions, provenance hashes, axes/quantity and
quality. No regression-output-head or base-coordinate axis is added.

Model-free readers reconstruct full/block matrices, removed CLS mass, retained
row mass, parity and all 36 DNA-query-pair L1 distances, including min/mean/max,
near-identical fraction and status on both full-key support and the display
block. Display row sums need not equal one. Near-identical query rows are valid
limitations of attention flow; they do not establish output attribution or
selective token-pair interactions. Failed parity/missing/nonfinite/shape-invalid
attention is saved with observed finite outputs, null arrays and a stage/reason;
it blocks completion and automatic retry. Execution/storage/budget faults keep
the existing explicit settlement or poisoned-manager semantics.

See [the CPU rollout contract](docs/planning/issue143-rollout-contract.md).
CPU fixtures do not represent the actual 1,500-row rollout: scientific freeze
and real validation remain #146, and actual Testset rollout remains #147.

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

## Raw token IG (#142)

`nt_ig.compute_ig` integrates one source through the full native encoder and
regression head. It replaces word-embedding outputs while preserving native
IDs, position construction, token-dropout and attention masks. Alpha=1 must
match the native and stored FP32 prediction with atol=1e-5, rtol=1e-4. The zero
reference includes CLS and padding word embeddings; positional processing and
masks stay native. Signed mass for the nine actual DNA spans and the three
special positions remains separate. No base projection is produced.

Quality has **no production defaults**. Supply `completeness_atol`,
`completeness_rtol`, `near_zero`, `convergence_atol`, `convergence_rtol`, and
`max_steps` (400, 800 or 1600) in a policy object. Midpoint integration starts
at 200 steps, doubles up to that cap and requires two consecutive resolutions.
Completeness sums all 12 signed token masses; near-zero output differences use
absolute tolerance alone and retain a null relative residual with a reason.
Convergence uses the L1 difference between consecutive signed vectors. See the
[IG contract](docs/planning/issue142-ig-contract.md) for the exact equations.
Synthetic test tolerances are not approved actual-NT quality limits.

The Python driver uses the same checked validation inputs, prediction run,
study root, durable receipts and budget as above:

```python
import json
from pathlib import Path
from xai_test.cas12a_NT_analysis.analyze import run_ig
from xai_test.cas12a_NT_analysis.nt_ig_artifacts import read_ig

policy = json.loads(Path("frozen-ig-policy.json").read_text())
report = run_ig(
    verified_inputs,
    study,
    "validation",
    reserved_seconds=300,
    batch_size=32,
    device="cpu",
    policy=policy,
    integration_batch_size=16,
    probe_ids=validation_source_ids,
)
# Retain study.receipt outside the study directory after every acknowledgement.
stored = read_ig(study, "validation", require_complete=True)
```

The reservation above is an example for a synthetic CPU study, not a measured
production forecast. The existing production device/Drive rules and #146 gate
still apply. `run_ig` currently executes validation probes only because its
prediction driver is validation-only. `plan_ig` additionally accepts verified
#140 coverage/case manifests and checks the complete Test population and
partition/model/protocol provenance. Its full/subset work union reuses overlap
and preserves statistics IDs. Passing a Test coverage manifest to a validation
prediction run fails before loading. Future Test execution belongs to #148.

One immutable plan per run binds explicit policy, integration batch, intended
IDs, prediction checksum and coverage/case manifests. Every whole-source
NPZ/JSON shard binds the plan hash, actual tokens/spans/masks, config-checked
reference shape, native/endpoint/reference outputs, every completed resolution,
quality, execution session, units and provenance. Readers reconstruct verdicts,
DNA/special mass and output from committed files without torch, Transformers,
the original model or dataset. Changed plans and corrupt/missing committed
files fail; unreferenced files are ignored. Earlier code identities require a
replacement run within the same study, retaining all historical charges.

Disconnected/nonfinite gradients and outputs are saved as failed diagnostics.
Connected zero attribution is valid; zero norm gives undefined normalization
with a reason. `coverage` includes diagnostic sources; `accepted_coverage`
includes only successful sources. Failed sources always block `complete` and
automatic retry under that plan. A handled numerical failure closes and charges
the job before the driver reports failure. Execution/storage faults instead
retain unresolved-job settlement or poisoned-manager recovery. Interrupted
uncommitted sources are recalculated only after explicit termination settlement;
verified committed sources are reused. No real NT GPU execution was performed.
