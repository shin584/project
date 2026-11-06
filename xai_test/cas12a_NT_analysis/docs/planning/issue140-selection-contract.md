# #140 CPU selection contract

Date: 2026-11-06. Scope: immutable IG coverage and prediction-only technical
case selection. The request is `implement #140`; the issue's scientific
pre-validation freeze remains a gate for #146. This implementation does not
run validation, inspect new Test predictions, freeze a real study population,
or supply a cost/quality threshold. Existing FP16 Test results were known when
these rules were designed; this is not registration against an unseen Testset.

The user confirmed proceeding with the CPU implementation and public save/read
test boundaries after clarification. Independent Standards and Spec design
reviews found no blockers; their schema clarifications (required provenance
fields and rejecting inconsistent full-mode sizes) are incorporated below.

## Public boundaries

`nt_selection.freeze_coverage(partition_path, decision_path, directory, *,
decision_checksum, mode, size=None, fixture=False)` writes a new directory and
returns its manifest SHA256. `read_coverage(directory, *, checksum)` checks the
independently retained checksum, copied inputs, schema and reconstructed list.
The existing read-only partition verifier pins production membership. Synthetic
fixtures are explicit; they retain the 400-target fallback floor.

The cost decision is an external #146 input, not a forecast calculated here.
Its version-1 JSON contains `schema: 1`, `partition: validation`,
`partition_sha256` (the frozen partition membership digest), `analysis_identity`
(an object requiring `study_id`, `run_id` as nonempty strings and
`model_sha256`, `protocol_sha256` as 64 lowercase hex digests),
`full_rejected` (boolean), `reason` (nonempty text), `measurement_reference`
(nonempty external reference), and `measurement_sha256` (64 lowercase hex).
The caller supplies its expected byte checksum. Preserve the decision bytes;
requiring a hashed reference does not independently verify measurement quality.
Only #146 may accept that scientific/cost evidence before production freezing.

Full mode preserves all 1,500 production Test IDs in source-row order and
requires `full_rejected: false`; an explicit size must equal the population
size. Subset mode requires `full_rejected: true`
and 400 <= size < population size. Sort source rows, apply
`numpy.random.Generator(PCG64(42)).permutation`, then take the prefix. Store
selection order and ascending storage order separately, seed, algorithm,
NumPy version, population/selection digests, counts, protocol and copied-input
hashes. Random selection reads membership only. Persist immutable bytes using
#139's acknowledged DirectoryStore; a partial directory is rejected on retry.

`save_cases(coverage_directory, directory, *, coverage_checksum, predictions,
prediction_identity)` first checks coverage. Predictions are saved as a
normalized snapshot of **only** source_row, sequence, label and raw_prediction.
Require precisely the complete population, unique nonnegative integer IDs,
finite fraction labels in [0,1], finite raw output and absolute errors, valid
34bp ACGT sequences with TTT at [4,7), and the frozen per-row sequence hashes.
Prediction identity must equal the cost decision's analysis identity. Upstream
#147 must verify the actual FP32 prediction artifact before supplying rows;
this module cannot attest execution precision from scalar values.

Choose at most five per category, in order: out_of_range, high_error,
low_error, excluding earlier allocations. Out-of-range sorts by descending
clipping distance, descending raw absolute error, ascending source_row. High
error requires error >= global Q75 and > global Q25, sorted descending error
then ascending source_row; low error requires error <= global Q25, sorted
ascending error then ascending source_row. Quantiles use NumPy `method=linear`
on float64 raw absolute errors over the **whole** population before exclusions.
Save implementation version, actual thresholds, Q25==Q75 flag, eligible and
remaining counts, unfilled slots, PAM tags, raw values and explicit reasons.
No quota or forced filling. Selection never opens explanation artifacts.

`read_cases(coverage_directory, directory, *, coverage_checksum, checksum)`
verifies bytes, identity and coverage binding, then reconstructs the complete
selection from saved prediction inputs. Readers reject modified ordering,
duplicates, missing inputs, inconsistent derived fields and invalid protocols,
even when only local checksum sidecars have been recomputed. Independently
retained manifest checksums remain the trust anchors; coordinated replacement
of all trusted inputs is outside the integrity guarantee.

`ig_work(coverage, cases)` provides ascending unique union IDs, separate
statistics IDs, additional case-only IDs and every row's roles. Full coverage
reuses all case IG. Subset statistics use exactly the frozen random subset;
technical cases never enlarge its denominator. The inputs are checked against
their reconstructed protocols and membership before forming this plan.

`save_annotations(..., annotations)` and `read_annotations(...)` store and
verify separate post-selection notes, referring to the original case checksum.
Notes may refer only to selected cases; they never rewrite the case artifact.

## Verification and review

Use synthetic CPU files at the save/read and IG work-plan interfaces, including
1,500 synthetic Test IDs, nested prefixes, exact quantile examples, ties,
duplicate/missing prediction IDs, shortages, equal quartiles, overlap reuse,
invalid costs/modes/floors and tampered bytes/checksums. Run focused files during
implementation, then the touched analysis test subtree. No full repo suite.
Independent Standards and Spec reviews assess this contract before coding and
the implementation before commit. Keep historical ADRs, artifacts and GitHub
issues unchanged. Record concrete results in the implementation report.
