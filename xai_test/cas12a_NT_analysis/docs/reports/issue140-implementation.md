# #140 implementation and verification

Date: 2026-11-06. Branch: `model_analysis`. Starting point:
`004293d2fc135d40b87f0782acc16e5d505e7932`.

The user requested `implement #140` and confirmed the CPU implementation and
save/read integration test boundaries. This completes implementation of the
selection component; actual validation, scientific cost acceptance and real
pre-Test coverage freezing remain #146 work. No GitHub issue or historical ADR
was modified.

## Independent design review

Separate Standards and Spec reviewers found no blocking mismatch in the
[selection contract](../planning/issue140-selection-contract.md).
Their clarifications are incorporated: the analysis identity must bind study,
run, model and protocol; an explicit full-mode size must match population size.

## Implementation

`nt_selection.py` reuses the existing read-only production partition verifier
and #139's immutable, acknowledged DirectoryStore writes. It preserves the
1,500-target production membership and permits a nested PCG64(42) subset only
with a checksum-bound validation decision rejecting full coverage. The fallback
floor is 400 even for fixtures. Selection/storage orders, versions, protocols,
digests and copied evidence are stored and checked on reload.

Technical cases are selected only from normalized complete-population
predictions and labels. Raw out-of-range, high-error and low-error categories
follow the ticket's exact priority, sort keys and linear quartile rules.
Exclusions, shortages, PAM tags and equal-quartile limitations are explicit.
Saved prediction inputs reconstruct the complete selection on reload without
opening explanation artifacts. Independent checksums bind separate notes to
the original case list.

The IG work plan exposes a unique ID union, per-ID roles and case-only work.
Its separate statistics ID list remains exactly the frozen population/subset;
technical cases never change that denominator. Full mode reuses all case IG.

## Validation

Focused red/green cycles cover the agreed saved-coverage, case selection,
annotation and IG work-plan boundaries. All new data are synthetic CPU files;
the tests never load real checkpoints or inspect actual new Test predictions.
Fixtures exercise 1,500-ID full coverage, nested 400/500-row fallback, a pinned
PCG64 vector, manually calculated quartiles, category priority, ties, shortages,
Q25=Q75, no forced filling, overlap reuse, complete input coverage, invalid
cost/identity/mode/floor inputs and corrupt artifacts. Recomputed local
checksum sidecars cannot bypass reconstruction or an independently retained
original manifest checksum.

Commands from the repository root:

```text
python -m pytest xai_test/cas12a_NT_analysis/test/test_selection.py -q
45 passed in 5.48s

python -m pytest xai_test/cas12a_NT_analysis/test -q
64 passed in 40.04s

python -m ruff check xai_test/cas12a_NT_analysis/nt_selection.py xai_test/cas12a_NT_analysis/test/test_selection.py
All checks passed!

python -m ruff format --check xai_test/cas12a_NT_analysis/nt_selection.py xai_test/cas12a_NT_analysis/test/test_selection.py
2 files already formatted
```

Only the touched analysis subtree was tested, following CLAUDE.md. The full
repository suite was not run.

## Implementation review

### Standards

Independent review: **0 actionable findings**. The change follows documented
asset-preservation, reproducibility and subtree-test rules. Production
membership stays pinned; retained checksums protect stored inputs and lists;
readback reconstructs selection; IG work preserves the frozen statistics
denominator. No actionable baseline smells were identified.

### Spec

Independent review: **0 findings**. The implementation matches #140's
deterministic coverage, case-selection rules, checksum rejection, IG reuse and
separate statistics denominator. Tests cover the required full/fallback, ties,
shortages, degenerate quantiles, overlap and tampering scenarios. Actual cost
acceptance and FP32 provenance remain explicitly assigned to #146/#147.

Total: Standards 0; Spec 0. Neither axis has a remaining issue.

## Execution boundaries

**No real GPU execution, new FP32 Test observation, training or real coverage
freeze.** Existing checkpoints, partitions and evaluation artifacts remain
unchanged. Existing FP16 Test results were known when the selection rules were
designed; this is not preregistration against an unseen Testset.

The selector checks integrity and deterministic behavior, not scientific
measurement validity. #146 must accept the referenced validation cost evidence
and forecast. #147 must supply verified FP32 predictions and correct label
provenance; scalar values alone cannot establish their execution history.
Callers must independently retain manifest checksums. Coordinated replacement
of all trusted inputs/checksums is outside the integrity guarantee. Sequential
writers are assumed, consistent with #139; partial directories are rejected,
not repaired or overwritten.
