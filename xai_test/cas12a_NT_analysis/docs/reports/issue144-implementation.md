# #144 implementation and verification

Date: 2026-11-07. Branch: `model_analysis`. Starting point:
`fbfaf2ae433f22212cb9426fe3828c62a416bed9`.

The user requested `implement #144` and confirmed the sequence-calculation,
checked-artifact join and immutable export/recalculation test boundaries with
`y`. The [agreement contract](../planning/issue144-agreement-contract.md) passed
independent Standards and Spec design review with zero blockers. This is a CPU
implementation; scientific freeze/real validation remain #146 and actual Test
agreement remains #148.

## Implementation

`nt_agreement.sequence_agreement` compares the nine actual signed DNA token IG
masses with raw 34 x A/C/G/T mutant-minus-WT deltas. It preserves original
base-valid masks, WT/PAM masks, eligible alternative masks and counts, base
means, actual token spans, valid base coordinates and token mean/span-sum ISM.
PAM exclusion leaves t1/t2 with 4bp/5bp ISM support while their IG continues
to cover the full 6bp. Missing alternatives/bases remain null plus a reason.

Primary is per-sequence Spearman over t1–t5 with all non-WT alternatives,
absolute IG / 6 and mean-base mean-alternative absolute raw ISM. Both PAM
versions, both token sets, mass/span-sum comparisons and per-token population
mean-profile correlations are stored. Exact ties use midranks. Constant or
missing axes retain null/reason and do not make an otherwise valid calculation
fail. Summaries retain population IDs, total/valid counts, sorted distributions,
medians, exclusion reasons and per-profile paired denominators.

Each PAM version's nine-token density comparison has separate 6-mer/1bp and
pure-3′-flank/remainder diagnostics. Global ranks are computed once; group
mean ranks and their differences are saved. Population within/between
covariances share the same global standard-deviation product. Their normalized
contributions sum to the nine-token Spearman with an explicit arithmetic check.
These are descriptive diagnostics, not causal effects or confound fractions.

`nt_agreement_artifacts.read_agreement` uses the existing checked prediction,
IG and ISM readers. It rebuilds the immutable IG population plan and matches
run/model/tokenizer/protocol identity, source and prediction hashes, actual
token IDs/masks/spans and WT predictions by source ID. It takes the final
accepted IG resolution. Full/random statistics use only frozen statistical
IDs; additional cases and validation probes have separate summaries, and case
roles remain attached to overlapping statistical rows. Missing/failed methods
retain expected rows and block completion, as do incomplete method coverage
and unresolved jobs. Low or normally undefined correlations remain valid.

`compare_saved` provides the join for reports already verified by the existing
readers; its callers must establish committed-byte integrity and method
quality using those readers. It is not a replacement for their validation.

`save_agreement` exports normalized inputs, recomputed results and a hash
manifest to a new empty directory outside the study. It returns a manifest
SHA256 to retain independently. Exports record protocol and Python/NumPy/SciPy
versions. `load_agreement` checks the exact file set and hashes, recomputes all
primary and auxiliary results from the saved inputs, and compares canonical
JSON. It rejects corruption, fabricated derived summaries and overwrite.
Reading/exporting neither reserves GPU time nor mutates the study ledger.

## Verification

Focused red/green checks began with the sequence-calculation boundary and then
the saved-artifact join/export boundary. Worked tie examples have primary
rho=-2/3 and nine-token rho=13/18, independently checked with SciPy. Group-only
agreement gives within=0 and between=1. Additional independent population
covariance calculations check both partitions across tied random examples.
Tests verify separate alternative/base averaging, PAM denominators, empty
spans, constant axes and nonfinite/missing inputs.

Stored fixtures commit IG and ISM in opposite shard orders, use distinct
accepted integration resolutions and retain source-specific masses. A
subprocess forbids torch/Transformers imports and reloads both the committed
study and standalone export after original source/model assets become
unavailable. Identity/token mismatches, missing/failed sources, corruption,
partial exports, unexpected/missing files, fabricated summaries and output
paths within the study are checked through public interfaces.

Full/subset fixtures use the actual #140 selection/manifest and #142 planning
interfaces: the full denominator is 1,500; fallback is 400. Case-only examples
deliberately have opposite agreement and remain outside the statistical
distribution and mean profiles. These are synthetic fixtures, not actual Test
observations or approvals of real NT quality policies.

Focused command from the repository root:

```text
python -m pytest xai_test/cas12a_NT_analysis/test/test_agreement.py xai_test/cas12a_NT_analysis/test/test_agreement_artifacts.py -q
18 passed in 59.67s
```

Final analysis-subtree verification from the repository root:

```text
python -m pytest xai_test/cas12a_NT_analysis/test -q
172 passed in 299.65s (0:04:59)
```

Ruff lint/format checks and staged whitespace verification passed. The full
repository suite is outside this task's scope and was not run.

## Independent implementation review

The `code-review` skill reviewed only this task's staged diff against the
starting commit; unrelated working-tree changes were excluded. Both axes ran
as independent read-only agents.

### Standards

0 actionable findings. No documented-standard breaches or consequential
baseline smells. The read-only study boundary, original masks, calculation
versions, immutable hashes and public-interface tests satisfy the contract.

### Spec

0 actionable findings. Primary/auxiliary statistics, frozen denominators,
source-ID joins, null policies, global covariance diagnostics and model-free
replay satisfy the confirmed #144 CPU scope. No scope creep identified.

Total findings: Standards 0; Spec 0. Neither axis has a remaining issue.

## Execution limits

No actual NT GPU execution, new FP32 Test observation, retraining or changes
to existing dataset partitions, checkpoints, evaluation artifacts, ADRs or
GitHub issues were performed. The existing validation-only execution drivers
are unchanged. Export integrity records checked provenance at export time;
standalone replay does not independently prove remote study history or
scientific validity. Existing sequential-writer/receipt trust limits remain.
