# V2-A one-time authorized retry

On 2026-10-28 the user explicitly requested: `v2-a 재시도해봐`.
This supersedes the no-retry rule for the first source-drift interruption only.
The original execution and its negative operational finding remain preserved.

The exception permits six fresh fixed runs (baseline seeds42/43/44 followed by
candidate42/43/44), with at most **seven aggregate attempts** including the
previous failed baseline42 attempt. The original **655.8727229001373 seconds**
remain charged to the same **21,600-second cumulative ceiling**. No additional
synthetic learning checks, new recipe, changed thresholds, intermediate-checkpoint
resume, seed reselection, partition change or checkpoint replacement is authorized.
Any new failure/interruption/timeout stops every subsequent fit. Another retry
cannot be obtained by nesting this exception or creating a second child study.

`StudyProtocol.plan(..., retry_from=original_study, retry_authorization=...)`
verifies the original immutable journal/config/protocol/synthetic ownership,
requires one fully accounted failed first fit, and writes a single exclusive
`authorized-retry.json` receipt under the original study. The previous journal,
run state, inputs and reports are unchanged. The child starts its time journal
at the original cumulative charge, retains the previous attempt in configuration
and reporting, and preserves the original six-fit scheduler for its new runs.

The retry source is committed and checked out into a **detached worktree** before
readiness verification or real training. Inputs are copied byte-for-byte and
verified against the existing independent dataset/split/checkpoint pins. Other
commits and pre-commit operations in the shared workspace cannot change this
worktree's HEAD or source bytes. No commits or source edits occur inside it during
execution. All outputs use fresh destinations outside the isolated source tree.

Readiness selects the existing essential study checks plus four new fake-backend
retry fixtures. It does not run the repository-wide suite or any real synthetic
optimizer updates. The focused checks cover cumulative-time carry, preservation
of the original failed attempt, duplicate authorization refusal, unknown/corrupt
original accounting refusal, new failure stopping later fits, and damaged parent
evidence producing an incomplete report with independent scanner evidence.

Original retry provenance is rechecked at the shared execution-configuration
boundary, including finalist/Test and final reporting. A changed parent ledger,
config or ownership receipt cannot authorize Test or an improved V2-A finding.
Supplementary CI and all validation/Test adoption rules remain unchanged.

Executed from the isolated `xai_test/` directory (these destinations are now
reserved, and the one-time retry has been consumed):

```powershell
python -m SpCas9.spcas9_research_run verify --destination C:/project/MC_project/.scratch/issue88-retry-readiness-final
python -m SpCas9.spcas9_research_run execute --destination C:/project/MC_project/.scratch/issue88-retry-execution --readiness C:/project/MC_project/.scratch/issue88-retry-readiness-final --synthetic-ledger C:/project/MC_project/.scratch/issue83-synthetic-budget.json --retry-from C:/project/MC_project/.scratch/issue88-execution/studies/ngg-issue88 --retry-authorization '2026-10-28 user explicitly requested V2-A retry after shared-workspace source drift; original failed attempt retained; one exception only'
```

Before isolation, the four new fixtures passed without real fitting; Ruff passed
all six changed Python files and mypy passed the five changed source modules.
Independent Standards/Spec source review verifies the exception before execution.
Actual readiness and retry outcomes are recorded separately after execution.

## Completed execution, 2026-10-28

**V2-A: `negative_validation`, unresolved.** All six new runs completed in the
detached worktree at `445fbb9`. Both frozen validation comparisons failed the
improvement criteria. This is a completed negative study result, distinct from
the previous operationally incomplete execution. The tested fixed ranking-loss
recipe did not establish the required NGG improvement. No recipe/threshold/seed
was adjusted, and no additional fits were started.

| Comparison / equal-head NGG metric | Reference | Candidate | Change | Required |
| --- | ---: | ---: | ---: | --- |
| Three-seed MAE | 0.086875603 | 0.087699474 | +0.9483% | At least 2% relative reduction |
| Three-seed mean Spearman | 0.610201708 | 0.602403036 | -0.007798672 | At least +0.02 |
| Seed42 vs stored-v2 MAE | 0.087237574 | 0.087178214 | -0.0680% | At least 2% relative reduction |
| Seed42 vs stored-v2 mean Spearman | 0.602725701 | 0.607693801 | +0.004968100 | At least +0.02 |

MAE uses the original published-percentage/100 scale. Each head's metric is
averaged over seeds first, then heads equally. Mean head correlation is distinct
from pooled correlation; both are preserved in the full report, with per-head,
per-seed and population counts/MAE/MSE/Spearman/Pearson.

All NGG/non-NGG per-head protection conditions passed against the refitted
three-seed baseline. Against stored v2, seed42 additionally exceeded the maximum
5% relative non-NGG MAE deterioration for `SpCas9-NG` (+6.2252%),
`eSpCas9(1.1)` (+5.4576%) and `HypaCas9` (+5.5453%). Both comparisons therefore
fail independently. The candidate is not adopted, stored v2 remains unchanged,
and candidate Test was not accessed. No finalist freeze or Test output exists.

Supplementary 95% percentile protospacer-paired bootstrap used seed20261027,
2,000 valid joint draws in 2,000 attempts, with zero invalid draws:

- Relative equal-head NGG MAE change: **[-0.8238%, +2.9780%]**.
- Equal-head mean NGG Spearman change: **[-0.023160, +0.006390]**.

These intervals are supplementary and did not gate the decision. They include
zero; the unfavorable point estimates do not establish universal harm or a
conclusion over arbitrary training seeds. The fixed practical improvement gates
failed. Stored-v2 comparison also mixes the ranking change with the changed
selection/stopping/scheduling policy, while the paired-seed comparison shares
those policies. Historical Test exposure remains recorded.

| Run | Epochs run | Selected epoch | Charged fit seconds | Reload parity |
| --- | ---: | ---: | ---: | --- |
| baseline42 | 31 | 26 | 908.492631 | pass |
| baseline43 | 26 | 21 | 711.446782 | pass |
| baseline44 | 28 | 23 | 1037.476536 | pass |
| candidate42 | 36 | 31 | 1093.669241 | pass |
| candidate43 | 21 | 16 | 652.495428 | pass |
| candidate44 | 31 | 26 | 859.385941 | pass |

All three paired initial-weight hashes match. Every run's persisted evidence
manifest, selected checkpoint and prediction bytes were independently rehashed;
all six checkpoint reload checks passed. The 99 source files match both the
preserved snapshot and the still-isolated live source. Original and copied
dataset/split/stored-v2 checkpoint identities all remain unchanged.

New fits consumed **5262.966558 seconds**. Including the original failed attempt
and synthetic history, cumulative training is **5918.839281 seconds** under the
21,600-second ceiling. There are exactly **seven aggregate attempts**: the prior
one failed attempt plus these six completed runs. Separate evaluation consumed
112.165940 seconds and bootstrap 84.186256 seconds. Remaining time does not grant
additional attempts. V2-B remains independently `verified` using the preserved
actual stored-v2 scanner evidence; no new scanner fitting was performed.

Final isolated readiness passed **84 essential fixtures**, with zero failures or
skips. The repository-wide suite was not run. No extra synthetic optimizer
updates were performed. Standards and Spec review each have zero unresolved
source findings. The launcher's PowerShell status was 1; the durable journal has
six completed runs, `execution-errors.json` is empty, and the sealed final reports
and every required artifact hash were verified. The raw launch log is preserved.

The full local execution is `.scratch/issue88-retry-execution/`; the detached
source remains `.scratch/issue88-retry-isolated/`. Checkpoints and large CSV/log
artifacts are local. The [artifact manifest](evidence/issue88-retry-artifact-manifest.json)
records 199 execution artifact hashes, the log hash, original attempt lineage,
all six checkpoints, paired initialization and input-integrity verification.
The checkout alone does not supply these large local artifacts.

Reviewable records:

- [Final research results](evidence/issue88-retry-research-results.json)
- [Both validation comparisons and all metrics](evidence/issue88-retry-validation-results.json)
- [Frozen execution config](evidence/issue88-retry-execution-config.json)
- [Final pre-fit freeze](evidence/issue88-retry-pre-fit-freeze.json)
- [One-time authorization receipt](evidence/issue88-retry-authorization.json)
- [Detached isolation/input-copy evidence](evidence/issue88-retry-isolation.json)
- [Readiness source/environment](evidence/issue88-retry-implementation-readiness.json)
- [84 essential fixture results](evidence/issue88-retry-essential-tests.xml)
- [Independent Standards/Spec review](evidence/issue88-retry-code-review.md)
