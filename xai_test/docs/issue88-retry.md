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

Example from the isolated `xai_test/` directory:

```powershell
python -m SpCas9.spcas9_research_run verify --destination C:/project/MC_project/.scratch/issue88-retry-readiness
python -m SpCas9.spcas9_research_run execute --destination C:/project/MC_project/.scratch/issue88-retry-execution --readiness C:/project/MC_project/.scratch/issue88-retry-readiness --synthetic-ledger C:/project/MC_project/.scratch/issue83-synthetic-budget.json --retry-from C:/project/MC_project/.scratch/issue88-execution/studies/ngg-issue88 --retry-authorization '2026-10-28 user explicitly requested V2-A retry after shared-workspace source drift'
```

Before isolation, the four new fixtures passed without real fitting; Ruff passed
all six changed Python files and mypy passed the five changed source modules.
Independent Standards/Spec source review verifies the exception before execution.
Actual readiness and retry outcomes are recorded separately after execution.
