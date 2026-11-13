# The NT XAI demo presents stored artifacts and compares no numbers across the two nucleases

Status: accepted, amended by #169 — the two restrictions were settled in the design session recorded
on issue #155 (Q1–Q28) and implemented across #156–#163. Both restrictions stand; the `cross` view
that section 1 describes as carrying them was retired in #169, which is recorded in the Amendment
section below. A Status line is carried here because this ADR re-frames
[ADR 0003](0003-demo-cli-compares-model-a-and-model-b.md) for a second surface, so a later reader
needs to know which of the two is current for which demo: both are, for their own.

The **NT XAI 데모** (`xai_test/nt_xai_demo/`) is one entry point over two finished NT fine-tune
analyses — 9-Hydra v3 (SpCas9) and NT-Cas12a. Two decisions define what it is, and both are
restrictions rather than features, so they are recorded together.

## Amendment (#169): the restrictions stay, the screens that carried them are gone

The demo's surface was rebuilt as one screen — pick a target, read that target's XAI result — because
the branching menu below (`report` / `cases` / `explain` / `agreement` / `cross`, each gated on a
nuclease choice) was the reason nobody could read it, including its author. The meeting it serves
shows what was done; it is not a presentation that needs a table of contents.

Neither decision in this ADR is reversed. Decision 2 (stored-artifact-only, no model loaded) is
untouched. Decision 1 (no comparison across the two nucleases) is **strengthened**: the nuclease is
now chosen when the process starts, so the two never appear on one screen and there is no surface on
which a comparison could be built. What is retired is only the *implementation* — the `cross` view,
and with it the `cases`, `explain` and `agreement` views. A future reader following section 1 to the
`cross` view will not find it.

Two consequences are worth recording, because each one closed a hole rather than opening one:

- **The SpCas9 side shows the `generalisation` stratum only.** Section 1's prohibition on pooling the
  two strata is now unreachable by construction: one stratum is displayed, so there is no pair to
  average. The nine `pam_library` cases are dropped from the demo, not from the analysis.
- **The one qualitative finding survives without the `cross` view.** "Each model's most
  mutation-sensitive region overlaps that model's own PAM" is visible on each nuclease's own screen,
  because the ISM bar and the IG tokens are laid out in position order one above the other. It is
  read off each model separately, which is what the finding always was — stating it on a shared
  screen was never what made it true.

The retired `report` is rewritten short (performance figures, the IG–ISM agreement, one limit line),
and it keeps the wording this ADR exists to protect: every displayed value carries its stratum, head,
statistic and real denominator.

## 1. Integration is at the entry point only; the cross view is one qualitative finding

The demo never places the two nucleases' numbers side by side. Its `cross` view carries exactly one
qualitative finding — **each model's most mutation-sensitive region overlaps that model's own PAM** —
and states on the same screen what it does not compare: performance metrics (different dataset,
split and label definition) and the IG–ISM rank agreement figures (SpCas9's 0.70 is the
`generalisation` stratum's SpCas9-head magnitude median over 200 rows / 193 targets; Cas12a's 0.40 is
the t1–t5 token statistic over all 1,500 Test rows — not the same statistic counted over the same
population). Even the shared finding is asymmetric: the two PAMs face opposite directions (SpCas9 3′,
Cas12a 5′), one analysis resolves to five 6-mer tokens and the other to single positions.

The two analyses also differ in which questions they answered at all — only SpCas9 has a replicate
noise floor and a head-order reversal, only Cas12a has the rollout degeneracy limit — so each
`report` opens with the list of questions its own analysis answered. Without that, the asymmetric
section lists read as something missing from one side.

Within SpCas9 the same prohibition holds one level down: `ig_ism_validation.json` records
`"pooling": "forbidden"`, so the `generalisation` stratum (200 rows / 193 targets) and the
`pam_library` stratum (200 rows / 5 targets) are reported separately and never averaged into one
figure. Every displayed value carries its stratum, head, statistic and real denominator for the same
reason.

### Relation to ADR 0003

[ADR 0003](0003-demo-cli-compares-model-a-and-model-b.md) decided the opposite for a different
surface, and that decision stands. The SaCas9 demo CLI shows Model A next to Model B deliberately,
because the claim it serves — that Model B's gains reflect mechanism learning — is comparative, and
because the two models share the axis that makes the comparison mean something: one Testset, one
label definition, and per-model absolute error quantiles that define the Concordant / Discordant case
types.

This demo has no such axis. 9-Hydra v3 and NT-Cas12a are different nucleases on different datasets
with different splits, different label definitions and different token geometries; there is no paired
row and therefore no error quantile to cut. ADR 0003's comparative frame is not being overturned — it
is unavailable here, and a comparison table built anyway would look like the one ADR 0003 authorises
while resting on nothing.

## 2. The demo is stored-artifact-only and loads no model

The demo imports neither `torch` nor `transformers` and holds no checkpoint. It reads the artifacts
the two analyses already wrote, checksummed through a single `data/manifest.json`, and its new GPU
computation is zero. A number on screen is a number some audited run recorded, not one computed in
the meeting's process. Legacy models (9-Hydra v2 Keras, the Cas12a CNN manifest / single-fit) are
therefore out of scope: their IG/ISM was never produced, and producing it is new GPU work.

Two consequences are accepted rather than solved:

- **The demo reads with its own thin reader**, importing neither `cas12a_NT_analysis` (relative
  package imports) nor `SpCas9_NT_analysis/v3_*` (flat `sys.path` assumptions) — they are not
  co-importable, and importing either would spread the demo's test scope across two analysis
  packages. The cost is that the analyses' own verification does not catch a misreading by the demo.
  Checksums only detect a changed file; they cannot detect an intact file read wrongly. The defence
  against misreading is the synthetic fixture tests, which cover the four known traps: head selection
  by name (ISM has a 9-head axis, IG a 3-head axis, in different orders), row join by `row_index` (the
  artifacts' only shared axis), percent ↔ fraction conversion, and excluding the WT slot from an ISM
  position profile.
- **Bulk data lives outside Git** (`.npz`, per-row CSV), so the real-artifact checks skip on any other
  machine and the restore condition is written down instead (`xai_test/nt_xai_demo/DATA.md`).

## Considered Options

- **One comparison table with both nucleases, captioned as approximate**: rejected. A caption does not
  survive being read aloud or screenshotted, and the audience's question after a shared table is
  always which model is better — a question the data cannot answer.
- **Loading the NT checkpoints and computing live**: rejected. It would add GPU work this ticket has
  none of, make a meeting depend on a model load, and put numbers on screen that no audit covers.
- **Importing the existing analysis readers instead of writing a thin one**: rejected for
  non-co-importability and test-scope spread; the misreading risk it would have covered is covered by
  the fixture tests instead.
