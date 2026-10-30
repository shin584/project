# Issue #97 independent code review

Fixed point: `7c4fafbbbe7711f010a6cea932ff3677063ee77b`, the pre-implementation
HEAD. Both agents reviewed the staged #97 files using
`git diff --cached 7c4fafbbbe7711f010a6cea932ff3677063ee77b -- <issue97 paths>`.
The commit list was empty during precommit review. Existing unrelated worktree
changes were excluded. Both reviews were read-only.

## Standards

0 findings, including re-review.

The changes respect `xai_test/AGENTS.md` scope and preservation rules, the
sequence-only inference ADR, and the agreed public model-argument test seam.
Consumer code is retained as forensic evidence without modifying e_system.
Snapshot hashes and the archived verification-code hash match the report.
Git attributes preserve those bytes across platforms. No substantive baseline
code smells were found.

The final finite-case gates and regression fixtures keep unsuccessful
characterization from publishing verified evidence. Documentation distinguishes
the retained run from the tightened current verifier.

## Spec

Initial review: 1 P2 finding; resolved on re-review, 0 remaining findings.

The initial verifier could catch a `ValueError` in an ordinary finite case and
still publish `status: verified`. A consumer rejecting above-one predictions
could therefore miss the required distinct-head mapping without failing the
CLI. The same gap affected the zero/tie case. This conflicted with the required
nine-protein mapping and raw/scaled-value checks.

Ordinary finite cases now require successful execution, complete outputs and
correct candidate maxima before publishing `report.json`. Historical crash
classification requires nine variant results and one actual model call. Two
regression fixtures first reproduced the issue, then passed after the fix:
rejecting above-one or zero predictions cannot publish verified evidence.

The retained actual observations satisfy the requested scenarios. The archived
verifier matches the original report's hash. No other missing requirements,
incorrect implementation or scope creep were found.

Standards: 0 findings. Spec: 1 resolved P2, 0 outstanding findings.

Final hook review: 0 remaining standards findings. Ruff's initial formatting of
the forensic copies was reverted to the exact staged bytes. An evidence-local
`.ruff.toml` protects those artifacts without changing root or e_system
configuration or weakening checks on implementation/tests. All five consumer
snapshots and the archived verifier retain their recorded hashes. Ordinary
implementation/test changes were formatting and import ordering only. Both
hooks and all nine focused checks passed afterward.
