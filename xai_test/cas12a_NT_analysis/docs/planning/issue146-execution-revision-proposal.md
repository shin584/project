# #146 execution revision proposal — not approved

Date: 2026-11-09. Basis: [returned-evidence audit](../reports/issue146-returned-evidence-audit.md).
This is a concrete design proposal, not a change to the frozen scientific
protocol, historical study or authorization to use a new GPU budget.

## Problem and desired behavior

The first 32-source T4 cost group took 1,711 seconds, including 1,398 seconds
of save/reload and verification. Repeated full-history scans grow with the
ledger; full-model hashing follows both methods for every source. Cost 2 hit
its 1,800-second reservation after 22 committed IG/rollout sources. The saved
64 IG sources and 64 rollouts pass numerical replay; the driver overhead,
not a saved numerical failure, is the confirmed implementation problem.

Desired behavior: bound repeated integrity work while preserving immutable
input bytes, durable output acknowledgement, model-state checks, full offline
replay and the existing budget stop. Record timing/progress before a fault
without masking its primary exception. Validate performance before offering
another production run. Do not promise that the current T4/forecast/floor will
fit six hours; that still requires complete new cost evidence.

## Proposed CPU implementation scope

1. **Preserve the primary failure and progress.** If a metered operation already
   poisoned its manager, rethrow that exception without another `guard()`.
   Retain phase, method, completed IDs and cumulative timings in independently
   acknowledged progress records. Progress is evidence only; it never settles
   an open job, supplies a termination time or counts an uncommitted shard.

2. **Introduce a phase-scoped verified view.** Reopen fully validates the
   journal, all referenced bytes and the independent receipt. A phase consumes
   an immutable in-memory view of those verified input bytes. Existing numerical
   readers continue checking hashes and derived values against those bytes.
   New Drive writes still use exclusive immutable paths, fsync/readback and
   external receipts. In-phase budget guards remain at the same compute and
   storage points. Ledger operations verify the current expected tip and newly
   acknowledged writes without rescanning every historical envelope and file.
   Before success, independently read and fully verify the durable chain and
   referenced bytes again, inside the phase reservation. Reopen/offline audits
   retain full verification. Corruption or a conflicting tip blocks completion.

   **Guarantee change requiring approval:** external corruption of an old Drive
   file during a phase may be detected at the final full verification instead
   of the next heartbeat. Computation uses the previously verified immutable
   bytes, never the changed bytes; a phase cannot complete after detection.
   The existing single-writer assumption remains; no distributed lock is added.

3. **Batch model-state verification and commits.** Prototype groups of four
   sources. Check parameter/buffer identity and mutation versions at each source;
   retain a full cryptographic model-state check at loading/fresh parity,
   before each group commit and at phase end. Commit individual source shards
   only after their group's full check passes. Keep existing raw NPZ/JSON source
   formats and roles, and retain whole-group recalculation after interruption.
   Save/reload each method batch once against the verified view, then use the
   unchanged offline readers for final audit.

   **Guarantee change requiring approval:** full-model byte hashes are checked
   at group boundaries rather than after every individual method result.
   Per-source mutation-version checks do not detect every possible out-of-band
   `.data` write. The design relies on the frozen, reviewed native forward and
   IG adapter having no such writes. Deliberate mutate/restore cases and this
   limit must be reviewed explicitly; do not describe this as identical to the
   previous per-method full-byte verification schedule. A mutation present at
   an old method boundary could be caught there but missed by a later group
   hash if restored using a version-counter blind spot. Both schedules can
   miss a mutate/restore completed entirely between their own checks.

4. **Investigate live-model memory.** Add lifecycle observations and weak-reference
   tests around native load, fresh reload and phase cleanup. The returned peaks
   increase, but a CPU probe did not reproduce a leak. Release only objects owned
   by the driver; never assert that allocator cache clearing fixes live references.
   Keep any CUDA-specific claim pending real observed evidence.

5. **Measure IG overhead separately.** Prototype disabling gradients for model
   parameters during IG while retaining input gradients, restoring flags even
   on failure. Compare native endpoints, signed vectors and completeness against
   the current implementation using analytic and tiny native models. This is
   an optimization hypothesis, not a claimed measured T4 speedup. Preserve FP32,
   IDs/masks, zero reference, integrator, numerical limits and retry ladder.

## Verification required before another delivery

- Public strict and phase-view storage paths: stale/missing receipts, write
  failures, conflicting tips, corrupt initial/consumed/final files and immutable
  old study preservation. No cached corrupted input may be consumed.
- Grouped numerical path: unchanged outputs and source roles, a mutation that
  blocks commit, interruption before/after group acknowledgement, and documented
  behavior for mutation-version blind spots. No failed or partial group completes.
- Fake-clock overrun: original exception survives, progress remains diagnostic,
  and no blocked state or budget is reset automatically.
- Returned-archive replay and measured file-read counts on CPU: historical reads
  must not grow quadratically inside a phase. Distinguish a local read-count
  result from actual Drive or T4 throughput; retain real GPU benchmarking as
  a subsequent human execution step.
- Focused checks, then only the analysis subtree; independent Standards and
  Spec reviews before committing production changes.

## Historical budget and human approval

The original study stays blocked and incomplete. Runtime termination is unknown;
do not mint a new receipt, fabricate an upper bound, erase charges, initialize a
replacement root or call the remaining budget certified. Any recovery policy or
new/revised GPU budget requires a separate explicit decision after this review.

Initial approval sought is only the significant execution-architecture revision
and its CPU implementation/testing. It does not authorize a new GPU run,
relax scientific limits, change Test membership, retrain a model, replace any
checkpoint or close #146. Applicable rule: `xai_test/AGENTS.md`, Human approval,
“Making significant architectural changes.” Preserve prior contracts as history;
an approved superseding contract must describe the changed guarantees explicitly.
