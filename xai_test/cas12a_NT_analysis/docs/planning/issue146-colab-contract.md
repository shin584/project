# #146 Colab execution bundle contract

Scope: prepare executable Colab code and CPU verification, as explicitly
requested on 2026-11-07. The human mounts Drive, accepts the displayed protocol
hash before observing new validation outputs, executes the notebook and returns
an evidence archive. Actual GPU success and #146 completion remain contingent
on those returned artifacts. No Test prediction or training entry point is added.

## Frozen inputs and execution

Use the existing production source/partition/export verifiers and one study
root with the shared 21,600-second ledger. Before model loading, freeze a
canonical protocol containing validation-only source IDs, exact method settings,
the numerical policy, forecast rules, code hash and design-review evidence.
Bind this to verified inputs and installed package versions. Subsequent phases
must use the same protocol, code, environment and GPU model; a hardware change
requires fresh probes, retaining previous study charges. Require explicit
single-executor and no-new-FP32-Test acknowledgements. Keep #139's sequential
writer contract: open/stale jobs block execution, but no distributed locks or
guaranteed detection of simultaneous starts are claimed.

Production preset: native FP32/eager, autocast and TF32 disabled; prediction
and ISM mutant batches 32, IG alpha batches 16, rollout source batches 1.
Use a seed-42 PCG64 permutation of ascending existing validation IDs to choose
8 correctness rows, 2 separate warmup rows, then 3 disjoint production groups
of 32 rows. Freeze the actual IDs; these are a new declared engineering
protocol, not a claimed prior approval of the historical eight-row candidate.
Fixtures must be explicitly marked and may use smaller groups.

Native, singleton/batch, fresh reload, IG endpoint and attention parity use
the existing atol=1e-5/rtol=1e-4 mechanics. Declare completeness atol=1e-4,
rtol=0.05, near_zero=1e-6; convergence L1 atol=1e-4/rtol=0.05; midpoint
resolutions 200,400,800,1600. Retain the existing all-embedding zero reference,
raw fraction units, near-zero absolute-only test and separate special masses.
Rollout query-L1 threshold is 0.01; degeneracy is descriptive and does not
reject a valid result. These values are pre-observation engineering acceptance
choices, not measured accuracy or biological guarantees. Any numerical failure
is saved, blocks dependent phases and cannot silently change the frozen policy.
Cost phases run parity on all 32 actual source rows, comparing the full native/
adapter batch with singleton and fresh-reload outputs. Record the observed
sample/batch sizes; the inherited three-row driver check alone is insufficient.
The canonical protocol precedes loading; prediction-bound IG/rollout plans are
committed after prediction shards exist within the acknowledged phase job.

Each phase is one acknowledged job: correctness, warmup, cost-1, cost-2,
cost-3, in that order. Reserve 1,800 seconds for correctness and each cost
phase, 900 for warmup; full reservations remain charged without refunds.
Record model loading/parity separately from prediction, ISM, IG, rollout,
weight checks, method save/reload and other overhead. Time using synchronized
wall clocks; separately record nested tokenizer and synchronization time and
CUDA peak allocated/reserved bytes. Nested times are not added twice. Eager
execution uses no explicit compilation. Each phase exercises the actual native
model and all method readers; save raw method evidence, policies, actual IDs,
environment, backend and accepted resolutions, not just pass flags.

Persist an immutable receipt outside the study after every acknowledged ledger
mutation, including final finish/settlement. Receipt-storage failure poisons
the manager and blocks further execution. Never reconstruct a missing receipt
from the study. A running/partial phase requires independently confirmed prior
process termination and explicit settlement. Completed phases can be reused;
retrying a partially measured phase requires an explicit replacement attempt
and fresh whole-phase measurement, preserving all old evidence and charges.
Numerically failed phases cannot be retried under the same frozen protocol.

## Forecast and coverage freeze

Recompute all method quality from committed bytes, including failed sources.
Do not forecast from correctness or warmup throughput. For each production
phase compute per-source prediction/ISM/rollout/IG and save/reload/check/other
rates; use the maximum rate across all three complete groups. Scale each IG
row's measured time by the ratio of cumulative resolutions through max_steps
to its actual cumulative attempted steps, so the forecast allows the full
declared retry ladder. Record the maximum observed accepted resolution too.
The final external receipt write is checked against the active reservation but
its own latency cannot be included in the acknowledgement it writes. Exclude
that tail from claimed measured rates and cover it with the fixed forecast
margin; earlier receipt writes are included in measured storage/common costs.

Elapsed forecast = already charged + 1.5 * (1,500 * prediction/ISM/rollout/common
rate + IG_work_rows * IG rate + 8 * maximum measured loading/parity time) + 600.
The 1.5 multiplier and 600-second fixed reserve cover throughput variation,
additional verification and retries; they are conservative planning margins,
not a guarantee about unseen inputs or different hardware. ISM rate is for
all 102 mutants/source, hence 153,000 total. Common costs measured with all
methods on a row are conservatively applied to all 1,500 targets. Full mode
uses 1,500 IG rows. Only if its total exceeds 21,600 choose the largest integer
N in [400,1499] whose N+15 IG work rows fit. If none fit, save an infeasible
decision and stop before Test. For affordability also construct charged
reservations: two sessions per method, each covering half that method's
projected work, one eighth of common cost, one loading/parity allowance and
75 seconds fixed reserve, with the 1.5 factor applied to measured costs.
Round each reservation upward to whole minutes. Compare already charged plus
their sum with 21,600, so no-refund rounding is included in full/fallback
decisions. Actual future reservations must fit this plan and the same ledger;
unlimited retries/reloads are not authorized.

Use #140's checked `freeze_coverage`/`read_coverage` for the immutable full or
fallback list. The cost decision binds measurement hashes and the scientific
protocol; coverage identity is the intended future Test analysis identity,
not one validation subgroup. Save a final receipt binding the coverage hash,
protocol/code/input identities, ledger tip, forecast, ordering, case rules
and historical-FP16-known status. Fixture evidence must explicitly say no GPU
executed and cannot authorize real Test analysis.
Use #142's exact four-field selection identity: study/run/model and the digest
of the intended Test prediction protocol. Retain the scientific freeze hash
and code digest as separate receipt fields, so existing `plan_ig` can consume
the coverage and cases without changing their contracts.

## Delivery and verification

Public interfaces: protocol preparation/freezing; one phase execution and
model-free phase replay; model-free forecast/coverage freeze; evidence export.
Verify these interfaces with known-answer cost arithmetic, full/fallback/floor
decisions, synthetic tiny native ESM save/reload, failure blocking, interrupted
phase settlement/replacement, durable receipt failure and corruption checks.
Run focused files during development and only the analysis subtree at the end.
The final code-review uses independent Standards and Spec agents.

Deliver a Korean-guided `.ipynb`, a standalone source ZIP made from this
working tree (no dependency on an unpushed remote commit), a manifest of its
exact bytes and an evidence-export command. The archive excludes real model
weights, datasets and unrelated working-tree changes. Output export includes
the full study and independently retained receipts/protocol/freeze evidence;
never overwrite previous artifacts. Notebook starts in explicit first-study
or resume mode; absence of history must not silently initialize a new budget.

References: [Colab FAQ](https://research.google.com/colaboratory/faq.html)
(manual Drive mounting, runtime interruption and file export);
[PyTorch CUDA synchronization](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html).
