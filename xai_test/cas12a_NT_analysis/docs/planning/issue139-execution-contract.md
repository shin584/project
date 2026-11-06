# #139 execution contract

Date: 2026-11-06. Scope: CPU implementation of #139, following the accepted
[interview](issue139-durable-run-interview.md) and revised
[acceptance criteria](issue139-durable-run-issue-body.md). The user's subsequent
`implement #139` request authorizes implementation. The interview's removal of
concurrency detection takes precedence over the published issue. No GCS, locks,
leases, Drive API coordination, training, new partitions, or Test predictions.

The user has now confirmed that the preceding design review is complete and
reiterated the implementation request. That current direction supersedes the
stale implementation-waiting status in the earlier local documents and issue.
This document records the remaining technical implementation choices, with a
fresh independent review before coding. It does not claim that scientific
IG tolerances, retry or forecast policy for #140–#149 are frozen, and does not
authorize a validation/GPU experiment. ADR 0010's pre-validation scientific
freeze remains necessary before actual validation in #146.

## Public boundaries and checks

The acceptance criteria already select these integration boundaries: verified
inputs; scalar FP32 prediction followed by export and model-free readback; and
durable study/session recovery. Tests use synthetic source files, a locally
saved small ESM scalar regressor with the actual 4,105-row embedding contract,
and a directory representing external storage. Clock and storage faults may be
injected at those external boundaries. No real checkpoint is loaded in tests.

Input verification reuses the existing read-only manifest verifier and library
reader. Production membership is pinned to #110's source/table/membership
digests. A production export is pinned to the selected-export hashes in the
historical durable receipt, including config, tokenizer config, vocabulary and
weights. Pin the receipt's own SHA256 to
`d79ab828a2d4eccd97cc44c019d515d5b8c3e19ff5580850b0c313b05241f623`.
Fixtures are explicitly marked and cannot enter production. No split
creation function is called. Only validation rows, sorted by source row, are
returned to the driver.

Saved config and tokenizer are loaded locally without network fallback. All
4,096 ACGT 6-mers and A/C/G/T must have distinct, in-range IDs. The 4,107-token
vocabulary has only `<eos>` and `<bos>` outside the 4,105-row embedding; neither
may appear in input. The executed token stream is CLS + five 6-mers + four
single bases, then padding to length 12. Masks, unknown IDs, special positions,
lossless reconstruction, retokenization, and all padded batch IDs are checked.
Layout is 0-based half-open: flank [0,4), PAM [4,8), protospacer [8,31),
flank [31,34). Actual DNA spans are [0,6), [6,12), [12,18), [18,24),
[24,30), [30,31), [31,32), [32,33), [33,34).

Prediction requires eval mode, FP32 parameters/output, autocast disabled, TF32
disabled, eager attention, no gradients and unchanged state dict. Compare
native versus adapter, singleton versus production batches, and a fresh local
reload on the first min(3,N) validation rows, at atol=1e-5, rtol=1e-4. Run the
same comparisons on every session, including GPU changes. Record actual
parameter/output dtype, autocast, TF32, backend, device, installed distribution
versions, Python and a copied, hashed source snapshot including imported
read-only Cas12a modules. Frozen code, protocol, input, model, tokenizer and
packages must match for a run; GPU hardware may change and is recorded in a
new session. The driver measures loading/parity/export time under the ledger.

## Sequential durable study ledger

One fixed study root belongs to #138; every replacement run uses that same
root and 21,600-second cap. Production requires a mounted Drive path outside
the VM work directory. The mounted filesystem is the durability boundary:
readback/hash verification confirms that boundary's acknowledgement, not
independent proof of Google's remote persistence. Operators must preserve and
reuse that root across VM changes. Offline tests prove the protocol only.

Use immutable numbered JSON state records with a previous-record SHA256 and
an integrity envelope. Read the entire chain on every reopen. Missing numbers,
invalid JSON, bad hash, wrong study identity and corrupt or missing referenced
files fail closed. Writes use temporary files, fsync, rename and byte readback;
temporary remnants or acknowledgement ambiguity block continued execution.
Creation is explicit and requires an empty study directory. Reopen requires
an independently retained acknowledgement receipt with the expected chain tip;
never silently initialize a missing ledger. An old receipt or a missing suffix
fails closed. The caller must retain each returned receipt outside this study
directory, including across VM loss. Coordinated replacement of both the
journal and every trusted receipt is outside the integrity threat model.
This is a sequential integrity journal, not a concurrency protocol. It does
not detect or support simultaneous writers.

Before loading any model, persist and read back a start record containing run,
session, UTC start, positive reserved upper-bound seconds and coverage intent.
Completed charges plus the open reservation may not exceed the cap. There is
at most one unresolved job in the sequential workflow, irrespective of run ID.
Heartbeat elapsed time is monotonic within a session, recorded after GPU
synchronization, and never interpreted as a termination timestamp.

For simple conservative accounting, successful or failed closed jobs are
charged their full reserved upper bound, without refunds. Loading, parity,
forward, export, verification, failed attempts and revalidation are inside
that reservation. Before and after each GPU boundary and storage boundary,
including final close readback, check elapsed time against the reservation.
The returned acknowledgement receipt includes post-write elapsed time. Reopen
rejects a close without its independently retained post-write receipt, or one
whose elapsed time exceeds the reservation. A lost final acknowledgement must
be resolved manually, never assumed successful. Overrun records charge at least observed elapsed
time and permanently block further work pending explicit budget review. An
in-process timer cannot force-stop a hung GPU operation; the driver must not
claim a proved cap if a boundary overruns or the VM disappears.

After interruption, heartbeat alone never closes a job. Recovery requires an
operator-supplied external termination-evidence file, confirmed termination
UTC, and a conservative duration upper bound covering start-to-termination
and every recorded heartbeat. Preserve the evidence bytes/hash. Charge at
least the reservation or the supplied upper bound, whichever is greater;
never refund or count the same job twice. A second settlement is rejected.
If evidence or bound is absent, or the cap is exceeded, no model is loaded.
New run identity never discards historical charges or open reservations.

Persist verified artifact bytes before referencing them in a journal record.
An unreferenced shard after failure is ignored; referenced shards must match
hash, run identity, validation coverage and schema. A storage exception
poisons the current manager instance. Reopen and resolve any open job before
additional work. Do not erase the failed record or repair history silently.

## Prediction schema and resume

Each immutable shard contains schema version, run/identity, source row,
source identity, sequence and sequence hash, converted label, raw FP32 output,
clipped output, signed/absolute/squared error for both views, TTTV/TTTT stratum,
layout, actual token IDs/spans/masks, and fraction output units. A run manifest
stores hashes and coverage. Resume skips only verified committed shards and
rejects overlapping/incompatible coverage. Partial shard commits are reusable
after recovery; model loading and parity are charged again.

Model-free readback verifies all referenced bytes, identities, input ordering,
sequence hashes, coverage and derived row fields, then recomputes raw/clipped
Spearman, Pearson, MAE and MSE for all rows and each PAM stratum. Undefined
correlations are JSON null with a reason (empty population, fewer than two
rows, constant labels/predictions or numerical undefined); empty error metrics
are null with a reason. Nonfinite predictions or errors are hard failures.
Labels retain the existing percent /100 then clip [0,1] conversion. No new
scientific quality threshold or historical FP16 equivalence is implied.

The thin driver supports verified validation execution and separate model-free
inspection/recovery. Fixture runs are unmistakably labelled GPU-not-executed;
CPU fixture success is implementation evidence, not completion of #138.

## Review and validation record

Independent design review and implementation review findings, fixes, commands
and actual CPU results are recorded in the implementation report. #139's
GitHub state/labels and historical source/checkpoints/artifacts stay unchanged.
