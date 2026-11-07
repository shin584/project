# #146 Colab preparation and verification

Date: 2026-11-07. Branch: `model_analysis`. Starting commit: `a015741`.

After clarifying that the actual analysis runs in Google Colab, the user
explicitly requested executable code they can run and return as artifacts.
This implements that preparation scope. The earlier preflight report remains
a historical record; local CPU-only hardware does not prevent this handoff.
Actual GPU acceptance and #146 completion require the returned real evidence.

## Implementation

`nt_colab build` produces a Korean-guided notebook, its matching source ZIP
and independent SHA256 manifest. The exact current local source is packaged,
including the read-only Cas12a import closure, method contracts and independent
design-review record. It does not depend on pushing a branch or cloning an
older remote revision. Weights, datasets and unrelated working changes are
excluded. The notebook verifies its pinned ZIP checksum/member manifest,
installs declared dependencies while retaining Colab's CUDA PyTorch, mounts
Drive, accepts asset paths and requires explicit first-study/resume selection.

`nt_validation_protocol` deterministically selects disjoint validation-only
correctness, warmup and production groups. The production preset freezes
8 correctness rows, 2 warmup rows, three 32-row groups, actual method batches,
raw-unit numerical policies, retry ladder and budget forecast rules. A human
must accept the displayed protocol hash and confirm a single executor and no
new FP32 Test observation before model loading. Numerical values are declared
engineering acceptance choices, not measurements or biological guarantees.

`nt_validation` runs one acknowledged job per phase using the existing native
prediction, ISM, IG and rollout computation/storage/readers. All actual phase
rows participate in native/adapter/singleton/fresh-reload parity, including a
real batch of 32 in production probes. Loading/parity, method work, tokenizer
calls, synchronization, verification, storage/readback and CUDA memory peaks
are recorded. Nested timings are labelled and not summed twice. The final
external receipt write is reservation-guarded; its self-write latency is
excluded from claimed measured elapsed rates and covered by forecast margin.

The `Study` receipt sink retains an immutable acknowledgement outside the
study after each mutation. Finish/settlement publish only after their matching
close acknowledgement exists; receipt failure poisons the manager and receipt
latency can trigger a budget overrun. Full reservations remain charged without
refunds. No locks, concurrent coordination, automatic termination assumptions
or history reconstruction were added. Whole completed phases are reused;
partial measurements require external termination settlement and an explicit
replacement attempt. Classified native/ISM numerical faults and saved IG/
rollout failures persist as permanent failed-phase evidence and block retries
under the same frozen protocol. Resource/storage/budget faults stay separate.

Model-free replay verifies frozen input, code/config and environment identities,
actual source IDs, method plans and committed raw arrays, then recomputes
quality. The large full-validation binding is stored once as a hashed freeze
snapshot; phase snapshots reuse it and retain their own export metadata.

Forecasting uses only complete production groups, maximum measured rates,
the full allowed IG retry ladder, all 1,500 predictions/rollouts, all 153,000
ISM mutants, already charged time and explicit minute-rounded reservations
for eight future sessions. Full IG is tried first; only its cost rejection
allows the largest affordable fallback of at least 400 rows plus 15 possible
case-only rows. An infeasible floor stops before Test. Existing #140 selection
and #142 IG planning contracts consume the resulting exact four-field Test
identity; scientific protocol and code hashes remain separate receipt fields.

The final freeze binds ordered IDs, membership digest, settings, case rules,
forecast and ledger tip. Fixtures explicitly cannot authorize Test. The
returned archive includes the entire study, raw shards, snapshots, receipts,
protocol and final evidence. Unverifiable histories are returned as diagnostics,
not success. `nt_colab audit` checks the trusted archive checksum/file manifest
and replays evidence without torch, Transformers, source data or model loading.
Existing evidence must already exist during audit; audit does not repair it.

## Verification

Focused numerical-failure regression tests failed before the permanence fix,
then both native parity and nonfinite ISM checks passed. The tests use public
loader/driver interfaces with synthetic inputs and a tiny saved scalar ESM.
Known-answer forecast arithmetic covers full affordability, reservation rounding,
the largest fallback, its 15-case allowance and an infeasible 400-row floor.
Further checks cover protocol acceptance, phase order, complete-phase reuse,
external receipt loss/failure, finish latency overruns, explicit interruption
settlement/replacement and failed-gradient blocking. The full fixture flow
also feeds its frozen coverage through existing case selection and IG planning.

Returned evidence is replayed in a subprocess forbidding torch/Transformers
imports after the original model directory becomes unavailable. Notebook cells
are compiled; source manifests, hashes, deterministic packaging, standalone
source portability, overwrite rejection and unsafe archive paths are checked.

Focused verification from the repository root:

```text
python -m pytest xai_test/cas12a_NT_analysis/test/test_validation.py xai_test/cas12a_NT_analysis/test/test_validation_protocol.py xai_test/cas12a_NT_analysis/test/test_colab.py xai_test/cas12a_NT_analysis/test/test_run.py -q -x
27 passed in 272.20s (0:04:32)
```

Analysis-subtree verification (the full repository suite was not run):

```text
python -m pytest xai_test/cas12a_NT_analysis/test -q
1 failed, 189 passed in 625.30s (0:10:25)

python -m pytest xai_test/cas12a_NT_analysis/test/test_ig_driver.py -q
2 passed in 67.32s (0:01:07)
```

The first run's single failure was the existing IG interruption/resume test:
the agent changed production device metadata during that test, so the resumed
execution correctly rejected a changed code snapshot. Comparing the saved
snapshot with the final working tree confirmed that only the two bare `cuda`
strings changed to explicit `cuda:0`. With the source held fixed, both tests
in that file passed. All 190 subtree cases are therefore covered by successful
checks across the subtree run and final targeted rerun. The standalone source
handoff file also passed an independent two-test rerun.

Ruff lint/format, local documentation targets and staged whitespace checks
passed. No actual GPU execution is represented by these checks.

## Independent implementation review

The `code-review` skill reviewed this task's staged diff against `a015741`
with independent read-only Standards and Spec agents. Unrelated working-tree
changes were excluded. The final review also checked explicit `cuda:0` metadata
against the strict saved-method device checks.

### Standards

0 remaining actionable findings. Initial minor GPU-provenance findings were
fixed: execution is reported separately from acceptance and feasibility,
including real failed or unaffordable probes. Receipt persistence, budget
guards, freeze identities, immutable replay and mounted-Drive initialization
follow the existing contracts. No consequential baseline smells remain.

### Spec

0 remaining actionable findings. The initial native parity/ISM failure-retry
finding was fixed with permanent numerical diagnostics and public regression
checks. Final sink latency is explicitly distinguished from measured rates;
future Test identity matches existing selector/IG planning contracts. Full
batch parity, full-first charged reservations, coverage freeze and returned
artifact replay satisfy the approved Colab preparation scope.

Total findings: Standards 0; Spec 0. Actual GPU acceptance remains outstanding.

## Delivered bundle

Generated in `.scratch/issue146-colab/` within the analysis directory:

| File | SHA256 |
| --- | --- |
| `cas12a-validation-source.zip` | `2d527213bc8f27ddcbfa4115eee550a03636ce2a6c0709c1283a9ecfc2c343e9` |
| `Cas12a_NT_validation_colab.ipynb` | `babedd8f5444e36b0223618695d8d7838af225dcf94e67a92bf5cdd0909f0362` |

Release readback verified both hashes, all 31 packaged source files against
the working tree, unique IDs for all 15 notebook cells and compilation of
every code cell. `bundle.json` independently retains the delivery hashes.

## Execution limits

No real GPU analysis, new FP32 Test observation, retraining or production study
initialization occurred locally. Existing partitions, checkpoints, evaluations,
ADRs and GitHub issues were preserved. A Colab interruption or hardware/package
change can still require explicit recovery or fresh reviewed probes within
the same historical budget. Filesystem readback is not proof of remote Drive
replication, and forecast margins are not a guarantee about unseen inputs.

Use the [Korean guide](../colab-validation.md) and
[reviewed contract](../planning/issue146-colab-contract.md) for execution.
