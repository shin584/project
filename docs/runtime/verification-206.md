# Runtime verification bundles (#206)

#206 repairs the verification tools. Synthetic fixture passes are harness
regressions, **not official model measurements**. #212 performs the new required
two-model measurements and independent recovery after integration; #213 performs
the candidate and conditional precision investigations. #209 implements the
independent restored-code/model execution. The existing #193 contract, reports
and raw directory remain historical evidence, including their failures.

The additive `protocol-206.json` registers the additional seed/raw inventories,
candidate/diagnostic settings and budgets without editing `contract.json`.
Deployment defaults remain IG 32 and SHAP 64/4. Candidate selection (#210) and
qualified float64/step-sweep execution (#211) are separate producers, not new
numerical successes established by this ticket.

## Commands and outputs

Run from the dashboard_v3 worktree, using a new empty directory and a unique ID.
The directory must not contain old outputs. The tools reject implicit historical
paths, duplicate stage execution and attaching results generated before `init`.

```powershell
$bundle = 'C:/project-backups/dashboard-v3/issue205/<new-run-id>'
python docs/runtime/verify_contract.py init --work $bundle --run-id <new-run-id>
python docs/runtime/verify_contract.py reference --work $bundle
python docs/runtime/verify_contract.py verify --work $bundle
python docs/runtime/verify_contract.py diagnose --work $bundle
python docs/runtime/verify_contract.py ram --work $bundle
python docs/runtime/stage_offline.py <separate-staging-path> --work $bundle --output "$bundle/staged.json"
python docs/runtime/verify_contract.py investigation --work $bundle
python docs/runtime/assemble_verification.py --work $bundle --output <new-summary.json>
```

Capture `$LASTEXITCODE` after each command. A completed numerical failure returns
1 and retains its report/raw arrays/receipt; collect the remaining stages.
Do not use a success-only command chain. A stage exception or killed process
leaves an aborted/running marker: keep that directory for diagnosis and start a
**new bundle with all required measurements**. There is no partial resume.
Detected input changes permanently invalidate the journal even if the original
bytes are subsequently restored. Restaging an existing dependency tree is not
independent recovery evidence; that implementation and its official execution
belong to #209/#212.

`init` snapshots the contract, additive protocol, relevant local import closure,
checkpoints and manifests, and canonical input arrays. Layer names come from the
pinned checkpoint configuration, not the observed verification output. The
manifest freezes model/case/head/input/settings/protocol identity and the
expected checks/raw shapes. Every stage records its command, executable, initial
and final environment, backend identity, exit code and artifact checksums;
verification/diagnosis also bind the reference receipt. TF reference and torch
target results are explicitly distinguished. Checksums detect accidental
modification/mixing; these are not authenticated digital signatures.

The core retains all per-head outputs, per-layer activations, per-input analytic
and numeric gradients, per-input and projected IG contributions, baseline/
prediction/error terms, and SHAP seed 0/1/2, repeat and finer contributions/terms.
All intermediate forward/gradient calls are also retained and checked for finite
values. Diagnosis and offline minimum execution retain their raw arrays. RAM
records commands, child exits, held model state and finite minimum/IG results.

## Investigation producer contract

Before `init`, register an `investigation_command` argv list in a separately
registered additive protocol (passed with `--protocol`), starting with
`["{python}", "<frozen-producer.py>", ...]`. `{python}` and `{work}`
are substituted at execution. Freeze the producer script and any extra imported
code with repeated `--extra-code <file>` options. The command must identify a
frozen source file; it runs only inside the journaled investigation stage.
It writes `investigations.json` and `investigation-raw.npz` into `{work}` and
returns 0 for successful collection, including planned budget termination.
The stage stamps identity only on results created during that fresh attempt.
Never import historical results through this path.

Each `investigations` row has `id`, `state`, `planned`, `completed`, `raw_arrays`
and, where relevant, `reason` and `termination`. IDs and planned counts come from
the frozen protocol. `raw_arrays` names NPZ arrays. Completed investigations must
include every registered `required_arrays` entry; all retained arrays and all
numeric report fields must be finite. Candidate array names enumerate all cases,
settings and independent seeds before measurement. Confirmation arrays include
reused intersection measurements from this same bundle; planned 28 calls/model
counts the six new seed combinations and one fresh call on each of four cases.
Float32/float64 `planned: 1` counts a complete registered location/step sweep,
not a single gradient call; #211 adds the location-level preregistration/raw
inventory when its producer is implemented.

Absent producers explicitly write unverified rows, never passing placeholders.
An omitted candidate investigation remains unverified/incomplete. Unknown or
duplicated identities, invalid counts, missing raw evidence claimed as complete
and nonfinite diagnostics are failures. A conditional `not_applicable` row needs
a reason and zero completed calls; it does not represent missing work.

Prespecified budget termination uses `state: time_limited`,
`termination: planned_budget`, a reason, and accurate completed/incomplete counts.
Retain the completed raw arrays and receipts; assembly is allowed, further
measurement attempts in that stage are not. An unexpected interruption must not
be relabelled as planned exhaustion. A finished but unavailable float64
qualification remains unavailable; a finished unresolved investigation is a
different outcome.

## Aggregate result and replay

| Condition | `status.overall` | Exit |
| --- | --- | --- |
| Required failure/missing check/stage, offline failure, identity/integrity mismatch | `fail` | 1 |
| Required checks pass; investigation unverified or float64 unavailable/time_limited | `incomplete` | 1 |
| Required checks pass; completed investigation unresolved or no eligible candidate | `completed_with_findings` | 0 |
| Required checks and applicable investigations complete and meet criteria | `pass` | 0 |

Required failures always take precedence. Findings cannot relabel original
gradient/IG/SHAP failures. Default and finer IG completeness claims are separate.
Every layer has both scaled absolute and relative bounds; the absolute maximum
layer is a summary only. Python/Keras/torch/NumPy/SHAP/h5py version keys are all
mandatory and compared exactly, including the torch build suffix.

Repeat only the assembler command to regenerate a summary. It reads checksummed
snapshots/results, never loads models or adds the current code/contract hash to
old results. It works after moving the complete bundle without its original
source tree. Output must be outside the immutable bundle; existing inputs and
historical reports cannot be overwritten. Keep both checksummed archive copies
specified in #205; copying onto two local volumes does not establish off-host
independence. Actual archive verification is part of the official execution.
