# #146 execution preflight — incomplete

Date: 2026-11-07. Branch: `model_analysis`. Audited starting commit:
`7e1e26b992575b18a8ad4e7d73ba1fb9479660e6`.

The user requested `implement #146`. This report records the read-only checks
possible in the current environment and the inputs still required to execute
[issue #146](https://github.com/shin584/project/issues/146). It is not evidence
of successful probes, scientific approval or an IG population freeze.

## Verified prerequisites

The live issue is open with `needs-info` and no comments at inspection time.
Its first acceptance criterion requires an independently reviewed detailed
protocol, hash, probe IDs, numerical criteria, retries, forecast and budget/
resume policies frozen before validation results are viewed. The proposed
eight correctness rows are not an approved default. This requirement also
appears in [ADR 0010](../../../docs/adr/0010-cas12a-xai-controls-token-groups-and-prefers-full-ig.md).

The live completion comments for [#140](https://github.com/shin584/project/issues/140#issuecomment-6683946619),
[#141](https://github.com/shin584/project/issues/141#issuecomment-6684593905),
[#142](https://github.com/shin584/project/issues/142#issuecomment-6685291723)
and [#143](https://github.com/shin584/project/issues/143#issuecomment-6701950745)
confirm CPU implementation completion. They explicitly defer scientific
approval and actual validation to #146. Their closed state does not establish
that #146's execution gate has passed.

Read-only checks of the existing local
`cas12a/models/NT_cas12a/result/20261101-nt500m-full/` assets, using
`cas12a.cnn_manifest.verify_manifest` and `nt_inputs.export_identity` with
`fixture=False`, passed without instantiating a model:

| Check | Observed value |
| --- | --- |
| Existing partition counts | Train 12,000; validation 1,500; Test 1,500 |
| Partition file SHA256 | `e45c8b1128f85ba90b6f2bd17921b64b069081686321bc2809afc1fc8a3bdc9c` |
| Partition membership SHA256 | `680fc75f023e49ce6baae41a581e00d9ba06da024294691964f4ff1f03942f6d` |
| Production export receipt SHA256 | `d79ab828a2d4eccd97cc44c019d515d5b8c3e19ff5580850b0c313b05241f623` |
| Exact selected-export file set | Config, tokenizer config, vocabulary and safetensors; hashes match the pinned receipt |
| Selected weights SHA256 | `eae04f4ff269bd4cf8973b1a219019e6e349272555359878f7ac4ee9f6471969` |

These checks establish local membership/asset integrity only. They do not
verify source spreadsheet bytes, a remote mounted copy, native GPU parity,
scientific quality or the current study ledger. Historical evaluation outputs
were not opened for this preflight.

## Execution blockers

| Required input | Current evidence |
| --- | --- |
| Reviewed scientific protocol and pre-observation freeze evidence | Not located in the inspected analysis/project docs or supplied in this session. Existing method contracts explicitly defer scientific values to #146. |
| CUDA session | Local Python is 3.13.7 with `torch=2.12.0+cpu`; `torch.cuda.is_available()` is false, `torch.version.cuda` is null and device count is zero. `nvidia-smi` is unavailable. |
| Mounted Drive and durable study continuity | No executable Colab session, mounted production study root or independently retained latest receipt was supplied. Production `analyze.require_drive` requires `/content/drive/MyDrive` under an actual `/content/drive` mount. |

The [IG contract](../planning/issue142-ig-contract.md) provides configurable
completeness/convergence mechanics but no approved production tolerances.
The [rollout contract](../planning/issue143-rollout-contract.md) requires an
explicit query-distance policy and defers its scientific threshold to #146.
Synthetic fixture values cannot supply either approval.

The [execution contract](../planning/issue139-execution-contract.md) uses
sequential writers, retained receipts and conservative full-reservation
charging under a shared 21,600-second cap. It explicitly excludes locks and
concurrency detection. Resolve #146's concurrency requirement against that
accepted limitation in the reviewed execution protocol; this preflight does
not add coordination or claim that simultaneous writers are detected.

## Handoff before execution

Supply the reviewed protocol and independent review evidence, including its
trusted hash and evidence of freezing before real validation observation.
It must bind correctness IDs, explicit numerical policies, retry/stop rules,
production probe size and repeated batches, integration batch/backend,
measurement schema, forecast equations/margins and uncertain-time/resume/
single-writer policy. Preserve the distinction between this scientific freeze
and the later validation-informed IG coverage freeze.

Provide an executable GPU session with mounted Drive, the existing source/
partition/export assets and the study root plus independently retained latest
receipt, if a study already exists. Inspect/settle unresolved work using
external termination evidence before additional execution. Missing history
must not be treated as zero prior expenditure or recovered by initializing a
replacement root. An explicitly confirmed first-ever study can be initialized
through the existing acknowledged lifecycle.

Once those gates pass, run correctness probes and independently reload the
saved method outputs to recompute quality. Separately measure repeated
production-cost batches, including loading/warmup/compilation, normal
throughput, peak memory, retokenization, synchronization and storage/readback.
The existing per-method drivers are components; they do not by themselves
produce the required repeated cost-probe report or accept a scientific forecast.

Forecast all 1,500 IG rows first, including expenditure already charged,
1,500 predictions/rollouts, 153,000 ISM mutants, the accepted IG resolution,
storage/verification and retry margin. Only measured cost rejection permits
a seed-42 fallback of at least 400 rows, with up to 15 additional case-only
rows budgeted. If even that floor is infeasible, stop before Test observation.
Durably verify the coverage decision, ordered IDs, membership digest, method
settings and case rules before any new FP32 Test result is observed. Existing
FP16 Test results were already known; this is not an unseen-Test preregistration.

## Result and verification limits

All seven #146 acceptance criteria remain unfulfilled in this session. No
real model inference, probe, cost forecast, coverage decision or freeze receipt
was produced. No production study was initialized or charged. No dataset,
partition, checkpoint, existing evaluation, ADR or GitHub issue was modified.
Unrelated working-tree changes were preserved.

This change adds only this report. The checks above exercise existing read-only
verifiers; no new automated tests or CPU fixture runs are needed to validate
a documentation-only preflight. CPU test success would not resolve the real
execution blockers. #146 remains incomplete until its actual GPU evidence and
durable freeze are available.

Independent `code-review` agents reviewed this report against the starting
commit: Standards 0 actionable findings; Spec 0 actionable findings. Both
reviews preserve the incomplete execution status. Local Markdown targets and
staged whitespace checks passed. No repository test suite was run for this
documentation-only change.
