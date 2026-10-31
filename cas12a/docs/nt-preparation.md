# Prepare an NT-Cas12a run

Issue [#103](https://github.com/shin584/project/issues/103) provides the preparation
stage of the [NT workflow](https://github.com/shin584/project/issues/102).
`cas12a.nt_workflow.NTWorkflow` is the public lifecycle boundary for subsequent
training, export, evaluation and recovery work. This stage requires NumPy and
pandas, plus the pandas Excel reader dependency for Excel sources. It imports
neither PyTorch, Transformers nor TensorFlow.

```python
from pathlib import Path
from cas12a.nt_workflow import NTWorkflow

workflow = NTWorkflow(
    source=Path("/content/kim_2018_cas12a_dataset.xlsx"),
    partition_path=Path("/content/drive/MyDrive/cas12a-nt/partition.json"),
    local_work_root=Path("/content/cas12a-nt-work"),
    drive_output_root=Path("/content/drive/MyDrive/cas12a-nt/runs"),
)
run = workflow.prepare(run_id="nt-cas12a-001")
print(run.status, run.local_dir, run.durable_dir, run.checkpoint_dir)
```

Drive mounting and authorization belong to the calling environment. The roots
must be separate and non-overlapping. Checkpoints belong under the returned
local `checkpoint_dir`. The reusable partition path is separately configurable;
keep it outside individual run directories. If no run ID is supplied, preparation
generates a UUID. An existing local or durable destination causes refusal,
including an empty directory; use a fresh run ID to prepare another run on the
same verified partition.

Preparation checks the original source before applying any sequence repair.
It accepts CSV/TXT, TSV and the first worksheet of XLS/XLSX using the historical
Cas12a column names. Columns must be unique, nonempty names and include the
sequence and background-subtracted percentage target columns. Additional columns
are retained. Every source record has its zero-based data-row position as its
identity, matching the historical loader's index; excluded rows leave gaps.

Blank sequences and missing/unparseable targets are excluded with explicit
reasons; multiple reasons can apply to one row. Blank strings count as missing.
Explicit nonfinite numeric labels, including text `NaN` and infinities, cause
refusal. Usable sequences receive only case and surrounding-whitespace
normalization, then must contain exactly 34 ACGT bases and TTTN at positions 4:8.
TTTT is retained. Targets are divided by 100 and clipped to [0, 1]. Duplicate
normalized sequences or 23 bp protospacers at positions 8:31 cause refusal,
as does any usable count other than 15,000. The historical CNN loader, splits
and artifacts are unchanged.

Usable rows are sorted by source identity, permuted with
`numpy.random.default_rng(42).permutation`, then assigned in permutation order:
first 12,000 to `train`, next 1,500 to `validation`, last 1,500 to `test`
(the NT-Cas12a Testset). Complete membership coverage and globally unique
sequences/protospacers establish disjointness. The arithmetic mean of clipped
training targets is frozen in `baseline_train_mean`, alongside its training-row
count and training-partition digest. Validation and Testset labels do not fit
this baseline.

Each successful run writes matching evidence into local and durable directories:

| Artifact | Contents |
| --- | --- |
| `data.json` | Original columns and cell values, source-row identities, normalized usable rows, target ratios, exclusion reasons, source-file/table/sequence hashes |
| `partition.json` | Version, source/table hashes, per-row identities and sequence hashes, algorithm, seed, counts, ordered memberships, partition hashes, frozen baseline |
| `configuration.json` | Run ID/kind, independently resolved source/partition/work/output paths, checkpoint location, split settings and data/manifest hashes |
| `status.json` | Run ID/kind, creation time, configuration hash, preparation state and `model_completed: false` |

`partition_path` holds the reusable `partition.json`. Reuse validates the entire
saved document against the source and deterministic partition contract,
including the frozen baseline. This recomputation verifies integrity; the saved
membership and baseline are reused without rewriting the file. Any source-byte,
canonical-table, row-identity, membership, seed, count, algorithm, baseline or
serialization-version change causes refusal. Even a modified membership with
an updated membership checksum is refused. Damaged manifests are preserved for
inspection; preparation never repairs them or silently resplits.

The versioned canonical serialization is `cas12a-nt-json-v1`: Python JSON encoded
as UTF-8, with keys sorted, `separators=(",", ":")`, `ensure_ascii=False`,
`allow_nan=False`, and no trailing newline. Hashes are SHA-256. Source-file hashes
cover the exact bytes read; the table hash covers
`{"serialization": ..., "columns": [...], "rows": [...]}`, including excluded
rows. Each table row is `{"source_row": integer, "raw": {original column: value}}`.
The column list preserves source order. Parsed numeric cells use JSON numbers,
missing scalar cells use null, and blank strings remain strings. Datetime
metadata uses `{"datetime": ISO string}`; nonfinite numeric metadata outside the
target column uses `{"nonfinite": string}`. The sequence hash covers the
normalized 34 ASCII bases. The partition hash covers the ordered membership
mapping. The training-partition hash covers the complete ordered training row
records, including source metadata and clipped targets. Run configuration hashes
link the separate data and partition documents to each status record.

Run directories are exclusively reserved. Artifact bytes are checked in both
destinations before status becomes `prepared`. A later preparation error records
`preparation_failed` in destinations reserved by that call, where storage remains
writable, and propagates the original error. Pre-existing destinations are never
modified. Source/manifest validation failures occur before run reservation.
Prepared status establishes data/run readiness only: GPU preflight, training,
model export/reload and final evaluation remain separate lifecycle work.
Continue with [model validation and GPU preflight](nt-preflight.md) through
`NTWorkflow.preflight` before creating fresh fit state with `initialize_fit`.

For offline synthetic tables, explicitly opt into a fixture split:

```python
from cas12a.nt_workflow import SplitConfig

fixture_workflow = NTWorkflow(
    source=Path("synthetic.csv"),
    partition_path=Path("temporary/fixture-partition.json"),
    local_work_root=Path("temporary/work"),
    drive_output_root=Path("temporary/durable"),
    split=SplitConfig.fixture(train=8, validation=1, test=1),
)
fixture_run = fixture_workflow.prepare()
assert fixture_run.status["run_kind"] == "fixture"
```

Fixture status, configuration and partition all carry `run_kind: fixture`, even
when testing production-sized synthetic tables. Production counts cannot be
overridden without explicitly marking the configuration as a fixture. Run the
workflow checks with `python -m pytest cas12a/test/test_nt_workflow.py -q`, and the
affected subtree with `python -m pytest cas12a/test -q`.
