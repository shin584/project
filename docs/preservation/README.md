# Dashboard v3 asset preservation (#192)

Work in `C:/project/MC_project/dashboard_v3`, branch `dashboard_v3`. Its starting
commit is `9d680575212ad343f060570ea0490ff53ebc2473`. The original
`xai_dashboard_desktop` / `dashboard_core_v2` remains preserved. The older
distribution records build commit `54b77ee207891a8b90b21e818542ae9f9ed7911b`;
these are separate identities. The #177 instructions apply to the original
desktop work, while #191 explicitly requires this separate workspace.

## Disk budget and backup

`disk-plan.json` records the pre-copy inventory, free space, incremental peak
budget and recovery procedure. The backup includes the **entire** original
10,100,655,311-byte workfolder, including `dist/XaiDashboard`, ignored resources,
untracked `AGENTS.md`/`order.md`, and execution evidence under `.packaging`.
Its location is `C:/project-backups/dashboard-v3/issue192-20261118/workfolder`.
`workfolder-inventory.json` beside it records every file's size and SHA-256.
The complete-history `desktop-source.bundle` beside it preserves the Git history;
the workfolder's `.git` file is a pointer into the original repository and is
not sufficient to restore Git independently.

To restore source independently, clone that bundle into a **new** directory,
check out `dashboard_core_v2`, and overlay the backed-up workfolder except its
`.git` pointer. Verify files against the backup inventory. Do not overwrite the
original workfolder. To run the existing distribution, use **all** of
`workfolder/dist/XaiDashboard`, including `_internal`. Launch its executable
from an unrelated working directory with `--validate NEW_OUTPUT_DIRECTORY`.
The recorded smoke result is described in `result.json`; it does not certify
a clean offline PC, visual approval, or scientific XAI quality.

The budget reserves space for future environments, caches, temporary builds and
a new distribution. They are provisional allowances. #193/#204 must update
the budget from actual runtime choices **before** installing/building if those
requirements change. #192 installs no new model runtime and builds no new exe.

## Adopted models and offline recovery

The canonical v3 paths are:

- `models/non_nt/spcas9-9hydra-v2/model.keras` (396,500 bytes).
- `models/non_nt/cas12a-cnn-20261102/cas12a_cnn_manifest.keras` (1,261,811 bytes).

`model_source.NON_NT_FILES` registers both for acquisition; `resolve()` verifies
size and SHA-256, refuses corrupt files and never downloads these entries.
Prediction-family selection belongs to later tickets.
`models/non_nt/manifest.json` records hashes, sizes and acquisition paths for
the checkpoints and every retained provenance file. `model-vault.json` pins
the archive and manifest digests and gives both recovery locations, under
`C:/project-backups/...` and `D:/MC_project-backups/...`.

From this worktree, with Python's standard library only:

```powershell
$vault = Get-Content docs/preservation/model-vault.json | ConvertFrom-Json
python model_assets.py restore --archive $vault.locations[1] --destination C:/new-staging
python model_assets.py verify --root C:/new-staging
```

The destination must not exist. The archive hash is checked before unpacking;
the pinned manifest and every file are checked before staging is published.
Copy the verified `models/non_nt` tree into a fresh checkout's `models/` area.
Weights and bulk CSVs stay outside Git; the manifests and recovery instructions
are committed. The archives are version-pinned local recovery copies. Separate
physical/off-host media durability is not established; no remote upload was
performed. Existing-program backup and checkpoint recovery are independent.

SpCas9 retains the training checkpoint, training log, original export manifest,
#98 acceptance report, evaluation JSON, predictions and full published split.
The source SHA matches the export source and test evaluation checkpoint SHA;
the recorded prediction and split digests are verified. Existing fixed-fixture
export parity is retained, not newly run or a full-test export evaluation.
Cas12a retains all files from the 20261102 manifest run, including its source
snapshot and retrospective provenance limitation. Its validation metrics are
selection evidence; held-out test evaluation is #202.

## Independent NT resources

`resources.json` records byte-verified regular-file copies of the original
`models/` and `.packaging/assets/` (NT checkpoints, tokenizer, Fixed WebView2).
The copies use no hardlinks or symlinks and do not depend on an HF cache.
The original training and evaluation files remain at their acquisition paths.
Full backup hashes, copied resource hashes, source bundle and recovered-model
checks are separate evidence; none substitutes for running the copied app.
