# #212 required integration measurements

Work in `dashboard_v3`. The implementation base is `b358c9a` (all #206–#211
repairs integrated). The #193 review baseline is
`c7df95b22683fbcefc0e560266bff01ef6190dc2`; its history already includes #194
`ae299ed` (NT computation child). #212 changes the measurement stage boundary,
not the desktop program, runtime/checkpoints, defaults or historical results.

`protocol-212.json` combines the registered #210 candidate settings and #211
diagnostic settings. It adds `split_diagnostics: 212`: mandatory float32
diagnostics finish in their own single-attempt stage, with a receipt binding
the core/diagnosis receipts and `diagnostic-plan.json`. The plan fixes the union
of historical/new failures and the passing controls before the sweep. All six
steps, entire coordinate inputs and all 32 seed-193 directions are retained.
The original failures remain failures.

The investigation stage remains **unstarted** at the #212 handoff. Its registered
candidate producer and conditional float64 budget are unchanged. #213 consumes
the checksummed completed float32 rows/arrays, without repeating or overwriting
them. Both entry paths preserve the original combined #211 procedure for older
protocols. An interrupted stage or changed frozen input still requires a new
bundle and every required remeasurement.

Run these complete commands from the worktree with the declared Windows x64
Python 3.13.7. See [recovery-209.md](recovery-209.md) for the Python/stdlib/pip/OS
prerequisites and the scope of the in-process path/network guards.

```powershell
# Connected preparation; an existing complete declared wheelhouse can be reused.
python -m pip download --only-binary=:all: --dest C:/project-backups/dashboard-v3/issue209/wheels -r docs/runtime/offline-requirements.txt --disable-pip-version-check

# Choose a NEW nonexistent base and unique ID for every invocation.
python -X utf8 docs/runtime/run_required_212.py --base C:/project-backups/dashboard-v3/issue205/<new-run-id> --run-id <new-run-id> --wheelhouse C:/project-backups/dashboard-v3/issue209/wheels
$LASTEXITCODE
```

The controller writes preregistration before `init`, freezes it with the bundle,
then executes the following stages in separate processes: init, TensorFlow
reference, torch verification, original diagnostics, float32 sweep, RAM,
preparation, independent offline recovery, summary. Each exact expanded argv,
working directory, start/end, stdout/stderr and exit code is saved under the
base. Receipts retain execution environments and checksummed stage outputs;
the manifest retains the source/contract/protocol/model/input inventories.
The prepared controller runs outside the original checkout and the restored
target runs with `-I -S`, using installed declared wheels and restored assets.

A numerical failure exits 1 and collection continues while the journal is ready.
An aborted/running/invalidated journal stops collection and requires a fresh run.
The controller exits 1 for aggregate fail/incomplete even when required collection
and handoff are complete. Additional investigations are never silently counted
as passed. RAM covers interpreter/import/load/forward/IG with one held model per
child; offline RSS covers minimum forward/gradient while both models remain
held. Neither establishes full IG/SHAP/UI process-tree peak.

Read-only summary replay (no measurements, explicit output outside the bundle):

```powershell
python -X utf8 docs/runtime/assemble_verification.py --work C:/project-backups/dashboard-v3/issue205/<run-id>/bundle --output C:/project-backups/dashboard-v3/issue205/<run-id>/summary-replayed.json
$LASTEXITCODE
```

After verifying and archiving the #212 boundary, #213 may execute exactly the
previously unstarted stage, from the same unchanged checkout and run:

```powershell
python -X utf8 docs/runtime/verify_contract.py investigation --work C:/project-backups/dashboard-v3/issue205/<run-id>/bundle
$LASTEXITCODE
python -X utf8 docs/runtime/assemble_verification.py --work C:/project-backups/dashboard-v3/issue205/<run-id>/bundle --output C:/project-backups/dashboard-v3/issue205/<run-id>/summary-final.json
$LASTEXITCODE
```

Do not execute those #213 commands as part of #212. A normal boundary is not
permission to resume an interrupted stage or mix another run's measurements.
Keep the #212 archive immutable when later extending the live bundle for #213.
Both agreed local-volume copies must be checksummed; they do not establish
physical or off-host independence. Numerical quality remains limited to all
nine SpCas9 forward heads, NG gradient/XAI and the current Cas12a dual-input
path. Candidate success never promotes user defaults (IG 32, SHAP 64/4).

Archive the completed boundary and write its versioned evidence index using
the auxiliary recording tool below. It verifies both existing pinned model
vault copies against the measured checkpoints, checks every archive entry
against its source, then verifies the C:/D: archive checksums. It includes raw
arrays, frozen inputs, receipts, logs, summary and prepared wheels; the installed
recovery tree is reproducible from those wheels. The recording tool is included
in the archive and does not change the frozen measurement source files.

```powershell
python -X utf8 docs/runtime/evidence/archive_boundary_212.py C:/project-backups/dashboard-v3/issue205/<run-id> docs/runtime/evidence/<run-id>.json
$LASTEXITCODE
```
