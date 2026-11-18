# Independent minimum recovery (#209)

This procedure restores frozen code, pinned models/manifests and declared wheels
into a fresh directory. The target imports and loads from that restored tree.
It does not copy personal site-packages. Keras 3.14.1/h5py 3.14.0 and the complete
target dependency closure are pinned in `offline-requirements.txt`, also included
by the Windows build requirements. The TensorFlow reference remains a separate
environment/stage; TensorFlow wheels are rejected from target preparation.

Prerequisites supplied externally: existing Windows x64 CPython 3.13.7, its
standard library and native OS components (including the installed C runtime),
and pip for wheel installation. Python/OS provisioning on a clean offline PC and
the final application distribution remain #204. Preparation downloads can use
the network; recovery uses pip `--no-index --no-cache-dir --only-binary=:all:`.
The minimum target runs with `-I -S`, without site/user/PYTHONPATH/.pth injection.

## Commands from empty output directories

Use distinct, nonexistent directories outside the source checkout. Run these
commands from `dashboard_v3`. Each bundle ID is new, including after interruption
or source changes; do not reuse old preparation or restored directories.

```powershell
$recoveryBase = 'C:/project-backups/dashboard-v3/issue209/example-new-id'
$bundle = "$recoveryBase/bundle"
$prepared = "$recoveryBase/prepared"
$restored = "$recoveryBase/restored"

# Connected preparation only. Keep the wheelhouse for subsequent offline runs.
python -m pip download --only-binary=:all: --dest C:/project-backups/dashboard-v3/issue209/wheels -r docs/runtime/offline-requirements.txt --disable-pip-version-check
$LASTEXITCODE
python docs/runtime/verify_contract.py init --work $bundle --run-id example-new-id
$LASTEXITCODE
python docs/runtime/stage_offline.py prepare --work $bundle --wheelhouse C:/project-backups/dashboard-v3/issue209/wheels --output $prepared
$LASTEXITCODE
Push-Location $prepared
python -S "$prepared/payload/docs/runtime/stage_offline.py" $restored --prepared $prepared --work $bundle --output "$bundle/staged.json" --snapshot-only
$LASTEXITCODE
Pop-Location
```

The preparation manifest binds checksummed code/model/manifest bytes to the
#206 bundle manifest and lists every wheel version/checksum. Restore rejects
missing/altered/mixed inputs, aliases of original paths, existing destinations
and escaping paths. Installation and execution commands, return codes, stdout
and stderr are retained. `staged.json` records restored input/dependency hashes,
module origins, executable/prerequisites, probes, RSS and actual held models;
`offline-raw.npz` retains output and every input gradient. The offline stage
receipt binds these artifacts and logs. A completed failure returns 1 and is
retained; an unexpected interruption invalidates continuation in that bundle.

Transport the complete prepared directory and bundle to the recovery location
before the last command if necessary. That command needs only the prepared code
and frozen bundle, plus the declared Python/pip/OS prerequisites. `--snapshot-only`
checks the controller against the frozen code and validates frozen inputs instead
of consulting original paths. Other measurement stages retain the #206 live-input
checks. Changing frozen inputs still invalidates the bundle, and an interrupted
journal still forbids continuation. The target process also denies access to the
bundle and preparation directories, using only the newly restored tree.

During the target run, Python audit hooks deny reads/listing/dlopen under the
original code, model, bundle, preparation and personal site-packages paths, and
deny child-process creation. Every original frozen source/model path is probed;
unexpected denied accesses fail even if an importing library catches the error.
Imported module origins must belong to the restored tree or the external Python
standard library. TCP, DNS and UDP probes must be denied; TensorFlow discovery
and an actual import attempt must establish its absence before loading, and it
must remain absent after both models load.

These guards cover this Python process's audited operations. They are not an OS
firewall, a hostile-code sandbox, or proof of deployment on a fresh PC. Native
libraries and other processes are not isolated by these Python hooks.

Both real non-NT models are intentionally kept alive in this minimum test.
SpCas9 runs the registered NG head on S1, Cas12a the dual-input path on C1.
All forward/gradient outputs must be finite, with nonzero gradients on every
input. RSS describes this minimum forward/gradient process with both models
retained; it does not measure full IG/SHAP/UI or application-tree peak memory.

## Integration with the verification bundle

Use the same preparation/recovery commands after `init` in the complete command
sequence in [verification-206.md](verification-206.md). Collect the other required
stages even after valid numerical failures, then assemble outside the bundle:

```powershell
python docs/runtime/assemble_verification.py --work $bundle --output "$recoveryBase/summary.json"
$LASTEXITCODE
```

A minimum recovery-only bundle intentionally fails overall assembly because the
reference/core/diagnosis/RAM/investigation stages are absent. The offline claim
still reports its actual independent result. Missing provenance, original path
access, missing models/arrays, nonfinite values and nonzero stage exits all
prevent a passing offline claim. Historical staging reports without independent
recovery evidence cannot pass the new assembler.

#212 must generate a fresh official bundle at the final integration snapshot;
the #209 development run is limited implementation evidence. Broader numerical
quality, candidate experiments, final distribution and clean Windows validation
retain their existing owners (#212/#213/#204).
