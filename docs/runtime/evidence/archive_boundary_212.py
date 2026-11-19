import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification_bundle import load_manifest, verify_receipt


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


base = Path(sys.argv[1]).resolve()
record_path = Path(sys.argv[2]).resolve()
manifest = load_manifest(base / 'bundle', live=True)
run_id = manifest['run_id']
archive_path = base / (run_id + '-required-evidence.zip')
copy_path = Path('D:/MC_project-backups/dashboard-v3/issue205') / run_id / archive_path.name
for target in (archive_path, copy_path, record_path, base / 'archive-index.json',
               base / 'archive_boundary_212.py', base / 'archive-copy.json'):
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {target}')
handoff = json.loads((base / 'handoff.json').read_text())
journal = json.loads((base / 'bundle/journal.json').read_text())
if (journal['state'] != 'ready' or set(journal['completed']) !=
        {'reference', 'verify', 'diagnose', 'float32', 'ram', 'offline'}
        or handoff['manifest_sha256'] != sha(base / 'bundle/manifest.json')
        or handoff['summary_sha256'] != sha(base / 'summary.json')):
    raise ValueError('Archive requires the completed, unchanged #212 handoff boundary')
for stage in journal['completed']:
    verify_receipt(base / 'bundle', stage)
shutil.copyfile(Path(__file__), base / 'archive_boundary_212.py')
registration = json.loads((base / 'preregistration.json').read_text())
vault_pin = registration['model_vault']['vault_record_sha256']
vault_names = [name for name, item in manifest['sources'].items()
               if name.endswith('/model-vault.json') and item['sha256'] == vault_pin]
if len(vault_names) != 1:
    raise ValueError('Missing frozen registered model vault')
vault = json.loads((base / 'bundle/snapshot' / vault_names[0]).read_text())
vault_checks = []
contract = json.loads((base / 'bundle/snapshot/contract.json').read_text())
for location in vault['locations']:
    path = Path(location)
    if sha(path) != vault['archive_sha256']:
        raise ValueError(f'Model vault checksum mismatch: {path}')
    with zipfile.ZipFile(path) as archive:
        matches = {key: hashlib.sha256(archive.read(model['path'])).hexdigest()
                   for key, model in contract['models'].items()}
    if not all(matches[key] == model['sha256'] for key, model in contract['models'].items()):
        raise ValueError(f'Model vault checkpoint mismatch: {path}')
    vault_checks.append({'path': str(path), 'sha256': sha(path), 'checkpoint_hashes': matches})

files = sorted([p for name in ('bundle', 'prepared') for p in (base / name).rglob('*')
                if p.is_file() and '__pycache__' not in p.parts]
               + [p for p in base.iterdir() if p.is_file() and p.suffix in ('.json', '.txt', '.py')
                  and p.name not in ('archive-index.json', 'archive-copy.json')])
index = {p.relative_to(base).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in files}
index_path = base / 'archive-index.json'
index_path.write_text(json.dumps({'run_id': run_id, 'files': index, 'vault_checks': vault_checks}, indent=2)+'\n')
files.append(index_path)
with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in files:
        archive.write(path, path.relative_to(base).as_posix())
with zipfile.ZipFile(archive_path) as archive:
    if len(archive.namelist()) != len(files):
        raise ValueError('Archive entry inventory mismatch')
    for path in files:
        with archive.open(path.relative_to(base).as_posix()) as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != sha(path):
                raise ValueError(f'Archive entry checksum mismatch: {path}')
copy_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(archive_path, copy_path)
checksum = sha(archive_path)
if checksum != sha(copy_path):
    raise ValueError('Backup archive checksum mismatch')
backup = {'paths': [str(archive_path), str(copy_path)], 'sha256': checksum,
          'bytes': archive_path.stat().st_size, 'verified_entries': len(files),
          'scope': 'bundle snapshots, raw arrays, receipts, diagnostic checkpoints, commands/logs/summary and prepared wheels/assets; installed tree reproducible from wheels',
          'independence': 'two local volumes; no off-host or physical independence established'}
(base / 'archive-copy.json').write_text(json.dumps(backup, indent=2)+'\n')
commands = sorted([json.loads(p.read_text()) for p in base.glob('*.command.json')],
                  key=lambda row: row['started_at'])
summary = json.loads((base / 'summary.json').read_text())
record = {'issue': 212, 'run_id': run_id, 'base': str(base),
          'preregistration': json.loads((base/'preregistration.json').read_text()),
          'manifest_sha256': sha(base/'bundle/manifest.json'), 'frozen_sources': manifest['sources'],
          'canonical_inputs_sha256': manifest['inputs_sha256'], 'commands': commands,
          'journal': json.loads((base/'bundle/journal.json').read_text()),
          'handoff': json.loads((base/'handoff.json').read_text()),
          'summary': summary, 'artifact_inventory': index,
          'archive': backup, 'model_vault': vault_checks}
record_path.parent.mkdir(parents=True, exist_ok=True)
record_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(backup, indent=2))
