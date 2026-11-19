"""Audit the completed #201 artifacts and separate display from numerical gates."""

import hashlib
import json
import os
import runpy
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from xai.identity import file_sha256
from xai.quality import RESIDUAL_ATOL_PP, RESIDUAL_RTOL


def main():
    contract = json.loads((HERE / 'contract.json').read_text(encoding='utf-8'))
    assert RESIDUAL_ATOL_PP == contract['tolerances']['residual_atol_pp']
    assert RESIDUAL_RTOL == contract['tolerances']['residual_rtol']
    # Import the final verifier's policy declarations without running its main.
    verifier = runpy.run_path(str(HERE / 'verify.py'))
    with tempfile.TemporaryDirectory(prefix='nt-xai-source-') as temporary:
        verifier['snapshot_source'](Path(temporary), verifier['source_hashes']())
        with zipfile.ZipFile(Path(temporary) / 'measured-source.zip') as fresh:
            for name, digest in json.loads(fresh.read('source-sha256.json')).items():
                assert hashlib.sha256(fresh.read(name)).hexdigest() == digest
    out = HERE / 'evidence'
    with zipfile.ZipFile(out / 'measured-source.zip') as archive:
        pinned = json.loads(archive.read('source-sha256.json'))
        for name, digest in pinned.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    summary = dict(issue=201, contract_sha256=file_sha256(HERE / 'contract.json'), cases={},
                   repeat_scope='SHAP estimator repeatability over memoized real deterministic model outputs',
                   source_archive_sha256=file_sha256(out / 'measured-source.zip'))
    changes = {p: dict(measured=digest, final=file_sha256(ROOT / p))
               for p, digest in pinned.items() if file_sha256(ROOT / p) != digest}
    assert set(changes) <= {'desktop/xai_view.py', 'xai/charts.py', 'docs/nt-xai/verify.py'}, changes
    summary['post_measurement_source_changes'] = changes
    provenance = json.loads((HERE / 'output-provenance.json').read_text(encoding='utf-8'))
    baseline = json.loads((ROOT / 'docs/equivalence-baseline/candidates.json').read_text(encoding='utf-8'))
    targets = json.loads((ROOT / 'docs/equivalence-baseline/capture.json').read_text(encoding='utf-8'))['xai']
    seen_checkpoints = {}
    for cas, heads in [('SpCas9', 9), ('Cas12a', 1)]:
        report = json.loads((out / f'{cas}.json').read_text(encoding='utf-8'))
        assert report['status'] != 'incomplete'
        assert report['contract_sha256'] == summary['contract_sha256']
        assert report['source_sha256'] == pinned
        assert report['source_unchanged']
        seen_checkpoints[cas] = report['identity']['checkpoints']
        assert file_sha256(out / f'{cas}-raw.npz') == report['raw_sha256']
        assert len(report['results']) == heads
        if cas == 'Cas12a':
            assert report['identity']['checkpoints'] == [provenance['cas12a']['checkpoint_sha256']]
        target = next(t for t in targets if t['cas_type'] == cas)
        baseline_site = next(s for s in baseline['sites'] if s['site_key'] == target['site_key'])
        row = next(r for r in report['results'] if r['head'] == target['head'])
        observed = row['ig']['32']['pred_raw']
        expected = baseline_site['raw_score']
        assert np.isclose(observed, expected, atol=1e-8, rtol=1e-5)
        assert len(report['checks']) == (412 if cas == 'SpCas9' else 50)
        groups = {'display': [], 'numerical': [], 'provenance': []}
        for check in report['checks']:
            name = check['name']
            group = ('display' if name.endswith('/display') else 'provenance' if
                     name in ('source_unchanged', 'training_mean', 'training_std') or
                     name.endswith('/cache_roundtrip') else 'numerical')
            groups[group].append(check)
        counts = {group: {status: sum(c['status'] == status for c in checks)
                          for status in ('pass', 'fail', 'skip')} for group, checks in groups.items()}
        summary['cases'][cas] = dict(counts=counts, numerical_failures=[c for c in groups['numerical'] if c['status'] == 'fail'],
            fixed_case_prediction_parity=dict(status='pass', raw=observed, preserved_raw=expected,
                                               atol=1e-8, rtol=1e-5),
            report_sha256=file_sha256(out / f'{cas}.json'), raw_sha256=report['raw_sha256'],
            elapsed_seconds=report['elapsed_seconds'], report_status=report['status'])
    log_path = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir())) / 'dashboard-v3/logs/nt-xai.jsonl'
    methods = {cas: set() for cas in seen_checkpoints}
    assert log_path.is_file()
    for line in log_path.read_text(encoding='utf-8').splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        identity = record.get('identity') or {}
        for cas, checkpoints in seen_checkpoints.items():
            if identity.get('checkpoints') == checkpoints and identity.get('protocol') == contract['protocol']:
                methods[cas].add(record['request'][0])
    assert all({'ig', 'shap', 'rollout'} <= found for found in methods.values()), methods
    summary['internal_log'] = dict(exists=True, bytes=log_path.stat().st_size,
                                   methods_by_checkpoint={cas: sorted(found) for cas, found in methods.items()})
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({c: s['counts'] for c, s in summary['cases'].items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
