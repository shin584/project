"""Render the qualified real arrays with final UI code, without model recomputation."""

import base64
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

import core
from desktop.xai_view import explanation_fragment
from xai.identity import file_sha256
from xai.units import ExplanationUnits

HERE = Path(__file__).resolve().parent


def main():
    meta = json.loads((ROOT / 'docs/equivalence-baseline/capture.json').read_text(encoding='utf-8'))
    output = ROOT / '.packaging/evidence/issue201/final-display'
    output.mkdir(parents=True, exist_ok=True)
    observations = []
    for target in meta['xai']:
        cas, head = target['cas_type'], target['head']
        report = json.loads((HERE / f'evidence/{cas}.json').read_text(encoding='utf-8'))
        index, row = next((i, r) for i, r in enumerate(report['results']) if r['head'] == head)
        units = ExplanationUnits(**row['units'])
        ig_meta, shap_meta = row['ig']['32'], row['shap']['64-4-0-0']
        with np.load(HERE / f'evidence/{cas}-raw.npz') as arrays:
            ig = core.IGResult(arrays[f'h{index}_ig32'], ig_meta['pred_raw'], ig_meta['baseline_raw'],
                               ig_meta['error_pp'] / units.scale, units)
            shap = core.SHAPResult(arrays[f'h{index}_shap_64-4-0-0'], shap_meta['base_raw'], shap_meta['pred_raw'],
                                   shap_meta['error_pp'] / units.scale, 0, 0, units)
            rollout = core.RolloutResult(arrays['rollout_matrix'], report['rollout']['labels'],
                                         arrays['rollout_profile'], report['rollout']['n_layers'])
        cache = SimpleNamespace(integrated_gradients=lambda *a, **k: ig,
                                kernel_shap=lambda *a, **k: shap,
                                attention_rollout=lambda *a, **k: rollout)
        site = next(s for s in core.scan_all(meta['input_seq']).sites if s.site_key == target['site_key'])
        for method, params in [('ig', {'steps': 32}), ('shap', {'nsamples': 64, 'n_background': 4}), ('rollout', {})]:
            fragment = explanation_fragment(SimpleNamespace(), site, head, method, params, 80, lambda *a: None, cache=cache)
            images = re.findall('src="data:image/png;base64,([^"]+)"', fragment)
            assert len(images) == (1 if method == 'ig' else 2)
            assert 'class="warning"' not in fragment and 'class="error"' not in fragment
            (output / f'{cas}-{method}.html').write_text(fragment, encoding='utf-8')
            for i, image in enumerate(images):
                (output / f'{cas}-{method}-{i}.png').write_bytes(base64.b64decode(image))
            observations.append(dict(cas=cas, head=head, method=method, pngs=len(images), status='pass'))
    report = dict(issue=201, scope='final renderer over saved real arrays, no model inference', observations=observations,
                  renderer_sha256=file_sha256(ROOT / 'desktop/xai_view.py'),
                  charts_sha256=file_sha256(ROOT / 'xai/charts.py'), passed=len(observations), failed=0, skipped=0)
    (HERE / 'evidence/final-display.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
