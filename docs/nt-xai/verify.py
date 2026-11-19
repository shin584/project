"""Real NT #201 qualification. Frozen gates may fail; never relax them after measurement.

Run one Cas at a time: py -3.13 -X utf8 docs/nt-xai/verify.py SpCas9
and then Cas12a. Raw arrays, all checks and provenance are saved even on failure.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

import core
from predict import MODEL_REGISTRY, NTClassifierPredictor, SPCAS9_VARIANTS
from xai import explain, shap_cache
from xai.identity import file_sha256
from xai.quality import RESIDUAL_ATOL_PP, RESIDUAL_RTOL, record_quality

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / 'contract.json').read_text(encoding='utf-8'))
TOL = CONTRACT['tolerances']
if (RESIDUAL_ATOL_PP != TOL['residual_atol_pp'] or RESIDUAL_RTOL != TOL['residual_rtol']):
    raise ValueError('Live residual policy does not match the frozen NT contract')


def source_hashes():
    return {p: file_sha256(ROOT / p) for p in (
        'predict.py', 'xai/explain.py', 'xai/units.py', 'xai/quality.py',
        'xai/explain_cache.py', 'xai/shap_cache.py', 'xai/charts.py',
        'desktop/xai_view.py', 'docs/nt-xai/contract.json', 'docs/nt-xai/verify.py')}


def snapshot_source(output, hashes):
    """Every explicit rerun saves the exact source alongside its new reports."""
    with zipfile.ZipFile(output / 'measured-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, digest in hashes.items():
            payload = (ROOT / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise RuntimeError(f'Source changed while freezing: {name}')
            archive.writestr(name, payload)
        archive.writestr('source-sha256.json', json.dumps(hashes, indent=2))


def main(cas):
    torch.set_num_threads(4)
    out = HERE / 'evidence'
    out.mkdir(exist_ok=True)
    case = next(c for c in CONTRACT['cases'] if c['cas'] == cas)
    seq = case['sequence']
    heads = SPCAS9_VARIANTS if cas == 'SpCas9' else ['Cas12a']
    checks, arrays = [], {}
    start = time.monotonic()
    report = dict(issue=201, cas=cas, sequence=seq, contract_sha256=file_sha256(HERE / 'contract.json'),
                  source_sha256=source_hashes(), python=sys.version, platform=platform.platform(),
                  environment={k: importlib.metadata.version(k) for k in ('numpy', 'torch', 'transformers', 'shap', 'safetensors')},
                  checks=checks, results=[], status='running')
    snapshot_source(out, report['source_sha256'])

    def save():
        report['elapsed_seconds'] = round(time.monotonic() - start, 3)
        (out / f'{cas}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        np.savez_compressed(out / f'{cas}-raw.npz', **arrays)

    def check(name, passed, **details):
        checks.append(dict(name=name, status='pass' if bool(passed) else 'fail', **details))

    def stable(name, reference, candidate, atol, rtol):
        difference = float(np.max(np.abs(candidate - reference)))
        limit = float(atol + rtol * np.max(np.abs(reference)))
        check(name, difference <= limit, max_difference_pp=difference, limit_pp=limit)

    predictor = NTClassifierPredictor(MODEL_REGISTRY['A', cas])
    try:
        ident = predictor.explanation_identity()
        report['identity'] = ident.as_dict()
        if cas == 'SpCas9':
            import pandas as pd
            parent = ROOT.parent
            train = parent / 'xai_test/dataset/SpCas9_v3/train.csv'
            notebook = parent / 'SpCas9/SpCas9_v3.ipynb'
            columns = ['SpCas9', 'SpCas9-NG', 'VRQR variant', 'xCas', 'Sniper-Cas9',
                       'SpCas9-HF.1', 'eSpCas9(1.1)', 'HypaCas9', 'evoCas9']
            df = pd.read_csv(train)
            y = df[columns].values.astype(np.float32) / 100
            mean, std = np.nanmean(y, 0), np.nanstd(y, 0)
            report['training_statistics'] = dict(train_sha256=file_sha256(train), notebook_sha256=file_sha256(notebook),
                rows=len(df), source_columns=columns, heads=list(heads), mean=mean.tolist(), std=std.tolist(),
                registered_mean=list(predictor.spec.target_mean), registered_std=list(predictor.spec.target_std))
            check('training_mean', np.allclose(mean, predictor.spec.target_mean,
                                             atol=TOL['training_stats_atol_fraction'], rtol=0))
            check('training_std', np.allclose(std, predictor.spec.target_std,
                                            atol=TOL['training_stats_atol_fraction'], rtol=0))
        ordinary = predictor._logits([seq])[0]
        arrays['ordinary_prediction'] = ordinary
        check('prediction_finite', np.isfinite(ordinary).all())

        # All heads share the same real forward function. Memoize whole raw output
        # vectors only in this measurement harness to avoid repeating identical
        # background/mask evaluations for each head and repeat. No disk cache input.
        original_logits = predictor._logits
        logits_cache = {seq: ordinary}

        def memo_logits(seqs, on_batch=None):
            missing = list(dict.fromkeys(s for s in seqs if s not in logits_cache))
            if missing:
                outputs = original_logits(missing)
                logits_cache.update(zip(missing, outputs))
            if on_batch:
                on_batch(len(seqs), len(seqs))
            return np.asarray([logits_cache[s] for s in seqs])

        predictor._logits = memo_logits
        rollout = explain.attention_rollout(predictor.explain_handle(heads[0]), seq)
        arrays['rollout_matrix'] = rollout.token_matrix
        arrays['rollout_profile'] = rollout.position_profile
        check('rollout_finite', np.isfinite(rollout.token_matrix).all() and np.isfinite(rollout.position_profile).all())
        check('rollout_nonnegative', rollout.token_matrix.min() >= TOL['rollout_minimum']
              and rollout.position_profile.min() >= TOL['rollout_minimum'])
        check('rollout_row_sums', np.allclose(rollout.token_matrix.sum(1), 1,
                                            atol=TOL['rollout_row_sum_atol'], rtol=0))
        report['rollout'] = dict(n_layers=rollout.n_layers, labels=rollout.token_labels,
                                 meaning='attention information dependency, not efficiency attribution')

        for index, head in enumerate(heads):
            print(f'{cas} {index + 1}/{len(heads)} {head}', flush=True)
            h = predictor.explain_handle(head)
            x = h.ig_inputs(seq)
            baseline = h.ig_baseline(x)
            spans = predictor._token_spans(seq)
            special = [i for i, (a, b) in enumerate(spans) if a == b]
            check(f'{head}/specials_fixed', np.array_equal(baseline[0][special], x[0][special]))
            fp, grads = h.ig_grad_fn([a[None] for a in x])
            arrays[f'h{index}_gradient_endpoint'] = grads[0]
            raw_fn_pred = float(h.raw_fn(explain._onehot(seq)[None])[0])
            check(f'{head}/raw_function', np.isclose(raw_fn_pred, ordinary[index],
                  atol=TOL['function_atol_raw'], rtol=TOL['function_rtol']),
                  raw_fn=raw_fn_pred, prediction=float(ordinary[index]))
            check(f'{head}/ig_function', np.isclose(fp[0], ordinary[index],
                  atol=TOL['function_atol_raw'], rtol=TOL['function_rtol']),
                  ig=float(fp[0]), prediction=float(ordinary[index]))
            finite_gradient = [True]
            original_grad = h.ig_grad_fn

            def watched_grad(batch):
                values, gradients = original_grad(batch)
                finite_gradient[0] &= bool(np.isfinite(values).all() and all(np.isfinite(g).all() for g in gradients))
                return values, gradients

            h.ig_grad_fn = watched_grad
            ig_results, shap_results = {}, {}
            row = dict(head=head, units=asdict(h.units), ig={}, shap={})
            report['results'].append(row)
            for steps in CONTRACT['ig']['steps']:
                result = explain.integrated_gradients(h, seq, steps)
                ig_results[steps] = result
                arrays[f'h{index}_ig{steps}'] = result.attribution
                quality = record_quality(result, ident, ('ig', head, seq, steps))
                row['ig'][str(steps)] = dict(pred_raw=result.pred, baseline_raw=result.baseline_pred,
                    pred_display=float(h.units.prediction(result.pred)), baseline_display=float(h.units.prediction(result.baseline_pred)),
                    sum_pp=float(h.units.contribution(result.attribution).sum()), error_pp=quality['residual'])
                check(f'{head}/ig{steps}/finite', np.isfinite(result.attribution).all() and finite_gradient[0])
                check(f'{head}/ig{steps}/completeness', quality['numerical_status'] == 'pass',
                      error_pp=quality['residual'], limit_pp=quality['limit'])
                print(f'  IG {steps}: residual {quality["residual"]:.5g} pp', flush=True)
            for low, high in zip(CONTRACT['ig']['steps'], CONTRACT['ig']['steps'][1:]):
                stable(f'{head}/ig{low}-{high}/stability', h.units.contribution(ig_results[high].attribution),
                       h.units.contribution(ig_results[low].attribution),
                       TOL['ig_stability_atol_pp'], TOL['ig_stability_rtol'])
            for nsamples, background in CONTRACT['shap']['settings']:
                for bg, subset in CONTRACT['shap']['seed_pairs']:
                    result = explain.kernel_shap(h, seq, nsamples, background, background_seed=bg,
                                                 subset_seed=subset, use_disk_cache=False)
                    shap_results[nsamples, background, bg, subset] = result
                    key = f'{nsamples}-{background}-{bg}-{subset}'
                    arrays[f'h{index}_shap_{key}'] = result.values
                    quality = record_quality(result, ident, ('shap', head, seq, nsamples, background, bg, subset))
                    row['shap'][key] = dict(pred_raw=result.pred, base_raw=result.base_value,
                        error_pp=quality['residual'], sum_pp=float(h.units.contribution(result.values).sum()))
                    check(f'{head}/shap{key}/finite', np.isfinite(result.values).all() and
                          np.isfinite([result.pred, result.base_value, result.additivity_error]).all())
                    check(f'{head}/shap{key}/additivity', quality['numerical_status'] == 'pass',
                          error_pp=quality['residual'], limit_pp=quality['limit'])
                reference = shap_results[nsamples, background, 0, 0]
                repeated = explain.kernel_shap(h, seq, nsamples, background, use_disk_cache=False)
                check(f'{head}/shap{nsamples}-{background}/same_seed_repeat',
                      np.max(np.abs(reference.values - repeated.values)) <= TOL['same_seed_max_difference_raw']
                      and abs(reference.base_value - repeated.base_value) <= TOL['same_seed_max_difference_raw'])
                for bg, subset in CONTRACT['shap']['seed_pairs'][1:]:
                    other = shap_results[nsamples, background, bg, subset]
                    stable(f'{head}/shap{nsamples}-{background}/seed{bg}-{subset}/stability',
                           h.units.contribution(reference.values), h.units.contribution(other.values),
                           TOL['shap_stability_atol_pp'], TOL['shap_stability_rtol'])
                print(f'  SHAP {nsamples}/{background} seeds complete', flush=True)
            default_shap = shap_results[64, 4, 0, 0]
            cache_key = shap_cache.shap_cache_key(ident, head, seq, 64, 4)
            cached_path = shap_cache.save(cache_key, default_shap, out / 'shap-cache')
            hit = shap_cache.load(cache_key, out / 'shap-cache')
            check(f'{head}/cache_roundtrip', np.array_equal(hit.values, default_shap.values)
                  and hit.units == default_shap.units and hit.pred == default_shap.pred)
            row['cache'] = dict(file=str(cached_path.relative_to(HERE)), sha256=file_sha256(cached_path))
            # Use computed real arrays to exercise all renderer branches without
            # running the model a second time. Process-owned computation is tested separately.
            from desktop.xai_view import explanation_fragment
            from pam_scanner import scan_all
            meta = json.loads((ROOT / 'docs/equivalence-baseline/capture.json').read_text(encoding='utf-8'))
            target = next(t for t in meta['xai'] if t['cas_type'] == cas)
            site = next(s for s in scan_all(meta['input_seq']).sites if s.site_key == target['site_key'])
            cache = SimpleNamespace(integrated_gradients=lambda *a, **k: ig_results[32],
                                    kernel_shap=lambda *a, **k: default_shap,
                                    attention_rollout=lambda *a, **k: rollout)
            for method, params in [('ig', {'steps': 32}), ('shap', {'nsamples': 64, 'n_background': 4}), ('rollout', {})]:
                fragment = explanation_fragment(predictor, site, head, method, params, 80, lambda *a: None, cache=cache)
                count = len(re.findall('src="data:image/png;base64,', fragment))
                check(f'{head}/{method}/display', count == (1 if method == 'ig' else 2)
                      and 'class="warning"' not in fragment and 'class="error"' not in fragment, png_count=count)
                if index == 0:
                    display_dir = ROOT / '.packaging/evidence/issue201'
                    display_dir.mkdir(parents=True, exist_ok=True)
                    (display_dir / f'{cas}-{method}.html').write_text(fragment, encoding='utf-8')
            save()
            gc.collect()
        report['source_unchanged'] = source_hashes() == report['source_sha256']
        check('source_unchanged', report['source_unchanged'])
        report['counts'] = {status: sum(c['status'] == status for c in checks) for status in ('pass', 'fail', 'skip')}
        report['status'] = 'pass' if not report['counts']['fail'] else 'numerical_failures_recorded'
        report['raw_sha256'] = None
        save()
        report['raw_sha256'] = file_sha256(out / f'{cas}-raw.npz')
        (out / f'{cas}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'cas': cas, 'status': report['status'], 'counts': report['counts'], 'seconds': report['elapsed_seconds']}), flush=True)
        return 0 if report['status'] == 'pass' else 1
    except BaseException as error:
        report.update(status='incomplete', exception=repr(error))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
