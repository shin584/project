"""#200: composite Cas12a explanations at the existing predictor/UI/cache seams."""

import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import core
from desktop.api import Cancelled, XAI_DEFAULTS
from desktop.worker_renderer import NonNTComputationRenderer
from desktop.xai_view import explanation_fragment
from non_nt_cas12a_predict import MANIFEST_CAS_CONFIGS, NonNTCas12aPredictor
from non_nt_runtime import CAS12A_IG_PROTOCOLS, cas12a_derive, one_hot
from predict import NON_NT_REGISTRY
from test_desktop_xai import DNA, pngs
from xai.explain import IGResult, SHAPResult, kernel_shap
from xai.explain_cache import ExplainCache
from xai.quality import record_quality

WINDOW = "TTTAAGAATAAATAAAAGCGCCACAGGATTGACT"


@pytest.fixture(scope="module")
def predictor():
    return NonNTCas12aPredictor(NON_NT_REGISTRY[("A", "Cas12a")])


def test_predictor_uses_adopted_single_sequence_handle_and_identity(predictor):
    handle = predictor.explain_handle("Cas12a")
    assert handle.ig_path == "composite"
    assert handle.head == "Cas12a"
    assert handle.model_key == predictor.explanation_identity()
    assert handle.model_key.protocol == CAS12A_IG_PROTOCOLS["composite"]
    assert handle.model_key.checkpoints == (predictor.model_identity().checkpoint_sha256,)
    assert [x.shape for x in handle.ig_inputs(WINDOW)] == [(34, 4)]
    with pytest.raises(ValueError, match="Cas12a"):
        predictor.explain_handle("SpCas9-NG")


def test_shap_recomputes_all_features_and_matches_original_for_every_mixed_sequence(predictor, monkeypatch):
    handle = predictor.explain_handle("Cas12a")
    forward = predictor.runtime.forward
    seen = []

    def check_inputs(arrays, **kwargs):
        x, features = arrays
        sequences = ["".join("ACGT"[i] for i in row.argmax(axis=1)) for row in x]
        expected = np.stack([cas12a_derive(seq) for seq in sequences])
        np.testing.assert_array_equal(features, expected)
        seen.extend(sequences)
        return forward(arrays, **kwargs)

    monkeypatch.setattr(predictor.runtime, "forward", check_inputs)
    result = kernel_shap(handle, WINDOW, nsamples=64, n_background=4, use_disk_cache=False)
    assert len(set(seen)) > 34
    assert any(s[4:8] != WINDOW[4:8] for s in seen)
    original = forward([one_hot(WINDOW)[None], cas12a_derive(WINDOW)[None]])[0]
    assert result.pred == pytest.approx(original, abs=1e-6)
    assert result.values.shape == (34,) and np.isfinite(result.values).all()


@pytest.mark.parametrize("method,count", [("ig", 1), ("shap", 2)])
def test_actual_cas12a_charts_resize_reuse_and_path_cache_rejection(predictor, method, count, monkeypatch):
    renderer = NonNTComputationRenderer({("A", "Cas12a"): predictor})
    progress = []

    def render(size):
        return renderer.explanation_fragment(DNA, "Cas12a|+|16", "Cas12a", method,
                                             XAI_DEFAULTS[method], size,
                                             lambda d, t: progress.append((d, t)))

    first = render(40)
    assert len(pngs(first)) == count
    assert "354" in first and "제한 전" in first and "%p" in first
    assert progress and progress[-1][0] == progress[-1][1]
    assert all(a[0] / a[1] <= b[0] / b[1] for a, b in zip(progress, progress[1:]))
    progress.clear()
    second = render(100)
    assert pngs(first) == pngs(second)
    assert 'width:40%' in first and 'width:100%' in second
    assert not progress
    monkeypatch.setitem(CAS12A_IG_PROTOCOLS, "composite", "test-different-path")
    render(80)
    assert progress, "a changed path protocol must invalidate result and chart caches"


@pytest.mark.parametrize("method", ["ig", "shap"])
def test_cas12a_precision_can_cancel_without_caching_then_resume(predictor, method):
    cache = ExplainCache(use_disk_cache=False)
    identity = predictor.explanation_identity()
    progress = []

    def cancel(done, total):
        progress.append((done, total))
        raise Cancelled

    def compute(callback):
        if method == "ig":
            return cache.integrated_gradients(predictor, identity, "Cas12a", WINDOW, 256,
                                              on_batch=callback)
        return cache.kernel_shap(predictor, identity, "Cas12a", WINDOW, 512, 32,
                                 on_batch=callback)

    with pytest.raises(Cancelled):
        compute(cancel)
    assert progress and len(cache) == 0
    assert np.isfinite(compute(lambda d, t: None).pred) and len(cache) == 1


@pytest.mark.parametrize("method", ["ig", "shap"])
def test_cas12a_quality_failure_keeps_graphs_and_logs_frozen_absolute_limit(predictor, method, caplog):
    # .002 raw = .2%p, above Cas12a's frozen .1%p threshold.
    result = (IGResult(np.full(34, .01), 1.2, .1, .002) if method == "ig" else
              SHAPResult(np.full(34, .01), .1, 1.2, .002, 0, 0))
    identity = predictor.explanation_identity()
    with caplog.at_level(logging.INFO, logger="xai.quality"):
        record = record_quality(result, identity, (method, "Cas12a", "A" * 34, 32))
    assert record["limit"] == .1 and record["numerical_status"] == "fail"
    assert '"numerical_status": "fail"' in caplog.text
    cache = SimpleNamespace(integrated_gradients=lambda *a, **kw: result,
                            kernel_shap=lambda *a, **kw: result)
    site = next(s for s in core.scan_all(DNA, configs=MANIFEST_CAS_CONFIGS).sites
                if s.site_key == "Cas12a|+|16")
    fragment = explanation_fragment(predictor, site, "Cas12a", method,
                                    XAI_DEFAULTS[method], 80, lambda d, t: None, cache=cache)
    assert len(pngs(fragment)) == (1 if method == "ig" else 2)
    assert 'class="warning"' not in fragment and 'role="alert"' not in fragment
    assert "120.00%" in fragment and "표와 다릅니다" in fragment
    assert "354" in fragment and "PAM" in fragment and "인과관계" in fragment
    if method == "ig":
        assert "soft" in fragment and "영벡터" in fragment and "합성 경로" in fragment
    else:
        assert "무작위 ACGT" in fragment and "재계산" in fragment
    bad = (replace(result, attribution=np.full(34, np.nan)) if method == "ig" else
           replace(result, values=np.full(34, np.inf)))
    with pytest.raises(ValueError, match="NaN/inf"):
        record_quality(bad, identity, (method, "Cas12a", "A" * 34, 32))
    short = (replace(result, attribution=np.zeros(30)) if method == "ig" else
             replace(result, values=np.zeros(30)))
    with pytest.raises(ValueError, match="length"):
        record_quality(short, identity, (method, "Cas12a", "A" * 34, 32))


def test_cas12a_rollout_rejected_with_reason(predictor):
    renderer = NonNTComputationRenderer({("A", "Cas12a"): predictor})
    with pytest.raises(NotImplementedError, match="attention"):
        renderer.explanation_fragment(DNA, "Cas12a|+|16", "Cas12a", "rollout", {},
                                      80, lambda d, t: None)
