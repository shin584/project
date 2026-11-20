"""#199: actual SpCas9 XAI at the predictor, renderer and cancellation boundaries."""

import logging
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import core
import non_nt_predict
import predict
from desktop.api import Cancelled, XAI_DEFAULTS
from desktop.worker_renderer import NonNTComputationRenderer
from desktop.xai_view import explanation_fragment
from test_desktop_xai import DNA, pngs
from xai.explain import IGResult, SHAPResult
from xai.explain_cache import ExplainCache
from xai.quality import record_quality


@pytest.fixture(scope="module")
def predictor():
    return non_nt_predict.NonNTSpCas9Predictor(predict.NON_NT_REGISTRY[("A", "SpCas9")])


@pytest.mark.parametrize("method,count", [("ig", 1), ("shap", 2)])
def test_actual_non_nt_charts_resize_and_protocol_cache(predictor, method, count, monkeypatch):
    view = NonNTComputationRenderer({("A", "SpCas9"): predictor})
    progress = []

    def render(params=None, size=80):
        return view.explanation_fragment(
            DNA, "SpCas9|+|40", "SpCas9-NG", method,
            params or XAI_DEFAULTS[method], size, lambda d, t: progress.append((d, t)))

    first = render(size=40)
    assert len(pngs(first)) == count
    assert "%p" in first
    assert all(text not in first for text in ("제한 전", "인과관계", "기술적 참고", "출력 단위 참고"))
    assert progress and progress[-1][0] == progress[-1][1]
    assert all(a[0] / a[1] <= b[0] / b[1] for a, b in zip(progress, progress[1:]))
    progress.clear()
    second = render(size=100)
    assert pngs(first) == pngs(second)
    assert 'width:40%' in first and 'width:100%' in second
    assert not progress
    settings = {"steps": 64} if method == "ig" else {"nsamples": 128, "n_background": 4}
    render(settings)
    assert progress
    progress.clear()
    monkeypatch.setattr(non_nt_predict, "SPCAS9_EXPLANATION_PROTOCOL", "test-new-protocol")
    render()
    assert progress, "a protocol change must invalidate both chart and result caches"


@pytest.mark.parametrize("method", ["ig", "shap"])
def test_precision_calculation_cancels_without_caching_and_can_resume(predictor, method):
    cache = ExplainCache(use_disk_cache=False)
    identity = predictor.explanation_identity()
    window = "CCGGTCGCTTTAAGAATAAATAAAAGCGCC"
    progress = []

    def cancel(done, total):
        progress.append((done, total))
        raise Cancelled

    def compute(callback):
        if method == "ig":
            return cache.integrated_gradients(predictor, identity, "SpCas9-NG", window, 256,
                                              on_batch=callback)
        return cache.kernel_shap(predictor, identity, "SpCas9-NG", window, 512, 32,
                                 on_batch=callback)

    with pytest.raises(Cancelled):
        compute(cancel)
    assert progress and len(cache) == 0
    result = compute(lambda d, t: None)
    assert np.isfinite(result.pred) and len(cache) == 1


@pytest.mark.parametrize("method", ["ig", "shap"])
def test_finite_quality_failure_keeps_graphs(predictor, method):
    site = next(s for s in core.scan_all(DNA).sites if s.site_key == "SpCas9|+|40")
    # Deliberately outside tolerance; the view must render supplied finite results.
    ig = IGResult(np.full(30, .01), 1.2, .1, .8)
    shap = SHAPResult(np.full(30, .01), .1, 1.2, .8, 0, 0)
    cache = SimpleNamespace(integrated_gradients=lambda *a, **kw: ig,
                            kernel_shap=lambda *a, **kw: shap)
    fragment = explanation_fragment(predictor, site, "SpCas9-NG", method,
                                    XAI_DEFAULTS[method], 80, lambda d, t: None, cache=cache)
    assert len(pngs(fragment)) == (1 if method == "ig" else 2)
    assert 'class="warning"' not in fragment and 'role="alert"' not in fragment
    assert "품질 경고" not in fragment
    assert all(text not in fragment for text in ("제한 전", "표와 다릅니다", "배경 seed"))


@pytest.mark.parametrize("method", ["ig", "shap"])
def test_non_nt_quality_uses_frozen_bound_and_persists_internal_failure(predictor, method, caplog):
    # .006 raw is .6%p: fails the non-NT absolute .5%p bound, even though
    # NT's .5%p + 5% of the 100%p signal would pass it.
    result = (IGResult(np.full(30, .01), 1.1, .1, .006) if method == "ig" else
              SHAPResult(np.full(30, .01), .1, 1.1, .006, 0, 0))
    identity = predictor.explanation_identity()
    request = (method, "SpCas9-NG", "A" * 30, 32)
    with caplog.at_level(logging.INFO, logger="xai.quality"):
        record = record_quality(result, identity, request)
    assert record["limit"] == .5 and record["numerical_status"] == "fail"
    assert '"numerical_status": "fail"' in caplog.text
    assert '"family": "non_nt"' in caplog.text
    path = Path(os.environ["LOCALAPPDATA"]) / "dashboard-v3/logs/non_nt-xai.jsonl"
    assert '"numerical_status": "fail"' in path.read_text(encoding="utf-8")
    nt_identity = replace(identity, family="nt", model="A/SpCas9")
    nt_record = record_quality(result, nt_identity, request)
    assert nt_record["limit"] == pytest.approx(5.5)
    assert nt_record["numerical_status"] == "pass", "#201's NT criterion must stay unchanged"
    bad = replace(result, attribution=np.full(30, np.nan)) if method == "ig" else replace(
        result, values=np.full(30, np.inf))
    with pytest.raises(ValueError, match="NaN/inf"):
        record_quality(bad, identity, request)


def test_non_nt_spcas9_rollout_is_unavailable(predictor):
    view = NonNTComputationRenderer({("A", "SpCas9"): predictor})
    with pytest.raises((NotImplementedError, ValueError), match="attention|Attention"):
        view.explanation_fragment(DNA, "SpCas9|+|40", "SpCas9-NG", "rollout", {},
                                  80, lambda d, t: None)
