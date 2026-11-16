"""#150: #152의 디스크 캐시·워밍업과 #153의 진행률을 함께 검증한다."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict
from xai import explain, explain_cache, shap_cache


SEQ = "C" * 34


@pytest.fixture
def disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(shap_cache, "DEFAULT_DIR", tmp_path)
    result = explain.SHAPResult(np.full(34, 0.01), 0.3, 0.64, 0.0)
    shap_cache.save(shap_cache.shap_cache_key("A/Cas12a", "Cas12a", SEQ, 64, 4), result)
    return result


def test_disk_hit_skips_inference_and_progress(disk_cache):
    def unexpected(*args, **kwargs):
        pytest.fail("디스크 캐시 적중 시 추론과 진행률 콜백은 실행되면 안 된다")

    handle = SimpleNamespace(model_key="A/Cas12a", head="Cas12a", raw_fn=unexpected)
    got = explain.kernel_shap(handle, SEQ, 64, 4, on_batch=unexpected)
    np.testing.assert_array_equal(got.values, disk_cache.values)


@pytest.mark.parametrize("settings", [(128, 4), (64, 8)])
def test_changed_settings_compute_with_progress(disk_cache, settings):
    seen = []

    def raw_fn(x, on_batch=None):
        if on_batch is not None:
            for done in range(8, len(x) + 8, 8):
                on_batch(min(done, len(x)), len(x))
        return np.asarray(x)[:, :, 1].sum(axis=1) / 34

    handle = SimpleNamespace(model_key="A/Cas12a", head="Cas12a", raw_fn=raw_fn)
    got = explain.kernel_shap(handle, SEQ, *settings, on_batch=lambda d, t: seen.append(d / t))
    assert len(seen) > 1
    assert all(a <= b for a, b in zip(seen, seen[1:]))
    assert seen[-1] == pytest.approx(1.0)
    assert got.pred == pytest.approx(1.0)
    assert got.values.shape == (len(SEQ),)
    assert got is not disk_cache


def test_disk_hit_and_streamlit_rerun_leave_no_progress(disk_cache):
    from streamlit.testing.v1 import AppTest
    from xai import view

    # XAI 메모리 캐시는 코어에 있고 프로세스 하나에 하나다 (#182). 다른
    # 테스트가 남긴 항목이 이 테스트의 '1회차'를 적중으로 만들지 않게 비운다.
    explain_cache.CACHE.clear()
    script = f'''
from types import SimpleNamespace
import streamlit as st
from xai import view

class Predictor:
    def explain_handle(self, head):
        st.session_state["handle_calls"] = st.session_state.get("handle_calls", 0) + 1
        def raw_fn(*args, **kwargs):
            raise AssertionError("Inference ran despite a precomputed cache hit")
        return SimpleNamespace(model_key="A/Cas12a", head=head, raw_fn=raw_fn)

result = view.compute_shap(Predictor(), "A/Cas12a", "Cas12a", {SEQ!r}, 64, 4)
st.text(str(result.pred))
st.text(str(st.session_state["handle_calls"]))
'''
    try:
        app = AppTest.from_string(script, default_timeout=15).run()
        for _ in range(2):
            assert not app.exception
            assert not app.get("progress")
            assert [t.value for t in app.text] == ["0.64", "1"]
            app.run()
    finally:
        explain_cache.CACHE.clear()


def test_repeated_loads_warm_only_selected_model_once(monkeypatch):
    calls = []

    class Predictor:
        warmup = predict.NTClassifierPredictor.warmup

        def __init__(self, spec):
            self.spec = spec
            self.connected = True
            self.loaded = False

        def _logits(self, seqs):
            calls.append((self.spec.cas, len(seqs[0])))
            self.loaded = True

    monkeypatch.setattr(predict, "PREDICTOR_CLASSES", {key: Predictor for key in predict.MODEL_REGISTRY})
    predict.load_predictors.cache_clear()
    try:
        first = predict.load_predictors()
        for _ in range(5):
            assert predict.load_predictors() is first
        assert calls == [("Cas12a", 34)]
        assert first[("A", "Cas12a")].loaded
        assert not first[("A", "SpCas9")].loaded
    finally:
        predict.load_predictors.cache_clear()


def test_warmup_can_be_disabled(monkeypatch):
    def unexpected(self):
        pytest.fail("warmup=False인 호출은 모델을 로드하면 안 된다")

    monkeypatch.setattr(predict.NTClassifierPredictor, "warmup", unexpected)
    predict.load_predictors.cache_clear()
    try:
        predictors = predict.load_predictors(warmup=False)
        assert all(p._model is None for p in predictors.values())
    finally:
        predict.load_predictors.cache_clear()
