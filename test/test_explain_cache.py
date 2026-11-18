"""XAI 메모리 캐시(#182). 가중치 없이 가짜 predictor/handle로 돈다.

확인하는 것
  1) 같은 키로 두 번 부르면 두 번째는 계산 함수를 부르지 않는다
  2) 콜백만 다른 두 호출이 같은 캐시에 적중한다 (예측기 객체도 키가 아니다)
  3) 설정이 다르면 미적중 (기법·모델·head·서열·기법별 설정 전부)
  4) 적중이면 explain_handle조차 부르지 않는다 (모델을 올릴 수 있다)
  5) 캐시는 Streamlit을 import하지 않는다
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xai import explain, explain_cache

SEQ = "ACGTACGTAGGCTTACGATCGATCGGATCCAT"
ROOT = Path(__file__).resolve().parent.parent


class FakePredictor:
    """explain_handle 호출을 센다. 캐시 적중이면 늘어나지 않아야 한다."""

    def __init__(self, label="p"):
        self.label = label
        self.handle_calls = 0

    def explain_handle(self, head):
        self.handle_calls += 1
        return f"handle({self.label},{head})"


@pytest.fixture
def calls(monkeypatch):
    """계산 함수 셋을 기록만 하는 가짜로 바꾼다. 호출마다 새 결과 객체를 준다."""
    recorded = []

    def fake_ig(handle, window_seq, steps, chunk_size=explain.IG_CHUNK_SIZE, on_batch=None):
        recorded.append(("ig", handle, window_seq, steps, on_batch))
        return {"kind": "ig", "n": len(recorded)}

    def fake_shap(handle, window_seq, nsamples=512, n_background=16, seed=0, on_batch=None,
                  *, use_disk_cache=True):
        recorded.append(("shap", handle, window_seq, nsamples, n_background, on_batch))
        return {"kind": "shap", "n": len(recorded)}

    def fake_rollout(handle, window_seq):
        recorded.append(("rollout", handle, window_seq))
        return {"kind": "rollout", "n": len(recorded)}

    monkeypatch.setattr(explain, "integrated_gradients", fake_ig)
    monkeypatch.setattr(explain, "kernel_shap", fake_shap)
    monkeypatch.setattr(explain, "attention_rollout", fake_rollout)
    return recorded


@pytest.fixture
def cache():
    return explain_cache.ExplainCache()


# ── 1) 같은 키 → 두 번째는 계산하지 않는다 ──────────────────────────
def test_second_ig_call_with_the_same_key_reuses_the_result(calls, cache):
    predictor = FakePredictor()
    first = cache.integrated_gradients(predictor, "A/Cas12a", "Cas12a", SEQ, 32)
    second = cache.integrated_gradients(predictor, "A/Cas12a", "Cas12a", SEQ, 32)

    assert second is first
    assert len(calls) == 1
    assert predictor.handle_calls == 1, "적중인데 handle을 다시 만들었다 (모델을 올릴 수 있다)"


def test_second_shap_call_with_the_same_key_reuses_the_result(calls, cache):
    predictor = FakePredictor()
    first = cache.kernel_shap(predictor, "A/Cas12a", "Cas12a", SEQ, 64, 4)
    second = cache.kernel_shap(predictor, "A/Cas12a", "Cas12a", SEQ, 64, 4)

    assert second is first
    assert len(calls) == 1
    assert predictor.handle_calls == 1


def test_second_rollout_call_with_the_same_key_reuses_the_result(calls, cache):
    predictor = FakePredictor()
    first = cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", SEQ)
    second = cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", SEQ)

    assert second is first
    assert len(calls) == 1
    assert predictor.handle_calls == 1


def test_the_computed_arguments_reach_the_core_function(calls, cache):
    cache.kernel_shap(FakePredictor(), "A/Cas12a", "Cas12a", SEQ, 64, 4)
    kind, handle, seq, nsamples, n_background, _ = calls[0]
    assert (kind, handle, seq, nsamples, n_background) == (
        "shap", "handle(p,Cas12a)", SEQ, 64, 4)


# ── 2) 콜백·예측기 객체는 키가 아니다 ───────────────────────────────
def test_calls_differing_only_by_callback_hit_the_same_entry(calls, cache):
    predictor = FakePredictor()
    first = cache.kernel_shap(predictor, "A/Cas12a", "Cas12a", SEQ, 64, 4,
                              on_batch=lambda done, total: None)
    second = cache.kernel_shap(predictor, "A/Cas12a", "Cas12a", SEQ, 64, 4,
                               on_batch=lambda done, total: None)

    assert second is first, "콜백이 키에 들어갔다 — 호출마다 다른 함수 객체라 매번 빗나간다"
    assert len(calls) == 1
    assert calls[0][-1] is not None, "미적중 계산에는 콜백이 전달되어야 한다"


def test_ig_calls_differing_only_by_callback_hit_the_same_entry(calls, cache):
    predictor = FakePredictor()
    first = cache.integrated_gradients(predictor, "A/Cas12a", "Cas12a", SEQ, 32,
                                       on_batch=lambda done, total: None)
    second = cache.integrated_gradients(predictor, "A/Cas12a", "Cas12a", SEQ, 32,
                                        on_batch=lambda done, total: None)

    assert second is first
    assert len(calls) == 1


def test_a_different_predictor_object_hits_the_same_entry(calls, cache):
    """Streamlit은 재실행마다 예측기를 다시 얻는다. 객체가 달라도 같은 모델이다."""
    first = cache.kernel_shap(FakePredictor("a"), "A/Cas12a", "Cas12a", SEQ, 64, 4)
    other = FakePredictor("b")
    second = cache.kernel_shap(other, "A/Cas12a", "Cas12a", SEQ, 64, 4)

    assert second is first
    assert len(calls) == 1
    assert other.handle_calls == 0


# ── 3) 설정이 다르면 미적중 ─────────────────────────────────────────
@pytest.mark.parametrize("second", [
    pytest.param(dict(steps=64), id="steps"),
    pytest.param(dict(model_key="A/SpCas9"), id="model_key"),
    pytest.param(dict(head="SpCas9-HF1"), id="head"),
    pytest.param(dict(window_seq="T" + SEQ[1:]), id="window_seq"),
])
def test_ig_settings_each_make_a_new_entry(calls, cache, second):
    base = dict(model_key="A/Cas12a", head="Cas12a", window_seq=SEQ, steps=32)
    predictor = FakePredictor()
    cache.integrated_gradients(predictor, **base)
    cache.integrated_gradients(predictor, **{**base, **second})

    assert len(calls) == 2


@pytest.mark.parametrize("second", [
    pytest.param(dict(nsamples=128), id="nsamples"),
    pytest.param(dict(n_background=8), id="n_background"),
])
def test_shap_settings_each_make_a_new_entry(calls, cache, second):
    base = dict(model_key="A/Cas12a", head="Cas12a", window_seq=SEQ,
                nsamples=64, n_background=4)
    predictor = FakePredictor()
    cache.kernel_shap(predictor, **base)
    cache.kernel_shap(predictor, **{**base, **second})

    assert len(calls) == 2


def test_techniques_do_not_share_an_entry(calls, cache):
    predictor = FakePredictor()
    ig = cache.integrated_gradients(predictor, "A/Cas12a", "Cas12a", SEQ, 32)
    shap = cache.kernel_shap(predictor, "A/Cas12a", "Cas12a", SEQ, 64, 4)
    rollout = cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", SEQ)

    assert {r["kind"] for r in (ig, shap, rollout)} == {"ig", "shap", "rollout"}
    assert len(calls) == 3


def test_the_sequence_is_keyed_case_insensitively(calls, cache):
    predictor = FakePredictor()
    first = cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", SEQ)
    second = cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", SEQ.lower())

    assert second is first
    assert len(calls) == 1


# ── 살림 ────────────────────────────────────────────────────────────
def test_clear_drops_everything(calls, cache):
    predictor = FakePredictor()
    cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", SEQ)
    cache.clear()
    cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", SEQ)

    assert len(calls) == 2
    assert len(cache) == 1


def test_the_oldest_unused_entry_goes_first_when_full(calls):
    cache = explain_cache.ExplainCache(max_entries=2)
    predictor = FakePredictor()

    first = cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", "AAAA")
    stale = cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", "CCCC")
    assert cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", "AAAA") is first

    cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", "GGGG")  # 밀려나는 건 CCCC
    assert len(cache) == 2
    assert cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", "AAAA") is first, \
        "방금 쓴 항목이 밀려났다"
    assert cache.attention_rollout(predictor, "A/Cas12a", "Cas12a", "CCCC") is not stale, \
        "밀려났어야 할 항목이 남아 있다"
    assert len(calls) == 4


# ── 코어다: 화면 프레임워크를 모른다 ────────────────────────────────
def test_the_cache_does_not_import_streamlit():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from xai import explain_cache; "
         "assert 'streamlit' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ── 실제 계산을 한 번 통과시킨다 (가짜 handle, 가중치 없음) ─────────
class _LinearHandle:
    """ig_* 네 메서드만 있는 최소 handle. 선형 모델이라 IG 값이 확정적이다."""

    def __init__(self):
        self.grad_calls = 0

    def ig_inputs(self, window_seq):
        return [np.arange(len(window_seq), dtype=np.float64) / len(window_seq)]

    def ig_grad_fn(self, batch):
        self.grad_calls += 1
        x = np.asarray(batch[0], dtype=np.float64)
        return x.sum(axis=1), [np.ones_like(x)]

    def project(self, attrs):
        return attrs[0]


def test_a_real_ig_result_is_returned_and_then_reused():
    cache = explain_cache.ExplainCache()
    handle = _LinearHandle()

    class Predictor:
        def explain_handle(self, head):
            return handle

    first = cache.integrated_gradients(Predictor(), "A/Cas12a", "Cas12a", SEQ, 16)
    after_first = handle.grad_calls
    second = cache.integrated_gradients(Predictor(), "A/Cas12a", "Cas12a", SEQ, 16)

    assert isinstance(first, explain.IGResult)
    assert first.completeness_error < 1e-9
    assert second is first
    assert handle.grad_calls == after_first, "적중인데 모델을 다시 돌렸다"
