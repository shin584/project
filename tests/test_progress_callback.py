"""진행률 콜백(#153) 테스트. 가중치 없이 돈다 (가짜 handle + 가짜 torch).

확인하는 것
  1) on_batch를 안 넘기면 동작과 값이 전과 같다 (기존 호출부 보호)
  2) 콜백이 kernel_shap → raw_fn → _logits 세 지점을 통과한다
  3) 진행률이 뒤로 가지 않고 0~1 안에 있으며 마지막에 100%에 닿는다
  4) app.py 기본값(샘플 64 × 배경 4)에서 샘플 구간이 32단계로 올라간다
  5) _logits가 서열 수를 배치 크기로 나눈 만큼 정확히 콜백을 부른다
  6) IG 청킹 전후로 기여도와 completeness_error가 같다 (4b)
"""

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xai import explain                                  # noqa: E402

NSAMPLES = 64          # app.py의 'SHAP 샘플 수' 기본값
N_BACKGROUND = 4       # app.py의 'SHAP 배경 서열 수' 기본값
BATCH = 8              # predict.NT_BATCH_SIZE
SEQ = "ACGTACGTAGGCTTACGATCGATCGGATCCAT"      # 32 bp


# ══════════════════════════════════════════════════════
# SHAP 쪽 (4번)
# ══════════════════════════════════════════════════════

class _BatchingHandle:
    """predict.py의 raw_fn과 같은 모양. _logits의 배치 루프를 그대로 흉내낸다."""

    def __init__(self):
        self.calls = 0
        self.got_on_batch = False

    def raw_fn(self, X, on_batch=None):
        X = np.asarray(X)
        total = X.shape[0]
        self.calls += 1
        self.got_on_batch = self.got_on_batch or on_batch is not None
        out = []
        for i in range(0, total, BATCH):
            chunk = X[i:i + BATCH]
            out.append(chunk.reshape(chunk.shape[0], -1).sum(axis=1) / 100.0)
            if on_batch is not None:
                on_batch(min(i + BATCH, total), total)
        return np.concatenate(out)


class _OneArgHandle(_BatchingHandle):
    """on_batch를 아예 모르는 옛 handle. 콜백 없이 부를 때 깨지지 않아야 한다."""

    def raw_fn(self, X):           # noqa: D102
        return super().raw_fn(X)


def _run(handle, on_batch=None, nsamples=NSAMPLES, n_background=N_BACKGROUND):
    return explain.kernel_shap(handle, SEQ, nsamples=nsamples,
                               n_background=n_background, seed=0, on_batch=on_batch)


def test_without_callback_old_handles_still_work():
    """on_batch를 안 넘기면 raw_fn을 인자 하나로 부른다 (기존 handle 보호)."""
    res = _run(_OneArgHandle())
    assert res.values.shape == (len(SEQ),)
    assert res.additivity_error == pytest.approx(0.0, abs=1e-6)


def test_callback_reaches_the_batch_loop():
    handle = _BatchingHandle()
    seen = []
    _run(handle, on_batch=lambda d, t: seen.append((d, t)))
    assert handle.got_on_batch, "콜백이 raw_fn까지 내려가지 않았다"
    assert len(seen) > 1, "f(masks) 안에 훅이 걸려 단계가 뭉갰다"


def test_progress_is_monotonic_and_bounded():
    seen = []
    _run(_BatchingHandle(), on_batch=lambda d, t: seen.append((d, t)))
    pcts = [d / t for d, t in seen]
    assert all(0.0 <= p <= 1.0 for p in pcts), "진행률이 0~1을 벗어났다"
    assert all(b >= a for a, b in zip(pcts, pcts[1:])), "진행률이 뒤로 갔다"
    assert pcts[-1] == pytest.approx(1.0), "100%에 닿지 않았다"


def test_sample_phase_has_about_32_steps():
    """ADR 0001이 말한 '32단계'. 샘플 256회 ÷ 배치 8 = 32."""
    seen = []
    _run(_BatchingHandle(), on_batch=lambda d, t: seen.append((d, t)))
    expected_sample_steps = math.ceil(NSAMPLES * N_BACKGROUND / BATCH)
    # 배경 기준값 평가분이 앞에 몇 단계 더 붙는다. 그만큼만 여유를 둔다.
    assert expected_sample_steps <= len(seen) <= expected_sample_steps + 4


def test_callback_does_not_change_the_numbers():
    plain = _run(_BatchingHandle())
    with_cb = _run(_BatchingHandle(), on_batch=lambda d, t: None)
    assert np.allclose(plain.values, with_cb.values)
    assert plain.base_value == pytest.approx(with_cb.base_value)


@pytest.mark.parametrize("nsamples,n_background", [(16, 2), (64, 4), (128, 8)])
def test_progress_holds_for_other_settings(nsamples, n_background):
    seen = []
    _run(_BatchingHandle(), on_batch=lambda d, t: seen.append((d, t)),
         nsamples=nsamples, n_background=n_background)
    pcts = [d / t for d, t in seen]
    assert all(0.0 <= p <= 1.0 for p in pcts)
    assert all(b >= a for a, b in zip(pcts, pcts[1:]))


# ══════════════════════════════════════════════════════
# predict._logits 배치 루프 (4번의 훅 지점)
# ══════════════════════════════════════════════════════

@pytest.fixture
def nt_predictor(monkeypatch):
    """가중치도 torch도 없이 _logits만 돌 수 있게 만든 NTClassifierPredictor.

    _logits가 함수 안에서 import torch를 하므로 sys.modules에 가짜를 꽂으면 된다.
    여기서 보려는 것은 배치 루프와 콜백 호출 횟수뿐이라 텐서 연산은 필요 없다.
    """
    class _Arr:
        def __init__(self, a): self.a = a
        def detach(self): return self
        def float(self): return self
        def numpy(self): return self.a

    class _NoGrad:
        def __enter__(self): return None
        def __exit__(self, *exc): return False

    fake = types.ModuleType("torch")
    fake.no_grad = _NoGrad
    fake.sigmoid = lambda x: x
    monkeypatch.setitem(sys.modules, "torch", fake)

    import predict

    p = predict.NTClassifierPredictor(predict.MODEL_REGISTRY[("A", "Cas12a")])
    p._num_labels = 1
    p._tokenizer = object()
    p._dtype = None
    p._model = lambda **enc: types.SimpleNamespace(
        logits=_Arr(np.zeros((enc["n"], 1), dtype=np.float32)))
    p._encode = lambda seqs: {"n": len(seqs)}
    return p, predict


@pytest.mark.parametrize("n", [1, 8, 9, 40, 256])
def test_logits_calls_callback_once_per_batch(nt_predictor, n):
    p, predict = nt_predictor
    seen = []
    out = p._logits(["A" * 30] * n, on_batch=lambda d, t: seen.append((d, t)))
    assert out.shape == (n, 1)
    assert len(seen) == math.ceil(n / predict.NT_BATCH_SIZE)
    assert seen[-1] == (n, n), "마지막 단계가 전체에 닿지 않았다"
    assert all(t == n for _, t in seen), "total이 중간에 바뀌었다"


def test_logits_without_callback(nt_predictor):
    """기존 호출부(predict_batch 등)는 인자를 안 넘긴다."""
    p, _ = nt_predictor
    assert p._logits(["A" * 30] * 20).shape == (20, 1)


# ══════════════════════════════════════════════════════
# IG 배치 청킹 (4b번)
# ══════════════════════════════════════════════════════

T, H, L = 9, 16, 32                                   # 토큰 9개 × 임베딩 16, 서열 32bp
_RNG = np.random.default_rng(7)
_EMB = _RNG.normal(size=(T, H)).astype(np.float32)
_W = _RNG.normal(size=(T, H)).astype(np.float32)
_SPANS = [(0, 0)] + [(i * 6, min(i * 6 + 6, L)) for i in range(T - 1)]


class _IGHandle:
    """f(x) = Σ W·x² → ∇f = 2·W·x. 미분 가능하고 완결성이 성립한다."""

    def __init__(self):
        self.max_batch = 0

    def ig_inputs(self, seq):
        return [_EMB]

    def ig_grad_fn(self, batch):
        x = np.asarray(batch[0], dtype=np.float32)
        self.max_batch = max(self.max_batch, x.shape[0])
        return (_W[None] * x ** 2).sum(axis=(1, 2)), [2.0 * _W[None] * x]

    def project(self, attrs):
        per_token = np.asarray(attrs[0]).sum(axis=-1)
        pos = np.zeros(L, dtype=np.float64)
        for v, (a, b) in zip(per_token, _SPANS):
            if b > a:
                pos[a:b] += v / (b - a)
        return pos


def _ig_one_batch(handle, steps):
    """청킹 전 구현. 적분 단계를 한 배치에 모두 올린다."""
    inputs = handle.ig_inputs("x")
    alphas = (np.arange(steps) + 0.5) / steps
    batch = [alphas.reshape((-1,) + (1,) * x.ndim) * x[None] for x in inputs]
    _, grads = handle.ig_grad_fn(batch)
    attrs = [x * g.mean(axis=0) for x, g in zip(inputs, grads)]
    f_x = float(handle.ig_grad_fn([x[None] for x in inputs])[0][0])
    f_0 = float(handle.ig_grad_fn([np.zeros_like(x)[None] for x in inputs])[0][0])
    total = float(sum(a.sum() for a in attrs))
    return np.asarray(handle.project(attrs)), abs(total - (f_x - f_0))


@pytest.mark.parametrize("steps", [8, 32, 64, 100])
def test_ig_chunking_matches_single_batch(steps):
    old_attr, _ = _ig_one_batch(_IGHandle(), steps)
    new = explain.integrated_gradients(_IGHandle(), "A" * L, steps=steps)
    assert np.allclose(old_attr, new.attribution, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("steps", [8, 32, 64, 100])
def test_ig_completeness_stays_at_rounding_level(steps):
    new = explain.integrated_gradients(_IGHandle(), "A" * L, steps=steps)
    scale = abs(new.pred - new.baseline_pred)
    # 기존 구현도 이 수준이라 '일치'로 본다. 우연히 0이 나오는 경우가 있어
    # 기존 값에 배수를 거는 기준은 쓰지 않는다.
    assert new.completeness_error / scale < 1e-6


def test_ig_chunk_size_does_not_change_the_result():
    base = explain.integrated_gradients(_IGHandle(), "A" * L, steps=64, chunk_size=64)
    for cs in (1, 3, 8, 16, 128):
        got = explain.integrated_gradients(_IGHandle(), "A" * L, steps=64, chunk_size=cs)
        assert np.allclose(base.attribution, got.attribution, rtol=1e-5, atol=1e-8), cs


def test_ig_chunking_caps_the_batch_size():
    handle = _IGHandle()
    explain.integrated_gradients(handle, "A" * L, steps=100, chunk_size=8)
    assert handle.max_batch <= 8, "청크 크기를 넘는 배치가 역전파로 올라갔다"
