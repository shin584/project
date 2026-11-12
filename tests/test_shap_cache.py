"""SHAP 사전계산 캐시 테스트. 가중치 없이 돈다 (저장된 npz + 가짜 handle).

확인하는 것
  1) 사전계산한 서열 + 같은 설정 → 적중, 저장된 값 그대로
  2) 설정이 다르면 미적중 (nsamples만 바꿔도, n_background만 바꿔도)
  3) 적중 시 모델 추론이 한 번도 일어나지 않는다
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xai import explain, shap_cache                      # noqa: E402
from xai.shap_cache import shap_cache_key                # noqa: E402

MODEL_KEY = "A/Cas12a"
HEAD = "Cas12a"
SEQ = "A" * 34
NSAMPLES = 64          # app.py의 'SHAP 샘플 수' 기본값
N_BACKGROUND = 4       # app.py의 'SHAP 배경 서열 수' 기본값


class _SpyHandle:
    """model_key·head만 실어 나르는 가짜 handle. raw_fn이 불리면 기록한다."""

    def __init__(self, model_key=MODEL_KEY, head=HEAD):
        self.model_key = model_key
        self.head = head
        self.calls = 0

    def raw_fn(self, X):
        self.calls += 1
        return np.zeros(len(np.asarray(X)), dtype=np.float64)


def _result(length=34, fill=0.01):
    return explain.SHAPResult(
        values=np.full(length, fill, dtype=np.float64),
        base_value=0.3, pred=0.3 + length * fill, additivity_error=0.0,
    )


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """사전계산 파일 한 개(기본 설정)를 심은 임시 저장소."""
    monkeypatch.setattr(shap_cache, "DEFAULT_DIR", tmp_path)
    shap_cache.save(shap_cache_key(MODEL_KEY, HEAD, SEQ, NSAMPLES, N_BACKGROUND),
                    _result(), root=tmp_path)
    return tmp_path


# ── 캐시 키 ──────────────────────────────────────────────────────
def test_key_is_stable():
    a = shap_cache_key(MODEL_KEY, HEAD, SEQ, NSAMPLES, N_BACKGROUND)
    b = shap_cache_key(MODEL_KEY, HEAD, SEQ, NSAMPLES, N_BACKGROUND)
    assert a == b


@pytest.mark.parametrize("changed", [
    {"model_key": "A/SpCas9"},
    {"head": "SpCas9-NG"},
    {"window_seq": "C" + SEQ[1:]},
    {"nsamples": 128},            # 샘플 수만 달라도 다른 키여야 한다
    {"n_background": 8},          # 배경 수만 달라도 다른 키여야 한다
])
def test_key_changes_with_every_field(changed):
    base = dict(model_key=MODEL_KEY, head=HEAD, window_seq=SEQ,
                nsamples=NSAMPLES, n_background=N_BACKGROUND)
    assert shap_cache_key(**base) != shap_cache_key(**{**base, **changed})


def test_key_accepts_int_like_settings():
    """app.py가 number_input 값을 int()로 넘긴다. str/int 혼용이 키를 가르면 안 된다."""
    assert (shap_cache_key(MODEL_KEY, HEAD, SEQ, 64, 4)
            == shap_cache_key(MODEL_KEY, HEAD, SEQ, "64", "4"))


# ── 저장소 조회 ──────────────────────────────────────────────────
def test_load_hit(cache_root):
    got = shap_cache.load(shap_cache_key(MODEL_KEY, HEAD, SEQ, NSAMPLES, N_BACKGROUND),
                          root=cache_root)
    assert got is not None
    assert got.values.shape == (34,)
    assert got.base_value == pytest.approx(0.3)


def test_load_miss_returns_none(cache_root):
    assert shap_cache.load(shap_cache_key(MODEL_KEY, HEAD, SEQ, 128, N_BACKGROUND),
                           root=cache_root) is None


# ── kernel_shap 통합 (가중치 없이) ────────────────────────────────
def test_kernel_shap_hit_skips_inference(cache_root):
    handle = _SpyHandle()
    got = explain.kernel_shap(handle, SEQ, nsamples=NSAMPLES, n_background=N_BACKGROUND)
    assert got.base_value == pytest.approx(0.3)
    assert handle.calls == 0, "적중했는데 추론이 돌았다"


def test_kernel_shap_miss_falls_through_to_compute(cache_root):
    """미적중이면 계산 경로로 간다. 가짜 handle이라 끝까지는 못 가도, 추론은 시작돼야 한다."""
    handle = _SpyHandle()
    try:
        explain.kernel_shap(handle, SEQ, nsamples=128, n_background=N_BACKGROUND)
    except Exception:
        pass
    assert handle.calls > 0, "미적중인데 캐시 값을 돌려줬다"
