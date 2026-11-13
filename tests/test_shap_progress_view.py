"""SHAP 진행률 바(#153 7번) 테스트. 가중치 없이 Streamlit AppTest로 돈다.

확인하는 것
  1) 계산이 끝나면 진행률 바가 화면에 남지 않는다
  2) 캐시가 적중해도 0% 바가 남지 않는다 (@st.cache_data가 재생하는 출력까지 치운다)
  3) 콜백 객체가 호출마다 달라도 캐시는 적중한다 (_ 접두사가 키에서 빠지는지)
  4) 캐시 적중값이 처음 계산값과 같다

왜 래퍼가 캐시되면 안 되는지는 xai/view.py의 compute_shap docstring 참고.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
pytest.importorskip("shap")

# 한 번 돌 때마다 모델 추론을 흉내내는 가짜 handle을 쓴다. 실제 가중치는 필요 없다.
SCRIPT = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})

import numpy as np
import streamlit as st
from xai import view as xai_view

BATCH = 8
SEQ = "ACGTACGTAGGCTTACGATCGATCGGATCCAT"


class FakeHandle:
    def raw_fn(self, X, on_batch=None):
        X = np.asarray(X)
        total = X.shape[0]
        out = []
        for i in range(0, total, BATCH):
            c = X[i:i + BATCH]
            out.append(c.reshape(c.shape[0], -1).sum(axis=1) / 100.0)
            if on_batch is not None:
                on_batch(min(i + BATCH, total), total)
        return np.concatenate(out)


class FakePredictor:
    def explain_handle(self, head):
        st.session_state["handle_calls"] = st.session_state.get("handle_calls", 0) + 1
        return FakeHandle()


res = xai_view.compute_shap(FakePredictor(), "A/Cas12a", "Cas12a", SEQ, 64, 4)
st.text(f"base={{res.base_value:.6f}}")
st.text(f"handle_calls={{st.session_state.get('handle_calls', 0)}}")
"""


def _run():
    at = AppTest.from_string(SCRIPT, default_timeout=300).run()
    assert not at.exception, at.exception
    values = {}
    for t in at.text:
        key, _, val = t.value.partition("=")
        values[key] = val
    return at, values


@pytest.fixture(scope="module")
def runs():
    """같은 설정으로 두 번 돌린다. 1회차는 미적중, 2회차는 적중이어야 한다."""
    first = _run()
    second = _run()
    return first, second


def test_no_progress_bar_left_after_compute(runs):
    (at, _), _ = runs
    assert len(at.get("progress")) == 0, "계산이 끝났는데 진행률 바가 남았다"


def test_no_progress_bar_left_on_cache_hit(runs):
    _, (at, _) = runs
    assert len(at.get("progress")) == 0, "캐시 적중인데 0% 바가 남았다"


def test_callback_does_not_break_the_cache(runs):
    """_on_batch는 호출마다 다른 함수 객체다. _ 접두사 덕에 캐시 키에서 빠져야 한다."""
    (_, first), (_, second) = runs
    assert first["handle_calls"] == "1", "1회차에 계산이 안 돌았다"
    assert second["handle_calls"] == "0", \
        "캐시가 빗나갔다 — _on_batch의 _ 접두사를 확인하세요"


def test_cached_value_matches(runs):
    (_, first), (_, second) = runs
    assert first["base"] == second["base"]
