"""Streamlit 화면이 XAI 메모리 캐시(#182)를 쓰는지 AppTest로 확인한다. 가중치 없이 돈다.

확인하는 것
  1) 그래프 크기 슬라이더를 움직여도 XAI가 다시 계산되지 않는다
     (크기는 session_state에서 그릴 때만 읽으므로 키 재료가 아니다)
  2) 계산 설정(IG 단계 수)을 바꾸면 다시 계산된다 — 캐시가 설정을 무시하면
     화면의 숫자를 믿을 수 없다

캐시의 키 계약 자체는 test/test_explain_cache.py에서 본다.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from xai import explain_cache  # noqa: E402

SEQ = "ACGTACGTAGGCTTACGATCGATCGGATCCAT"

SCRIPT = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from xai import view

SEQ = {SEQ!r}


# NOTE: this script is written to a temp file with the locale encoding
# (cp949 here), so keep it ASCII. Korean notes belong in the test module.
class FakeHandle:
    \"\"\"A minimal IG handle. No model weights needed.\"\"\"

    def ig_inputs(self, window_seq):
        return [np.arange(len(window_seq), dtype=np.float64) / len(window_seq)]

    def ig_grad_fn(self, batch):
        x = np.asarray(batch[0], dtype=np.float64)
        return x.sum(axis=1), [np.ones_like(x)]

    def project(self, attrs):
        return np.asarray(attrs[0], dtype=np.float64)


class FakePredictor:
    def explain_handle(self, head):
        st.session_state["handle_calls"] = st.session_state.get("handle_calls", 0) + 1
        return FakeHandle()


# Same widget keys as app.py's sidebar: show_figure reads fig_pct.
st.slider("graph size", 10, 100, 80, step=10, key="fig_pct")
steps = st.number_input("IG steps", 8, 256, 8, step=8, key="ig_steps")

result = view.compute_ig(FakePredictor(), "A/Cas12a", "Cas12a", SEQ, int(steps))
view.show_figure(plt.figure())

st.text(f"handle_calls={{st.session_state.get('handle_calls', 0)}}")
st.text(f"pred={{result.pred:.6f}}")
"""


def _texts(app):
    values = {}
    for text in app.text:
        key, _, value = text.value.partition("=")
        values[key] = value
    return values


@pytest.fixture
def app():
    explain_cache.CACHE.clear()      # 다른 테스트가 남긴 항목과 섞이지 않게
    try:
        yield AppTest.from_string(SCRIPT, default_timeout=60).run()
    finally:
        explain_cache.CACHE.clear()


def test_the_first_run_computes_once(app):
    assert not app.exception, app.exception
    assert _texts(app)["handle_calls"] == "1"


def test_moving_the_graph_size_slider_does_not_recompute(app):
    first = _texts(app)

    app.slider[0].set_value(40).run()

    assert not app.exception, app.exception
    assert app.slider[0].value == 40, "슬라이더가 실제로 움직이지 않았다"
    assert _texts(app)["handle_calls"] == "1", "그래프 크기를 바꿨는데 XAI가 다시 돌았다"
    assert _texts(app)["pred"] == first["pred"]


def test_changing_a_compute_setting_does_recompute(app):
    app.number_input[0].set_value(16).run()

    assert not app.exception, app.exception
    assert _texts(app)["handle_calls"] == "2", "IG 단계 수를 바꿨는데 예전 값을 돌려줬다"
