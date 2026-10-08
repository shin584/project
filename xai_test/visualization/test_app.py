"""Streamlit `AppTest` rendering tests for the dashboard (issue #20).

Each representative sample must render every tab without an exception and
show its required elements: a Case Study sample with IG, a sample without
IG, and the boundary ids 0 and 513.
"""

from pathlib import Path

import pytest
from data_source import PHYSICAL_FEATURE_LABELS, PrecomputedDataSource
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent / "app.py")
IG_SAMPLE_ID = 216  # a Case Study sample (IG available)
NO_IG_SAMPLE_ID = 1

TAB_LABELS = ["최고 후보군", "위치별 히트맵", "예측 상세", "XAI 분석"]
FORBIDDEN_UI_TERMS = ("Model A", "Concordant", "Discordant", "case_id", "Case Study")


@pytest.fixture(scope="module")
def ig_ids():
    return set(PrecomputedDataSource().ig_sample_ids())


def run_app(sample_id: int) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.sidebar.selectbox(key="sample_id").set_value(sample_id).run()
    return at


def all_text(at: AppTest) -> str:
    parts = [e.value for e in at.markdown]
    parts += [e.value for e in at.caption]
    parts += [e.value for e in at.info]
    parts += [e.value for e in at.warning]
    parts += [e.label for e in at.expander]
    parts += [e.label for e in at.tabs]
    parts += [str(e.value) for e in at.metric]
    parts += [e.label for e in at.metric]
    parts += [e.value for e in at.subheader]
    parts += [e.value for e in at.title]
    return "\n".join(str(p) for p in parts)


def test_fixture_samples_have_the_expected_ig_status(ig_ids):
    assert IG_SAMPLE_ID in ig_ids
    assert NO_IG_SAMPLE_ID not in ig_ids


@pytest.mark.parametrize("sample_id", [IG_SAMPLE_ID, NO_IG_SAMPLE_ID, 0, 513])
def test_every_tab_renders_for_representative_samples(sample_id, ig_ids):
    at = run_app(sample_id)

    assert not at.exception
    assert [t.label for t in at.tabs] == TAB_LABELS
    text = all_text(at)

    # Tab 1: ranking table.
    assert len(at.dataframe) >= 1
    # Tab 2/4: ISM heatmap, complex mismatch bars, rollout profile, SHAP
    # waterfall, rollout heatmap (+ attribution logo when IG exists).
    expected_figures = 5 + (1 if sample_id in ig_ids else 0)
    assert len(at.get("imgs")) == expected_figures
    # Tab 3: detail table with truth/error and the 4 physical features.
    assert len(at.table) == 1
    detail = at.table[0].value
    for label in (
        "Model B 예측값",
        "실측값",
        "오차",
        *PHYSICAL_FEATURE_LABELS.values(),
    ):
        assert label in detail.index
    # Tab 4 caveats.
    assert "물리적 결합이 아님" in text
    assert "[CLS]" in text
    if sample_id in ig_ids:
        assert "Deterministic Projection Rule" in text
        assert "IG 미계산" not in text
    else:
        assert "IG 미계산" in text

    for term in FORBIDDEN_UI_TERMS:
        assert term not in text


def test_selector_offers_the_whole_testset():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    options = at.sidebar.selectbox(key="sample_id").options
    assert len(options) == 514
