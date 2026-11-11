"""The render foundation refuses a figure that hides where its number came from."""

from __future__ import annotations

import pytest

from xai_test.nt_xai_demo.render import (
    Denominator,
    Figure,
    RenderRefused,
    Section,
    render,
)

CLAIM_LIMIT = "모델 내부 정보 사용을 기술한다. 물리적 결합에 대한 주장이 아니다."


def figure(**overrides) -> Figure:
    fields = {
        "name": "IG-ISM 일치도 (중위수)",
        "value": 0.7,
        "stratum": "generalisation",
        "head": "SpCas9",
        "statistic": "spearman_rank_correlation (magnitude)",
        "denominator": Denominator(rows=200, unit="행", targets=193),
    }
    fields.update(overrides)
    return Figure(**fields)


def test_a_figure_prints_all_four_fields_with_its_value():
    lines = figure().lines()

    assert lines[0] == "IG-ISM 일치도 (중위수): 0.7"
    assert "층=generalisation" in lines[1]
    assert "head=SpCas9" in lines[1]
    assert "통계=spearman_rank_correlation (magnitude)" in lines[1]
    assert "분모=200행·193타깃" in lines[1]


@pytest.mark.parametrize("field", ["name", "stratum", "head", "statistic"])
def test_a_figure_without_one_of_the_four_is_refused(field):
    with pytest.raises(RenderRefused):
        figure(**{field: "  "})


@pytest.mark.parametrize(
    "stratum",
    [
        "generalisation+pam_library",
        "generalisation, pam_library",
        "pooled",
        "두 층 합산",
    ],
)
def test_a_stratum_that_names_more_than_one_layer_is_refused(stratum):
    with pytest.raises(RenderRefused) as refusal:
        figure(stratum=stratum)

    assert "pooling: forbidden" in str(refusal.value)


def test_a_denominator_must_be_a_real_count_with_a_unit():
    with pytest.raises(RenderRefused):
        Denominator(rows=0, unit="행")
    with pytest.raises(RenderRefused):
        Denominator(rows=200, unit=" ")
    with pytest.raises(RenderRefused):
        Denominator(rows=200, unit="행", targets=0)

    assert (
        Denominator(rows=1500, unit="행", note="Test 전수").text()
        == "1,500행 (Test 전수)"
    )
    assert Denominator(rows=200, unit="행", targets=5).text() == "200행·5타깃"


def test_the_claim_limit_is_always_printed_at_the_bottom():
    output = render([Section("성능", [figure()])], claim_limit=CLAIM_LIMIT)

    assert CLAIM_LIMIT in output
    assert output.strip().endswith(CLAIM_LIMIT)
    assert "주장 범위" in output


def test_rendering_without_a_claim_limit_is_refused():
    with pytest.raises(RenderRefused):
        render([Section("성능", [figure()])], claim_limit="   ")
    with pytest.raises(RenderRefused):
        render([], claim_limit=CLAIM_LIMIT)


def test_a_view_can_hold_figures_with_different_denominators_side_by_side():
    output = render(
        [
            Section(
                "IG-ISM 일치도",
                [
                    figure(denominator=Denominator(rows=200, unit="행", targets=193)),
                    figure(
                        stratum="pam_library",
                        value=0.1,
                        denominator=Denominator(rows=200, unit="행", targets=5),
                    ),
                ],
                notes=["두 층은 모집단 모양이 달라 합치지 않는다."],
            )
        ],
        claim_limit=CLAIM_LIMIT,
        title="SpCas9 (9-Hydra v3)",
        questions=["IG와 ISM이 같은 토큰을 중요하다고 보는가?"],
        pointers=["교차 소견은 `cross`"],
    )

    assert "200행·193타깃" in output and "200행·5타깃" in output
    assert "이 분석이 답한 질문" in output
    assert "교차 소견은 `cross`" in output


def test_values_are_formatted_without_inventing_precision():
    assert figure(value=0.70071660544).lines()[0].endswith("0.7007")
    assert figure(value=1500).lines()[0].endswith("1,500")
    assert figure(value=[0.1, 0.25]).lines()[0].endswith("[0.1, 0.25]")
    assert figure(value=True).lines()[0].endswith("예")
